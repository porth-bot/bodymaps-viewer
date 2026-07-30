/**
 * GLSL ES 3.00 shader sources.
 *
 * Every pass works in volume local millimetres: origin at the corner of voxel
 * (0,0,0), box spanning [0, dims*spacing], so a texture coordinate is just
 * position/extent. Patient world coordinates would work too, but the sample
 * case has its origin at -417 mm and ray positions that far out lose enough
 * mantissa to stipple the raycast.
 */

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler3D;
precision highp usampler3D;
precision highp sampler2D;
`;

/**
 * Shared sampling helpers. u_huMin/u_huRange undo the [0,1] normalisation the
 * upload applied, so window/level always happens in real Hounsfield units.
 */
const COMMON = `
uniform sampler3D u_volume;
uniform usampler3D u_label;
uniform sampler2D  u_lut;

uniform float u_huMin;
uniform float u_huRange;
uniform float u_level;
uniform float u_window;

// x = fill opacity, y = 1.0 when outline mode is on, z = outline thickness in voxels
uniform vec3  u_labelStyle;
uniform vec3  u_dims;

float sampleHU(vec3 uvw) {
  return u_huMin + texture(u_volume, uvw).r * u_huRange;
}

float windowed(float hu) {
  float lo = u_level - 0.5 * u_window;
  return clamp((hu - lo) / max(u_window, 1e-6), 0.0, 1.0);
}

// For deciding what to keep, never for what to display. A lung window starts at
// -1350 HU, below anything a CT records, so air scores 0.23 on the display ramp
// instead of 0 and every "is this air" test quietly stops working. Pinning the
// bottom to the data floor fixes that.
float opacityRamp(float hu) {
  float lo = max(u_level - 0.5 * u_window, u_huMin);
  float hi = max(u_level + 0.5 * u_window, lo + 1e-6);
  return clamp((hu - lo) / (hi - lo), 0.0, 1.0);
}

uint labelAt(vec3 uvw) {
  return texture(u_label, uvw).r;
}

// rgb is the structure colour, a is visibility times opacity. Both live in a
// 256x1 LUT, so toggling a structure rewrites 1 KB and never the label volume.
vec4 lutLookup(uint label) {
  return texelFetch(u_lut, ivec2(int(label), 0), 0);
}
`;

// ---------------------------------------------------------------------------
// 2D multiplanar reformat
// ---------------------------------------------------------------------------

/**
 * One origin plus two edge vectors in texture space, so axial, coronal and
 * sagittal share this shader and an oblique plane would need no change. The CPU
 * bakes the radiological convention into the vectors.
 */
export const SLICE_VERT = `${HEADER}
layout(location = 0) in vec2 a_uv;

uniform vec2 u_clipScale;
uniform vec2 u_clipOffset;
uniform vec3 u_texOrigin;
uniform vec3 u_texU;
uniform vec3 u_texV;

out vec3 v_uvw;
out vec2 v_planeUv;

void main() {
  v_uvw = u_texOrigin + a_uv.x * u_texU + a_uv.y * u_texV;
  v_planeUv = a_uv;
  gl_Position = vec4(a_uv * u_clipScale + u_clipOffset, 0.0, 1.0);
}
`;

export const SLICE_FRAG = `${HEADER}
${COMMON}
in vec3 v_uvw;
in vec2 v_planeUv;
out vec4 fragColor;

uniform vec3 u_texU;
uniform vec3 u_texV;

void main() {
  // The quad covers exactly [0,1], so this only fires on float error at the
  // border. Without the epsilon that error draws a black frame around the slice.
  const float EPS = 1e-4;
  if (any(lessThan(v_uvw, vec3(-EPS))) || any(greaterThan(v_uvw, vec3(1.0 + EPS)))) {
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  float g = windowed(sampleHU(v_uvw));
  vec3 rgb = vec3(g);

  uint lbl = labelAt(v_uvw);
  if (lbl > 0u) {
    vec4 c = lutLookup(lbl);
    if (c.a > 0.0) {
      float alpha = c.a * u_labelStyle.x;

      if (u_labelStyle.y > 0.5) {
        // Outline mode: keep the pixel only where the label changes within the
        // plane, so the anatomy underneath stays fully visible. Stepping along
        // the plane's own basis vectors means this works for any orientation.
        vec3 du = u_texU / max(dot(abs(u_texU), u_dims), 1.0) * u_labelStyle.z;
        vec3 dv = u_texV / max(dot(abs(u_texV), u_dims), 1.0) * u_labelStyle.z;
        bool edge =
          labelAt(v_uvw + du) != lbl || labelAt(v_uvw - du) != lbl ||
          labelAt(v_uvw + dv) != lbl || labelAt(v_uvw - dv) != lbl;
        // Still scaled by the global overlay gate. Dropping it here made
        // outline mode ignore both "Show structures" and the opacity slider,
        // so turning structures off left the contours on screen.
        alpha = edge ? c.a * u_labelStyle.x : 0.0;
      }
      rgb = mix(rgb, c.rgb, alpha);
    }
  }

  fragColor = vec4(rgb, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Slice planes drawn inside the 3D view
// ---------------------------------------------------------------------------

export const PLANE3D_VERT = `${HEADER}
layout(location = 0) in vec2 a_uv;

uniform mat4 u_mvp;
uniform vec3 u_p0;
uniform vec3 u_pu;
uniform vec3 u_pv;
uniform vec3 u_extent;

out vec3 v_uvw;

void main() {
  vec3 pos = u_p0 + a_uv.x * u_pu + a_uv.y * u_pv;
  v_uvw = pos / u_extent;
  gl_Position = u_mvp * vec4(pos, 1.0);
}
`;

export const PLANE3D_FRAG = `${HEADER}
${COMMON}
in vec3 v_uvw;
out vec4 fragColor;

uniform float u_airCutoff;

void main() {
  const float EPS = 1e-4;
  if (any(lessThan(v_uvw, vec3(-EPS))) || any(greaterThan(v_uvw, vec3(1.0 + EPS)))) discard;

  float hu = sampleHU(v_uvw);
  float g = windowed(hu);
  uint lbl = labelAt(v_uvw);

  // Drop air, or the three planes read as opaque black cards hiding everything
  // behind them. Tested against the clamped ramp so it still means "air" under
  // a lung window.
  if (opacityRamp(hu) < u_airCutoff && lbl == 0u) discard;

  vec3 rgb = vec3(g);
  if (lbl > 0u) {
    vec4 c = lutLookup(lbl);
    rgb = mix(rgb, c.rgb, c.a * u_labelStyle.x);
  }
  fragColor = vec4(rgb, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Segmentation surfaces
// ---------------------------------------------------------------------------

export const MESH_VERT = `${HEADER}
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normalMatrix;

out vec3 v_worldPos;
out vec3 v_normal;

void main() {
  v_worldPos = (u_model * vec4(a_position, 1.0)).xyz;
  v_normal = u_normalMatrix * a_normal;
  gl_Position = u_mvp * vec4(a_position, 1.0);
}
`;

/**
 * Two-sided Blinn-Phong with a rim term. Two-sided because a slice plane
 * cutting an organ exposes the inside of the surface, which a one-sided model
 * renders black. The rim is what keeps two similar-coloured organs apart when
 * they overlap.
 */
export const MESH_FRAG = `${HEADER}
in vec3 v_worldPos;
in vec3 v_normal;
out vec4 fragColor;

uniform vec3  u_color;
uniform float u_opacity;
uniform vec3  u_cameraPos;

void main() {
  vec3 N = normalize(v_normal);
  vec3 V = normalize(u_cameraPos - v_worldPos);
  if (dot(N, V) < 0.0) N = -N;

  // Headlight plus a cool fill from below. Reads as depth, no light rig needed.
  vec3 L1 = V;
  vec3 L2 = normalize(vec3(-0.4, -0.7, 0.3));

  float d1 = max(dot(N, L1), 0.0);
  float d2 = max(dot(N, L2), 0.0);
  vec3 H = normalize(L1 + V);
  float spec = pow(max(dot(N, H), 0.0), 48.0) * 0.35;

  float rim = pow(1.0 - max(dot(N, V), 0.0), 2.5);

  vec3 lit = u_color * (0.22 + 0.72 * d1 + 0.18 * d2 * vec3(0.75, 0.82, 1.0))
           + vec3(spec)
           + u_color * rim * 0.45;

  // Fresnel-ish alpha, so a translucent organ keeps a readable silhouette.
  float alpha = clamp(u_opacity * (0.65 + 0.5 * rim), 0.0, 1.0);
  fragColor = vec4(lit, alpha);
}
`;

// ---------------------------------------------------------------------------
// Volume raycasting
// ---------------------------------------------------------------------------

export const RAYCAST_VERT = `${HEADER}
layout(location = 0) in vec2 a_uv;
out vec2 v_uv;
void main() {
  v_uv = a_uv;
  gl_Position = vec4(a_uv * 2.0 - 1.0, 0.0, 1.0);
}
`;

/**
 * Single-pass raycaster.
 *
 * Rays come from unprojecting the fragment and are clipped analytically against
 * the volume box, rather than the usual trick of rendering the cube's front and
 * back faces into two textures. Cheaper, and it still behaves when the camera
 * is inside the volume.
 *
 * Each ray stops at the depth the mesh pass wrote, which is what makes organ
 * surfaces and the volume occlude one another instead of looking like two
 * unrelated images stacked together.
 */
export const RAYCAST_FRAG = `${HEADER}
${COMMON}
in vec2 v_uv;
out vec4 fragColor;

uniform mat4  u_invViewProj;
uniform vec3  u_cameraPos;
uniform vec3  u_extent;
uniform float u_stepMm;
uniform int   u_mode;          // 0 = composite, 1 = MIP
uniform float u_density;
uniform int   u_shade;
uniform float u_labelBoost;    // extra opacity for annotated structures
uniform sampler2D u_sceneColor;
uniform sampler2D u_sceneDepth;

// Ray/axis-aligned-box slab test. Returns false when the ray misses.
bool intersectBox(vec3 ro, vec3 rd, vec3 boxMax, out float t0, out float t1) {
  vec3 invD = 1.0 / rd;
  vec3 tA = (vec3(0.0) - ro) * invD;
  vec3 tB = (boxMax - ro) * invD;
  vec3 tMin = min(tA, tB);
  vec3 tMax = max(tA, tB);
  t0 = max(max(tMin.x, tMin.y), tMin.z);
  t1 = min(min(tMax.x, tMax.y), tMax.z);
  return t1 > max(t0, 0.0);
}

// Central-difference gradient, returned in HU per millimetre.
//
// The caller samples one voxel either side, so each difference spans two voxel
// widths: 2 * spacing millimetres, and spacing is u_extent/u_dims. Dividing by
// u_extent instead (the whole volume span) is off by dims/2 per axis, which is
// both a ~250x magnitude collapse, enough that the gradient-magnitude gate
// below never engages and shading silently does nothing, and a per-axis
// reweighting that tilts every normal toward whichever axis has fewest slices.
vec3 gradient(vec3 uvw, vec3 stepUvw) {
  float dx = sampleHU(uvw + vec3(stepUvw.x, 0.0, 0.0)) - sampleHU(uvw - vec3(stepUvw.x, 0.0, 0.0));
  float dy = sampleHU(uvw + vec3(0.0, stepUvw.y, 0.0)) - sampleHU(uvw - vec3(0.0, stepUvw.y, 0.0));
  float dz = sampleHU(uvw + vec3(0.0, 0.0, stepUvw.z)) - sampleHU(uvw - vec3(0.0, 0.0, stepUvw.z));
  return vec3(dx, dy, dz) / (2.0 * u_extent / u_dims);
}

// Warm tissue ramp. Pure greyscale volume rendering reads as fog; biasing the
// low end warm and the high end bone-white gives the depth cue radiologists
// expect from a 3D reconstruction.
vec3 tissueColor(float g) {
  vec3 soft = vec3(0.72, 0.42, 0.33);
  vec3 mid  = vec3(0.90, 0.74, 0.62);
  vec3 bone = vec3(1.00, 0.98, 0.94);
  return g < 0.5 ? mix(soft, mid, g * 2.0) : mix(mid, bone, (g - 0.5) * 2.0);
}

void main() {
  vec4 sceneColor = texture(u_sceneColor, v_uv);
  float sceneDepth = texture(u_sceneDepth, v_uv).r;

  vec2 ndc = v_uv * 2.0 - 1.0;
  vec4 farH = u_invViewProj * vec4(ndc, 1.0, 1.0);
  vec3 farP = farH.xyz / farH.w;
  vec3 rd = normalize(farP - u_cameraPos);
  vec3 ro = u_cameraPos;

  // Distance at which opaque geometry blocks this ray.
  float sceneDist = 1e20;
  if (sceneDepth < 1.0) {
    vec4 dh = u_invViewProj * vec4(ndc, sceneDepth * 2.0 - 1.0, 1.0);
    sceneDist = length(dh.xyz / dh.w - ro);
  }

  float t0, t1;
  if (!intersectBox(ro, rd, u_extent, t0, t1)) {
    fragColor = sceneColor;
    return;
  }
  t0 = max(t0, 0.0);
  t1 = min(t1, sceneDist);
  if (t1 <= t0) {
    fragColor = sceneColor;
    return;
  }

  vec3 stepUvw = 1.0 / u_dims;
  float stepMm = max(u_stepMm, 0.05);
  int maxSteps = int(min(floor((t1 - t0) / stepMm) + 1.0, 2048.0));

  // Jitter the first sample by a screen-space hash. Fixed-offset sampling puts
  // every ray's samples on the same planes, which shows up as concentric wood
  // grain rings; a per-pixel offset turns that into unstructured noise the eye
  // ignores.
  float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  float t = t0 + jitter * stepMm;

  vec3  accum = vec3(0.0);
  float alpha = 0.0;
  float maxG = 0.0;

  for (int i = 0; i < maxSteps; i++) {
    if (t > t1) break;
    vec3 pos = ro + rd * t;
    vec3 uvw = pos / u_extent;

    float hu = sampleHU(uvw);
    float g = windowed(hu);

    if (u_mode == 1) {
      maxG = max(maxG, g);
      t += stepMm;
      continue;
    }

    uint lbl = labelAt(uvw);
    vec4 lc = lbl > 0u ? lutLookup(lbl) : vec4(0.0);
    // Fold the global overlay gate into the per-structure alpha, so switching
    // structures off (or setting overlay opacity to zero) skips the whole
    // block rather than leaving the volume tinted in structure colours.
    float labelAlpha = lc.a * u_labelStyle.x;

    // Opacity ramp.
    //
    // The exponent matters more than it looks. A ray crosses a few hundred
    // samples, and alpha compounds, so a per-sample opacity of even 0.01
    // saturates to 0.97 over 300 steps. A cubic ramp therefore still renders
    // skin and fat as solid white and buries every organ behind them. A
    // fourth-power ramp pushes fat (around 0.12 of a soft-tissue window) down
    // to a few ten-thousandths per sample while leaving bone and contrast near
    // the top, which is what lets you see through the body wall to the
    // anatomy.
    float g2 = g * g;
    float a = g2 * g2 * u_density;
    vec3 c = tissueColor(g);

    if (labelAlpha > 0.0) {
      c = mix(c, lc.rgb, 0.85);
      a = max(a, labelAlpha * u_labelBoost);
    }

    if (a > 0.001) {
      if (u_shade == 1) {
        vec3 grad = gradient(uvw, stepUvw);
        float gl = length(grad);
        if (gl > 1e-5) {
          vec3 N = -grad / gl;
          vec3 V = -rd;
          if (dot(N, V) < 0.0) N = -N;
          float diff = max(dot(N, V), 0.0);
          vec3 H = normalize(V + V);
          float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.25;
          // Blend shading in with gradient magnitude so flat interiors, where
          // the normal is meaningless noise, are not randomly speckled.
          float w = clamp(gl / 400.0, 0.0, 1.0);
          c *= mix(1.0, 0.35 + 0.75 * diff, w);
          c += spec * w;
        }
      }

      // Opacity correction keeps the image identical when the step size
      // changes with the quality slider, instead of everything going
      // translucent at low quality.
      float aCorr = 1.0 - pow(1.0 - clamp(a, 0.0, 1.0), stepMm / 1.0);
      accum += (1.0 - alpha) * aCorr * c;
      alpha += (1.0 - alpha) * aCorr;
      if (alpha > 0.995) break;
    }

    t += stepMm;
  }

  if (u_mode == 1) {
    // Composite the projection over the geometry using its own intensity as
    // alpha, which is the same form the composite branch ends with. Replacing
    // the scene outright wherever maxG exceeded zero erased every organ
    // surface behind the volume, since almost any ray through a body finds
    // some non-zero intensity. Rays already stop at sceneDist, so maxG only
    // covers tissue in front of the surface.
    fragColor = vec4(vec3(maxG) + (1.0 - maxG) * sceneColor.rgb, 1.0);
    return;
  }

  fragColor = vec4(accum + (1.0 - alpha) * sceneColor.rgb, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Utility passes
// ---------------------------------------------------------------------------

export const BLIT_VERT = RAYCAST_VERT;

export const BLIT_FRAG = `${HEADER}
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_texture;
void main() {
  fragColor = vec4(texture(u_texture, v_uv).rgb, 1.0);
}
`;

/** Flat coloured lines, used for the volume bounding box and slice outlines in 3D. */
export const LINE_VERT = `${HEADER}
layout(location = 0) in vec3 a_position;
uniform mat4 u_mvp;
void main() {
  gl_Position = u_mvp * vec4(a_position, 1.0);
}
`;

export const LINE_FRAG = `${HEADER}
out vec4 fragColor;
uniform vec4 u_color;
void main() { fragColor = u_color; }
`;
