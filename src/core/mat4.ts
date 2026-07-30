/**
 * Minimal 4x4 / 3-vector math.
 *
 * Two storage conventions live in this file and mixing them up is the classic
 * source of silently-wrong rendering, so they are named apart:
 *
 *   Mat4 (Float64Array, ROW-major, m[row*4+col]) is used for anatomical
 *   affines. NIfTI writes its srow_* as rows, so row-major keeps the parser
 *   readable and lets `applyMat4` read like the maths.
 *
 *   GLMat (Float32Array, COLUMN-major) is what WebGL wants for uniforms.
 *   Everything in the `gl` prefix group produces this layout.
 */

import type { Mat4 } from './types';

export type Vec3 = [number, number, number];
export type GLMat = Float32Array;

export function mat4Identity(): Mat4 {
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function mat4From(values: number[]): Mat4 {
  if (values.length !== 16) throw new Error(`mat4From expects 16 values, got ${values.length}`);
  return Float64Array.from(values);
}

/** Row-major multiply: returns a*b. */
export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float64Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
      out[r * 4 + c] = s;
    }
  }
  return out;
}

/** Transform a point (implicit w=1) by a row-major affine. */
export function applyMat4(m: Mat4, p: Vec3): Vec3 {
  return [
    m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
    m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
    m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
  ];
}

/** Transform a direction (implicit w=0), i.e. ignore translation. */
export function applyMat4Dir(m: Mat4, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[4] * v[0] + m[5] * v[1] + m[6] * v[2],
    m[8] * v[0] + m[9] * v[1] + m[10] * v[2],
  ];
}

/**
 * General 4x4 inverse by Gauss-Jordan with partial pivoting.
 *
 * A cofactor expansion would be faster, but affines here are inverted a
 * handful of times at load, never per frame, and pivoting means a nearly
 * singular affine from a malformed file throws instead of producing NaNs
 * that only show up as a black screen.
 */
export function mat4Invert(m: Mat4): Mat4 {
  const a = new Float64Array(32);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) a[r * 8 + c] = m[r * 4 + c];
    a[r * 8 + 4 + r] = 1;
  }
  for (let col = 0; col < 4; col++) {
    let pivot = col;
    for (let r = col + 1; r < 4; r++) {
      if (Math.abs(a[r * 8 + col]) > Math.abs(a[pivot * 8 + col])) pivot = r;
    }
    const pv = a[pivot * 8 + col];
    if (Math.abs(pv) < 1e-12) throw new Error('mat4Invert: matrix is singular');
    if (pivot !== col) {
      for (let c = 0; c < 8; c++) {
        const t = a[col * 8 + c];
        a[col * 8 + c] = a[pivot * 8 + c];
        a[pivot * 8 + c] = t;
      }
    }
    const inv = 1 / a[col * 8 + col];
    for (let c = 0; c < 8; c++) a[col * 8 + c] *= inv;
    for (let r = 0; r < 4; r++) {
      if (r === col) continue;
      const f = a[r * 8 + col];
      if (f === 0) continue;
      for (let c = 0; c < 8; c++) a[r * 8 + c] -= f * a[col * 8 + c];
    }
  }
  const out = new Float64Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[r * 4 + c] = a[r * 8 + 4 + c];
  return out;
}

// --- vector helpers -------------------------------------------------------

export const vsub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const vadd = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const vscale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const vdot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const vlen = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);

export function vcross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function vnorm(a: Vec3): Vec3 {
  const l = vlen(a);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
}

// --- GL (column-major Float32) --------------------------------------------

export function glIdentity(): GLMat {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

/** Column-major multiply: returns a*b in the usual "apply b first" sense. */
export function glMultiply(a: GLMat, b: GLMat): GLMat {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

export function glPerspective(fovyRad: number, aspect: number, near: number, far: number): GLMat {
  const f = 1 / Math.tan(fovyRad / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

export function glOrtho(l: number, r: number, b: number, t: number, n: number, f: number): GLMat {
  const m = new Float32Array(16);
  m[0] = 2 / (r - l);
  m[5] = 2 / (t - b);
  m[10] = -2 / (f - n);
  m[12] = -(r + l) / (r - l);
  m[13] = -(t + b) / (t - b);
  m[14] = -(f + n) / (f - n);
  m[15] = 1;
  return m;
}

export function glLookAt(eye: Vec3, center: Vec3, up: Vec3): GLMat {
  const f = vnorm(vsub(center, eye));
  let s = vcross(f, up);
  if (vlen(s) < 1e-8) {
    // Camera is looking straight along `up`; pick any perpendicular so the
    // view does not collapse when the user orbits through the pole.
    s = vcross(f, Math.abs(f[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]);
  }
  s = vnorm(s);
  const u = vcross(s, f);
  const m = new Float32Array(16);
  m[0] = s[0]; m[4] = s[1]; m[8] = s[2];
  m[1] = u[0]; m[5] = u[1]; m[9] = u[2];
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
  m[12] = -vdot(s, eye);
  m[13] = -vdot(u, eye);
  m[14] = vdot(f, eye);
  m[15] = 1;
  return m;
}

/** Convert a row-major Mat4 to the column-major Float32 layout GL expects. */
export function toGLMat(m: Mat4): GLMat {
  const out = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[c * 4 + r] = m[r * 4 + c];
  return out;
}

export function glInvert(m: GLMat): GLMat {
  const rowMajor = new Float64Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) rowMajor[r * 4 + c] = m[c * 4 + r];
  return toGLMat(mat4Invert(rowMajor));
}

/**
 * Normal matrix: inverse-transpose of the upper-left 3x3, returned as a GL
 * mat3 (column-major, 9 floats). Needed because non-uniform voxel spacing
 * (0.82 x 0.82 x 2.5 mm here) makes the model matrix anisotropic, and naively
 * reusing it for normals tilts the lighting on every oblique surface.
 */
export function glNormalMatrix(model: GLMat): Float32Array {
  const a = new Float64Array(16);
  for (let i = 0; i < 16; i++) a[i] = model[i];
  // Treat the column-major 4x4 as row-major for the inverse, then transpose
  // back; the two flips cancel into exactly the inverse-transpose we want.
  const inv = mat4Invert(a as unknown as Mat4);
  const out = new Float32Array(9);
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) out[c * 3 + r] = inv[c * 4 + r];
  return out;
}
