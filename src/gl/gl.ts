/**
 * Thin WebGL2 helpers, not an abstraction layer: the renderer still writes plain
 * GL calls. This only kills boilerplate and makes driver failures legible.
 */

export class GLError extends Error {}

export function createContext(canvas: HTMLCanvasElement): WebGL2RenderingContext {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: true,
    depth: true,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    powerPreference: 'high-performance',
    desynchronized: false,
  });
  if (!gl) {
    throw new GLError(
      'WebGL2 is not available in this browser. The viewer needs WebGL2 for 3D textures.',
    );
  }
  return gl;
}

function compile(gl: WebGL2RenderingContext, type: number, source: string, name: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new GLError(`Could not create shader for ${name}`);
  gl.shaderSource(sh, source);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh) ?? '(no log)';
    // Number the source so the driver's "ERROR: 0:57" points at a line.
    const numbered = source
      .split('\n')
      .map((l, i) => `${String(i + 1).padStart(4)} | ${l}`)
      .join('\n');
    gl.deleteShader(sh);
    throw new GLError(`Failed to compile ${name}:\n${log}\n\n${numbered}`);
  }
  return sh;
}

export interface Program {
  program: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation | null>;
  attribs: Record<string, number>;
}

export function createProgram(
  gl: WebGL2RenderingContext,
  name: string,
  vertexSource: string,
  fragmentSource: string,
): Program {
  const vs = compile(gl, gl.VERTEX_SHADER, vertexSource, `${name}.vert`);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fragmentSource, `${name}.frag`);
  const program = gl.createProgram();
  if (!program) throw new GLError(`Could not create program ${name}`);
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) ?? '(no log)';
    gl.deleteProgram(program);
    throw new GLError(`Failed to link ${name}: ${log}`);
  }

  // Reflected, so adding a uniform to a shader needs no matching edit here.
  const uniforms: Record<string, WebGLUniformLocation | null> = {};
  const nUniforms = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number;
  for (let i = 0; i < nUniforms; i++) {
    const info = gl.getActiveUniform(program, i);
    if (!info) continue;
    // Array uniforms are reported as "name[0]"; store them under the bare name.
    const base = info.name.replace(/\[0\]$/, '');
    uniforms[base] = gl.getUniformLocation(program, info.name);
  }
  const attribs: Record<string, number> = {};
  const nAttribs = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES) as number;
  for (let i = 0; i < nAttribs; i++) {
    const info = gl.getActiveAttrib(program, i);
    if (!info) continue;
    attribs[info.name] = gl.getAttribLocation(program, info.name);
  }
  return { program, uniforms, attribs };
}

/**
 * Scalar 3D texture. R16F, not R8: 8 bits quantises a 2000 HU range into 7.8 HU
 * steps, which visibly bands a narrow liver window. Full float doubles the GPU
 * memory for precision nothing can see, and R16F is filterable in core WebGL2,
 * so the raycaster gets trilinear sampling for free.
 */
export function createVolumeTexture(
  gl: WebGL2RenderingContext,
  dims: [number, number, number],
  data: Float32Array,
): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new GLError('Could not create volume texture');
  gl.bindTexture(gl.TEXTURE_3D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  gl.texImage3D(
    gl.TEXTURE_3D, 0, gl.R16F,
    dims[0], dims[1], dims[2], 0,
    gl.RED, gl.FLOAT, data,
  );
  const err = gl.getError();
  if (err !== gl.NO_ERROR) {
    throw new GLError(
      `Uploading a ${dims[0]}x${dims[1]}x${dims[2]} volume texture failed (GL error 0x${err.toString(16)}). ` +
      `The GPU limit for 3D textures here is ${gl.getParameter(gl.MAX_3D_TEXTURE_SIZE)} per side.`,
    );
  }
  gl.bindTexture(gl.TEXTURE_3D, null);
  return tex;
}

/**
 * R8UI. Integer textures cannot be filtered, which is the point: interpolating
 * label 3 and label 7 would invent label 5 along every organ boundary.
 */
export function createLabelTexture(
  gl: WebGL2RenderingContext,
  dims: [number, number, number],
  data: Uint8Array,
): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new GLError('Could not create label texture');
  gl.bindTexture(gl.TEXTURE_3D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  gl.texImage3D(
    gl.TEXTURE_3D, 0, gl.R8UI,
    dims[0], dims[1], dims[2], 0,
    gl.RED_INTEGER, gl.UNSIGNED_BYTE, data,
  );
  const err = gl.getError();
  if (err !== gl.NO_ERROR) {
    throw new GLError(`Uploading the label texture failed (GL error 0x${err.toString(16)})`);
  }
  gl.bindTexture(gl.TEXTURE_3D, null);
  return tex;
}

/**
 * 256x1 RGBA table indexed by label value. Toggling a structure's visibility or
 * opacity rewrites 1 KB instead of touching the 12 MB label volume.
 */
export function createLutTexture(gl: WebGL2RenderingContext): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new GLError('Could not create LUT texture');
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(256 * 4));
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

export function updateLutTexture(gl: WebGL2RenderingContext, tex: WebGLTexture, rgba: Uint8Array): void {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 256, 1, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  gl.bindTexture(gl.TEXTURE_2D, null);
}

/**
 * Offscreen colour and depth for the 3D view. Depth is a texture, not a
 * renderbuffer, because the raycaster samples it to stop each ray at whichever
 * organ surface it hits. Without that, surfaces and volume cannot occlude each
 * other and one of them always wins.
 */
export interface RenderTarget {
  fbo: WebGLFramebuffer;
  color: WebGLTexture;
  depth: WebGLTexture;
  width: number;
  height: number;
}

export function createRenderTarget(gl: WebGL2RenderingContext, width: number, height: number): RenderTarget {
  const w = Math.max(1, width | 0);
  const h = Math.max(1, height | 0);
  const fbo = gl.createFramebuffer();
  const color = gl.createTexture();
  const depth = gl.createTexture();
  if (!fbo || !color || !depth) throw new GLError('Could not create render target');

  gl.bindTexture(gl.TEXTURE_2D, color);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  // Linear on colour: the 3D pass renders at 1.5x, and point-sampling it back
  // down would throw away exactly the samples we paid for.
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.LINEAR);
  for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);

  gl.bindTexture(gl.TEXTURE_2D, depth);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, w, h, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depth, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new GLError(`3D render target is incomplete (status 0x${status.toString(16)})`);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return { fbo, color, depth, width: w, height: h };
}

export function resizeRenderTarget(
  gl: WebGL2RenderingContext,
  target: RenderTarget,
  width: number,
  height: number,
): RenderTarget {
  const w = Math.max(1, width | 0);
  const h = Math.max(1, height | 0);
  if (target.width === w && target.height === h) return target;
  gl.deleteFramebuffer(target.fbo);
  gl.deleteTexture(target.color);
  gl.deleteTexture(target.depth);
  return createRenderTarget(gl, w, h);
}

/** Unit quad in [0,1]^2 as a triangle strip. Every screen-space pass uses it. */
export function createQuad(gl: WebGL2RenderingContext): WebGLVertexArrayObject {
  const vao = gl.createVertexArray();
  const buf = gl.createBuffer();
  if (!vao || !buf) throw new GLError('Could not create quad');
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return vao;
}
