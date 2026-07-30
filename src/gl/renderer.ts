/**
 * The renderer. Owns the GL context, every texture and program, and draws all
 * four viewports of a frame.
 *
 * It is deliberately stateless with respect to the UI: `render()` takes the
 * whole scene description each frame. That makes the draw path trivially
 * reproducible from a saved state and removes a whole class of bugs where a
 * toggle updates the UI but not the GPU.
 */

import type { LabelVolume, Mesh, Volume, WindowLevel } from '../core/types';
import { OrbitCamera } from '../camera/orbit';
import {
  createContext, createLabelTexture, createLutTexture, createProgram, createQuad,
  createRenderTarget, createVolumeTexture, resizeRenderTarget, updateLutTexture,
  type Program, type RenderTarget,
} from './gl';
import {
  BLIT_FRAG, BLIT_VERT, LINE_FRAG, LINE_VERT, MESH_FRAG, MESH_VERT,
  PLANE3D_FRAG, PLANE3D_VERT, RAYCAST_FRAG, RAYCAST_VERT, SLICE_FRAG, SLICE_VERT,
} from './shaders';
import { computeLayout, type LayoutMode, type ViewportRect } from './layout';
import {
  clipTransform, computeViewTransform, planeQuad3D, planeTexCoords, PLANES,
  type PlaneSpec, type ViewTransform,
} from './planes';
import { glIdentity, glInvert, glMultiply, glNormalMatrix, type GLMat, type Vec3 } from '../core/mat4';

export type VolumeRenderMode = 'composite' | 'mip';

export interface ViewSettings {
  zoom: number;
  /** Pan in millimetres along the view's screen x and y. */
  pan: [number, number];
}

export interface RenderState {
  layout: LayoutMode;
  /** Continuous voxel coordinates of the crosshair, in RAS voxel order. */
  crosshair: [number, number, number];
  windowLevel: WindowLevel;

  showLabels: boolean;
  labelOpacity: number;
  labelOutline: boolean;
  outlineWidth: number;

  views: Record<string, ViewSettings>;

  show3DVolume: boolean;
  volumeMode: VolumeRenderMode;
  volumeDensity: number;
  volumeShade: boolean;
  /** 0.25 (fast) to 2.0 (fine). Multiplies samples per voxel. */
  volumeQuality: number;
  volumeLabelBoost: number;

  showMeshes: boolean;
  meshOpacity: number;
  showSlicesIn3D: boolean;
  showBoundingBox: boolean;

  camera: OrbitCamera;
  /** Highlighted pane border. */
  activeView: string | null;
}

interface GpuMesh {
  vao: WebGLVertexArrayObject;
  positions: WebGLBuffer;
  normals: WebGLBuffer;
  indices: WebGLBuffer;
  count: number;
  /** Centroid in local mm, for back-to-front sorting of translucent surfaces. */
  centroid: Vec3;
}

const BACKGROUND: [number, number, number] = [0.043, 0.055, 0.075];

export class Renderer {
  readonly gl: WebGL2RenderingContext;
  private canvas: HTMLCanvasElement;

  private progSlice: Program;
  private progPlane3D: Program;
  private progMesh: Program;
  private progRaycast: Program;
  private progBlit: Program;
  private progLine: Program;

  private quad: WebGLVertexArrayObject;
  private lineVao: WebGLVertexArrayObject | null = null;
  private lineBuffer: WebGLBuffer | null = null;

  private volumeTex: WebGLTexture | null = null;
  private labelTex: WebGLTexture | null = null;
  private lutTex: WebGLTexture;

  private target: RenderTarget | null = null;

  private dims: [number, number, number] = [1, 1, 1];
  private spacing: [number, number, number] = [1, 1, 1];
  private extent: [number, number, number] = [1, 1, 1];
  private huMin = 0;
  private huRange = 1;

  private meshes = new Map<number, GpuMesh>();
  private meshColors = new Map<number, [number, number, number]>();
  private meshVisible = new Map<number, boolean>();

  /** Device pixels per CSS pixel, refreshed on resize. */
  private dpr = 1;
  private lastRects: ViewportRect[] = [];
  private superSample = 1;

  private frameTimes: number[] = [];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.gl = createContext(canvas);
    const gl = this.gl;

    this.progSlice = createProgram(gl, 'slice', SLICE_VERT, SLICE_FRAG);
    this.progPlane3D = createProgram(gl, 'plane3d', PLANE3D_VERT, PLANE3D_FRAG);
    this.progMesh = createProgram(gl, 'mesh', MESH_VERT, MESH_FRAG);
    this.progRaycast = createProgram(gl, 'raycast', RAYCAST_VERT, RAYCAST_FRAG);
    this.progBlit = createProgram(gl, 'blit', BLIT_VERT, BLIT_FRAG);
    this.progLine = createProgram(gl, 'line', LINE_VERT, LINE_FRAG);

    this.quad = createQuad(gl);
    this.lutTex = createLutTexture(gl);

    gl.disable(gl.DITHER);
  }

  get hasVolume(): boolean {
    return this.volumeTex !== null;
  }

  get volumeDims(): [number, number, number] {
    return this.dims;
  }

  get volumeExtent(): [number, number, number] {
    return this.extent;
  }

  get viewportRects(): ViewportRect[] {
    return this.lastRects;
  }

  /**
   * Upload a volume. `normalized` carries the same voxels mapped to [0,1] over
   * [min,max]; it is produced in the loader worker so the main thread never
   * walks 12 million voxels during an interaction.
   */
  setVolume(volume: Volume, normalized: Float32Array): void {
    const gl = this.gl;
    if (this.volumeTex) gl.deleteTexture(this.volumeTex);
    this.dims = volume.dims;
    this.spacing = volume.spacing;
    this.extent = volume.extent;
    this.huMin = volume.min;
    this.huRange = Math.max(volume.max - volume.min, 1e-6);
    this.volumeTex = createVolumeTexture(gl, volume.dims, normalized);
  }

  setLabels(labels: LabelVolume | null): void {
    const gl = this.gl;
    if (this.labelTex) {
      gl.deleteTexture(this.labelTex);
      this.labelTex = null;
    }
    if (labels) this.labelTex = createLabelTexture(gl, labels.dims, labels.values);
  }

  /** rgba is 256*4 bytes: per-label colour and (visibility * opacity). */
  setLut(rgba: Uint8Array): void {
    updateLutTexture(this.gl, this.lutTex, rgba);
  }

  setMesh(index: number, mesh: Mesh | null, color: [number, number, number], visible: boolean): void {
    const gl = this.gl;
    const existing = this.meshes.get(index);
    if (existing && (!mesh || mesh.indices.length === 0)) {
      gl.deleteVertexArray(existing.vao);
      gl.deleteBuffer(existing.positions);
      gl.deleteBuffer(existing.normals);
      gl.deleteBuffer(existing.indices);
      this.meshes.delete(index);
    }
    this.meshColors.set(index, color);
    this.meshVisible.set(index, visible);
    if (!mesh || mesh.indices.length === 0) return;

    if (existing) {
      gl.deleteVertexArray(existing.vao);
      gl.deleteBuffer(existing.positions);
      gl.deleteBuffer(existing.normals);
      gl.deleteBuffer(existing.indices);
    }

    const vao = gl.createVertexArray()!;
    const positions = gl.createBuffer()!;
    const normals = gl.createBuffer()!;
    const indices = gl.createBuffer()!;
    gl.bindVertexArray(vao);

    gl.bindBuffer(gl.ARRAY_BUFFER, positions);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, normals);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.normals, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indices);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);

    const b = mesh.bounds;
    this.meshes.set(index, {
      vao, positions, normals, indices,
      count: mesh.indices.length,
      centroid: [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2],
    });
  }

  setMeshAppearance(index: number, color: [number, number, number], visible: boolean): void {
    this.meshColors.set(index, color);
    this.meshVisible.set(index, visible);
  }

  clearMeshes(): void {
    const gl = this.gl;
    for (const m of this.meshes.values()) {
      gl.deleteVertexArray(m.vao);
      gl.deleteBuffer(m.positions);
      gl.deleteBuffer(m.normals);
      gl.deleteBuffer(m.indices);
    }
    this.meshes.clear();
  }

  get meshCount(): number {
    return this.meshes.size;
  }

  /** Sync the drawing buffer to the element size. Returns true if it changed. */
  resize(): boolean {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssW = this.canvas.clientWidth;
    const cssH = this.canvas.clientHeight;
    const w = Math.max(1, Math.round(cssW * dpr));
    const h = Math.max(1, Math.round(cssH * dpr));
    if (this.canvas.width === w && this.canvas.height === h && this.dpr === dpr) return false;
    this.canvas.width = w;
    this.canvas.height = h;
    this.dpr = dpr;
    return true;
  }

  /** Average frame time over the recent window, in milliseconds. */
  get frameTimeMs(): number {
    if (this.frameTimes.length === 0) return 0;
    return this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
  }

  render(state: RenderState): void {
    const t0 = performance.now();
    const gl = this.gl;
    this.resize();

    const cssW = this.canvas.clientWidth;
    const cssH = this.canvas.clientHeight;
    const rects = computeLayout(state.layout, cssW, cssH);
    this.lastRects = rects;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if (!this.volumeTex) {
      this.frameTimes.push(performance.now() - t0);
      if (this.frameTimes.length > 30) this.frameTimes.shift();
      return;
    }

    for (const rect of rects) {
      if (rect.view === 'volume') this.render3D(rect, state);
      else this.renderSlice(rect, state);
    }

    gl.disable(gl.SCISSOR_TEST);
    const dt = performance.now() - t0;
    this.frameTimes.push(dt);
    if (this.frameTimes.length > 30) this.frameTimes.shift();
  }

  /** Device-pixel GL viewport for a CSS-pixel rect, flipping to GL's bottom-left origin. */
  private glViewport(rect: ViewportRect): [number, number, number, number] {
    const d = this.dpr;
    const x = Math.round(rect.x * d);
    const w = Math.round(rect.width * d);
    const h = Math.round(rect.height * d);
    const y = this.canvas.height - Math.round(rect.y * d) - h;
    return [x, y, w, h];
  }

  /**
   * The fit/zoom/pan transform for a slice pane. The overlay and the pointer
   * handling both need this, and both must agree with what was drawn, so it
   * is derived here from the same inputs the draw call uses rather than
   * recomputed independently.
   */
  viewTransformFor(rect: ViewportRect, views: Record<string, ViewSettings>): ViewTransform | null {
    if (rect.view === 'volume') return null;
    const spec = PLANES[rect.view];
    const settings = views[rect.view] ?? { zoom: 1, pan: [0, 0] as [number, number] };
    return computeViewTransform(spec, this.extent, rect.width, rect.height, settings.zoom, settings.pan);
  }

  private bindVolumeUniforms(prog: Program, state: RenderState): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, this.volumeTex);
    gl.uniform1i(prog.uniforms['u_volume'] ?? null, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, this.labelTex);
    gl.uniform1i(prog.uniforms['u_label'] ?? null, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
    gl.uniform1i(prog.uniforms['u_lut'] ?? null, 2);

    gl.uniform1f(prog.uniforms['u_huMin'] ?? null, this.huMin);
    gl.uniform1f(prog.uniforms['u_huRange'] ?? null, this.huRange);
    gl.uniform1f(prog.uniforms['u_level'] ?? null, state.windowLevel.level);
    gl.uniform1f(prog.uniforms['u_window'] ?? null, state.windowLevel.window);
    gl.uniform3f(
      prog.uniforms['u_labelStyle'] ?? null,
      state.showLabels && this.labelTex ? state.labelOpacity : 0,
      state.labelOutline ? 1 : 0,
      state.outlineWidth,
    );
    gl.uniform3f(prog.uniforms['u_dims'] ?? null, this.dims[0], this.dims[1], this.dims[2]);
  }

  private renderSlice(rect: ViewportRect, state: RenderState): void {
    const gl = this.gl;
    const spec: PlaneSpec = PLANES[rect.view as 'axial' | 'coronal' | 'sagittal'];
    const [vx, vy, vw, vh] = this.glViewport(rect);
    if (vw <= 0 || vh <= 0) return;

    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(vx, vy, vw, vh);
    gl.viewport(vx, vy, vw, vh);
    gl.clearColor(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);

    const prog = this.progSlice;
    gl.useProgram(prog.program);
    this.bindVolumeUniforms(prog, state);

    const sliceIndex = state.crosshair[spec.sliceAxis];
    const tc = planeTexCoords(spec, this.dims, sliceIndex);
    gl.uniform3f(prog.uniforms['u_texOrigin'] ?? null, tc.origin[0], tc.origin[1], tc.origin[2]);
    gl.uniform3f(prog.uniforms['u_texU'] ?? null, tc.u[0], tc.u[1], tc.u[2]);
    gl.uniform3f(prog.uniforms['u_texV'] ?? null, tc.v[0], tc.v[1], tc.v[2]);

    const t = computeViewTransform(
      spec, this.extent, rect.width, rect.height,
      state.views[rect.view]?.zoom ?? 1,
      state.views[rect.view]?.pan ?? [0, 0],
    );
    const clip = clipTransform(t);
    gl.uniform2f(prog.uniforms['u_clipScale'] ?? null, clip.scale[0], clip.scale[1]);
    gl.uniform2f(prog.uniforms['u_clipOffset'] ?? null, clip.offset[0], clip.offset[1]);

    gl.bindVertexArray(this.quad);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }

  private render3D(rect: ViewportRect, state: RenderState): void {
    const gl = this.gl;
    const [vx, vy, vw, vh] = this.glViewport(rect);
    if (vw <= 0 || vh <= 0) return;

    // Supersample the offscreen pass a little when quality is high. The final
    // bilinear read then acts as a cheap antialias for the mesh silhouettes,
    // which the default framebuffer's MSAA cannot do for us because the
    // raycast composites over an offscreen target.
    this.superSample = state.volumeQuality >= 1.5 ? 1.5 : 1;
    const fw = Math.round(vw * this.superSample);
    const fh = Math.round(vh * this.superSample);
    this.target = this.target
      ? resizeRenderTarget(gl, this.target, fw, fh)
      : createRenderTarget(gl, fw, fh);

    const aspect = vw / vh;
    const viewProj = state.camera.viewProjection(aspect);
    const camPos = state.camera.position();

    // --- offscreen: opaque geometry -------------------------------------
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.target.fbo);
    gl.viewport(0, 0, this.target.width, this.target.height);
    gl.disable(gl.SCISSOR_TEST);
    gl.clearColor(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 1);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);

    if (state.showSlicesIn3D) this.drawSlicePlanes3D(viewProj, state);
    if (state.showBoundingBox) this.drawBoundingBox(viewProj);
    if (state.showMeshes) this.drawMeshes(viewProj, camPos, state);

    // --- onscreen: volume raycast composited over that -------------------
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(vx, vy, vw, vh);
    gl.viewport(vx, vy, vw, vh);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);

    if (state.show3DVolume) {
      const prog = this.progRaycast;
      gl.useProgram(prog.program);
      this.bindVolumeUniforms(prog, state);

      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, this.target.color);
      gl.uniform1i(prog.uniforms['u_sceneColor'] ?? null, 3);
      gl.activeTexture(gl.TEXTURE4);
      gl.bindTexture(gl.TEXTURE_2D, this.target.depth);
      gl.uniform1i(prog.uniforms['u_sceneDepth'] ?? null, 4);

      gl.uniformMatrix4fv(prog.uniforms['u_invViewProj'] ?? null, false, glInvert(viewProj));
      gl.uniform3f(prog.uniforms['u_cameraPos'] ?? null, camPos[0], camPos[1], camPos[2]);
      gl.uniform3f(prog.uniforms['u_extent'] ?? null, this.extent[0], this.extent[1], this.extent[2]);

      const minSpacing = Math.min(this.spacing[0], this.spacing[1], this.spacing[2]);
      gl.uniform1f(prog.uniforms['u_stepMm'] ?? null, minSpacing / Math.max(state.volumeQuality, 0.1));
      gl.uniform1i(prog.uniforms['u_mode'] ?? null, state.volumeMode === 'mip' ? 1 : 0);
      gl.uniform1f(prog.uniforms['u_density'] ?? null, state.volumeDensity);
      gl.uniform1i(prog.uniforms['u_shade'] ?? null, state.volumeShade ? 1 : 0);
      gl.uniform1f(prog.uniforms['u_labelBoost'] ?? null, state.showLabels ? state.volumeLabelBoost : 0);
      const [near, far] = state.camera.nearFar();
      gl.uniform2f(prog.uniforms['u_near_far'] ?? null, near, far);

      gl.bindVertexArray(this.quad);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindVertexArray(null);
    } else {
      const prog = this.progBlit;
      gl.useProgram(prog.program);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.target.color);
      gl.uniform1i(prog.uniforms['u_texture'] ?? null, 0);
      gl.bindVertexArray(this.quad);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindVertexArray(null);
    }
  }

  private drawSlicePlanes3D(viewProj: GLMat, state: RenderState): void {
    const gl = this.gl;
    const prog = this.progPlane3D;
    gl.useProgram(prog.program);
    this.bindVolumeUniforms(prog, state);
    gl.uniformMatrix4fv(prog.uniforms['u_mvp'] ?? null, false, viewProj);
    gl.uniform3f(prog.uniforms['u_extent'] ?? null, this.extent[0], this.extent[1], this.extent[2]);
    // Just above pure air in the current window. Tying the cutoff to the
    // windowed value rather than a fixed HU means it keeps working on a lung
    // window, where far more of the image is legitimately dark.
    gl.uniform1f(prog.uniforms['u_airCutoff'] ?? null, 0.03);
    gl.bindVertexArray(this.quad);

    for (const view of ['axial', 'coronal', 'sagittal'] as const) {
      const spec = PLANES[view];
      const q = planeQuad3D(spec, this.dims, this.spacing, this.extent, state.crosshair[spec.sliceAxis]);
      gl.uniform3f(prog.uniforms['u_p0'] ?? null, q.p0[0], q.p0[1], q.p0[2]);
      gl.uniform3f(prog.uniforms['u_pu'] ?? null, q.pu[0], q.pu[1], q.pu[2]);
      gl.uniform3f(prog.uniforms['u_pv'] ?? null, q.pv[0], q.pv[1], q.pv[2]);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    gl.bindVertexArray(null);
  }

  private drawMeshes(viewProj: GLMat, camPos: Vec3, state: RenderState): void {
    const gl = this.gl;
    const prog = this.progMesh;

    // Surface nets reports vertices in voxel-index units scaled by spacing, so
    // they sit half a voxel from this renderer's local mm origin, which puts
    // voxel centres at (i+0.5)*spacing. Without this shift the surfaces would
    // float half a voxel off the slice planes they were extracted from.
    const model = glIdentity();
    model[12] = this.spacing[0] * 0.5;
    model[13] = this.spacing[1] * 0.5;
    model[14] = this.spacing[2] * 0.5;
    const mvp = glMultiply(viewProj, model);

    gl.useProgram(prog.program);
    gl.uniformMatrix4fv(prog.uniforms['u_mvp'] ?? null, false, mvp);
    gl.uniformMatrix4fv(prog.uniforms['u_model'] ?? null, false, model);
    gl.uniformMatrix3fv(prog.uniforms['u_normalMatrix'] ?? null, false, glNormalMatrix(model));
    gl.uniform3f(prog.uniforms['u_cameraPos'] ?? null, camPos[0], camPos[1], camPos[2]);

    const opaque = state.meshOpacity >= 0.99;
    const visible = [...this.meshes.entries()].filter(([i]) => this.meshVisible.get(i) !== false);

    if (opaque) {
      gl.disable(gl.BLEND);
      gl.depthMask(true);
      for (const [index, m] of visible) this.drawOneMesh(prog, index, m, 1);
      return;
    }

    // Translucent surfaces need back-to-front order, and must not write depth
    // or they would hide the organs behind them. Leaving depth writes off also
    // means the raycaster's rays pass through them, which is what a
    // semi-transparent surface should do.
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    const sorted = visible.sort((a, b) => {
      const da = distanceSq(a[1].centroid, camPos);
      const db = distanceSq(b[1].centroid, camPos);
      return db - da;
    });
    for (const [index, m] of sorted) this.drawOneMesh(prog, index, m, state.meshOpacity);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }

  private drawOneMesh(prog: Program, index: number, m: GpuMesh, opacity: number): void {
    const gl = this.gl;
    const c = this.meshColors.get(index) ?? [200, 200, 200];
    gl.uniform3f(prog.uniforms['u_color'] ?? null, c[0] / 255, c[1] / 255, c[2] / 255);
    gl.uniform1f(prog.uniforms['u_opacity'] ?? null, opacity);
    gl.bindVertexArray(m.vao);
    gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
  }

  private drawBoundingBox(viewProj: GLMat): void {
    const gl = this.gl;
    if (!this.lineVao) {
      this.lineVao = gl.createVertexArray();
      this.lineBuffer = gl.createBuffer();
    }
    const [ex, ey, ez] = this.extent;
    const c: Vec3[] = [
      [0, 0, 0], [ex, 0, 0], [ex, ey, 0], [0, ey, 0],
      [0, 0, ez], [ex, 0, ez], [ex, ey, ez], [0, ey, ez],
    ];
    const edges = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7];
    const verts = new Float32Array(edges.length * 3);
    edges.forEach((ci, n) => {
      verts[n * 3] = c[ci][0];
      verts[n * 3 + 1] = c[ci][1];
      verts[n * 3 + 2] = c[ci][2];
    });

    gl.bindVertexArray(this.lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const prog = this.progLine;
    gl.useProgram(prog.program);
    gl.uniformMatrix4fv(prog.uniforms['u_mvp'] ?? null, false, viewProj);
    gl.uniform4f(prog.uniforms['u_color'] ?? null, 0.35, 0.45, 0.6, 1);
    gl.drawArrays(gl.LINES, 0, edges.length);
    gl.bindVertexArray(null);
  }

  /** Read back a single pixel of the 3D pass. Used to pick a surface under the pointer. */
  dispose(): void {
    const gl = this.gl;
    this.clearMeshes();
    if (this.volumeTex) gl.deleteTexture(this.volumeTex);
    if (this.labelTex) gl.deleteTexture(this.labelTex);
    gl.deleteTexture(this.lutTex);
    if (this.target) {
      gl.deleteFramebuffer(this.target.fbo);
      gl.deleteTexture(this.target.color);
      gl.deleteTexture(this.target.depth);
    }
  }
}

function distanceSq(a: Vec3, b: Vec3): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}
