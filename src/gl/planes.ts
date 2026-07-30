/**
 * Plane geometry and the screen <-> voxel mapping. The only place that knows the
 * display convention; everything downstream just asks for a plane.
 *
 * Volumes arrive in canonical RAS voxel order (i to the patient's Right, j
 * Anterior, k Superior). Radiological convention falls out of
 * right = cross(viewDirection, up):
 *
 *   axial     seen from the feet    screen right = patient Left,  up = Anterior
 *   coronal   seen from the front   screen right = patient Left,  up = Superior
 *   sagittal  seen from the left    screen right = Posterior,     up = Superior
 */

import type { ViewKind } from '../core/types';

export interface PlaneSpec {
  /** Volume axis (0=i, 1=j, 2=k) along screen x, and whether it runs backwards. */
  axisX: 0 | 1 | 2;
  flipX: boolean;
  axisY: 0 | 1 | 2;
  flipY: boolean;
  /** Axis held constant: the slice normal. */
  sliceAxis: 0 | 1 | 2;
  labels: { left: string; right: string; top: string; bottom: string };
  sliceName: string;
}

export const PLANES: Record<Exclude<ViewKind, 'volume'>, PlaneSpec> = {
  axial: {
    axisX: 0, flipX: true,
    axisY: 1, flipY: false,
    sliceAxis: 2,
    labels: { left: 'R', right: 'L', top: 'A', bottom: 'P' },
    sliceName: 'Axial',
  },
  coronal: {
    axisX: 0, flipX: true,
    axisY: 2, flipY: false,
    sliceAxis: 1,
    labels: { left: 'R', right: 'L', top: 'S', bottom: 'I' },
    sliceName: 'Coronal',
  },
  sagittal: {
    axisX: 1, flipX: true,
    axisY: 2, flipY: false,
    sliceAxis: 0,
    labels: { left: 'A', right: 'P', top: 'S', bottom: 'I' },
    sliceName: 'Sagittal',
  },
};

export interface PlaneTexCoords {
  origin: [number, number, number];
  u: [number, number, number];
  v: [number, number, number];
}

/**
 * Texture-space basis for a slice. The half-voxel offset lands on the centre of
 * slice `index`: on a boundary, linear filtering silently averages two slices.
 */
export function planeTexCoords(
  spec: PlaneSpec,
  dims: [number, number, number],
  index: number,
): PlaneTexCoords {
  const origin: [number, number, number] = [0, 0, 0];
  const u: [number, number, number] = [0, 0, 0];
  const v: [number, number, number] = [0, 0, 0];

  origin[spec.axisX] = spec.flipX ? 1 : 0;
  u[spec.axisX] = spec.flipX ? -1 : 1;

  origin[spec.axisY] = spec.flipY ? 1 : 0;
  v[spec.axisY] = spec.flipY ? -1 : 1;

  const n = dims[spec.sliceAxis];
  origin[spec.sliceAxis] = (Math.max(0, Math.min(n - 1, Math.round(index))) + 0.5) / n;

  return { origin, u, v };
}

/** [width, height] in mm, as laid out on screen. */
export function planeSizeMm(
  spec: PlaneSpec,
  extent: [number, number, number],
): [number, number] {
  return [extent[spec.axisX], extent[spec.axisY]];
}

export interface ViewTransform {
  /** Screen pixels per mm, zoom included. */
  pixelsPerMm: number;
  /** Pixels. */
  drawnW: number;
  drawnH: number;
  /** Quad centre offset from the viewport centre, in pixels, x right and y up. */
  panPxX: number;
  panPxY: number;
  viewportW: number;
  viewportH: number;
}

/**
 * Fit the plane to the viewport, then zoom and pan. The fit is in millimetres,
 * not voxels: at 0.82 x 0.82 x 2.5 mm the coronal plane is 348 x 71 voxels but
 * 284 x 178 mm, so fitting by voxel count squashes the body to a third of its
 * height.
 */
export function computeViewTransform(
  spec: PlaneSpec,
  extent: [number, number, number],
  viewportW: number,
  viewportH: number,
  zoom: number,
  panMm: [number, number],
): ViewTransform {
  const [wMm, hMm] = planeSizeMm(spec, extent);
  const margin = 0.98;
  const base = Math.min(viewportW / Math.max(wMm, 1e-6), viewportH / Math.max(hMm, 1e-6)) * margin;
  const pixelsPerMm = base * zoom;
  return {
    pixelsPerMm,
    drawnW: wMm * pixelsPerMm,
    drawnH: hMm * pixelsPerMm,
    panPxX: panMm[0] * pixelsPerMm,
    panPxY: panMm[1] * pixelsPerMm,
    viewportW,
    viewportH,
  };
}

/** Clip-space scale/offset for the unit quad, matching SLICE_VERT. */
export function clipTransform(t: ViewTransform): { scale: [number, number]; offset: [number, number] } {
  return {
    scale: [(2 * t.drawnW) / t.viewportW, (2 * t.drawnH) / t.viewportH],
    offset: [
      -t.drawnW / t.viewportW + (2 * t.panPxX) / t.viewportW,
      -t.drawnH / t.viewportH + (2 * t.panPxY) / t.viewportH,
    ],
  };
}

/**
 * Pointer position (viewport top-left origin, y downward) to the quad's [0,1]^2.
 * Not clamped: callers want to know when the pointer is off the slice.
 */
export function screenToPlaneUv(t: ViewTransform, localX: number, localY: number): [number, number] {
  const yFromBottom = t.viewportH - localY;
  const x0 = t.viewportW / 2 - t.drawnW / 2 + t.panPxX;
  const y0 = t.viewportH / 2 - t.drawnH / 2 + t.panPxY;
  return [(localX - x0) / Math.max(t.drawnW, 1e-6), (yFromBottom - y0) / Math.max(t.drawnH, 1e-6)];
}

export function planeUvToScreen(t: ViewTransform, u: number, v: number): [number, number] {
  const x0 = t.viewportW / 2 - t.drawnW / 2 + t.panPxX;
  const y0 = t.viewportH / 2 - t.drawnH / 2 + t.panPxY;
  const x = x0 + u * t.drawnW;
  const yFromBottom = y0 + v * t.drawnH;
  return [x, t.viewportH - yFromBottom];
}

/** Voxel under the pointer. The slice axis keeps whatever index it is on. */
export function planeUvToVoxel(
  spec: PlaneSpec,
  dims: [number, number, number],
  sliceIndex: number,
  u: number,
  v: number,
): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  const su = spec.flipX ? 1 - u : u;
  const sv = spec.flipY ? 1 - v : v;
  out[spec.axisX] = su * dims[spec.axisX] - 0.5;
  out[spec.axisY] = sv * dims[spec.axisY] - 0.5;
  out[spec.sliceAxis] = sliceIndex;
  return out;
}

export function voxelToPlaneUv(
  spec: PlaneSpec,
  dims: [number, number, number],
  voxel: [number, number, number],
): [number, number] {
  const su = (voxel[spec.axisX] + 0.5) / dims[spec.axisX];
  const sv = (voxel[spec.axisY] + 0.5) / dims[spec.axisY];
  return [spec.flipX ? 1 - su : su, spec.flipY ? 1 - sv : sv];
}

/** Slice quad in volume local mm, for the three planes drawn in the 3D view. */
export function planeQuad3D(
  spec: PlaneSpec,
  dims: [number, number, number],
  spacing: [number, number, number],
  extent: [number, number, number],
  index: number,
): { p0: [number, number, number]; pu: [number, number, number]; pv: [number, number, number] } {
  const clamped = Math.max(0, Math.min(dims[spec.sliceAxis] - 1, Math.round(index)));
  const p0: [number, number, number] = [0, 0, 0];
  p0[spec.sliceAxis] = (clamped + 0.5) * spacing[spec.sliceAxis];

  // No flips here. Those are a 2D display convention; in 3D the quad is just the
  // physical plane, so both edge vectors run along the positive axes.
  const inPlane = ([0, 1, 2] as const).filter((a) => a !== spec.sliceAxis);
  const pu: [number, number, number] = [0, 0, 0];
  const pv: [number, number, number] = [0, 0, 0];
  pu[inPlane[0]] = extent[inPlane[0]];
  pv[inPlane[1]] = extent[inPlane[1]];
  return { p0, pu, pv };
}
