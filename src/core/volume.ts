/** Volume utilities: GPU normalisation, sampling, and voxel/world conversion. */

import { applyMat4, mat4Invert, type Vec3 } from './mat4';
import type { AxCodes, LabelVolume, Volume } from './types';

/**
 * Raw voxels to [0,1] over the volume's real-unit range, for upload as a
 * half-float texture; the shader undoes it with u_huMin/u_huRange so
 * window/level still happens in true Hounsfield units. Uploading HU directly
 * would quantise: near 1000 HU a half float steps by about 0.5 HU, where [0,1]
 * is uniformly good to about 0.0005, or 1 HU over a 2000 HU span.
 */
export function normalizeForGpu(volume: Volume): Float32Array {
  const { values, slope, intercept, min, max } = volume;
  const out = new Float32Array(values.length);
  const range = max - min;
  if (range <= 0) return out;
  const invRange = 1 / range;
  // Rescale folded in: one multiply-add per voxel, not two passes over 12 million.
  const a = slope * invRange;
  const b = (intercept - min) * invRange;
  for (let i = 0; i < values.length; i++) out[i] = values[i] * a + b;
  return out;
}

/** Rescaled value (Hounsfield units for CT) at integer voxel coordinates. */
export function sampleVoxel(volume: Volume, i: number, j: number, k: number): number | null {
  const [nx, ny, nz] = volume.dims;
  const x = Math.round(i);
  const y = Math.round(j);
  const z = Math.round(k);
  if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return null;
  return volume.values[z * nx * ny + y * nx + x] * volume.slope + volume.intercept;
}

export function labelAtVoxel(label: LabelVolume, i: number, j: number, k: number): number {
  const [nx, ny, nz] = label.dims;
  const x = Math.round(i);
  const y = Math.round(j);
  const z = Math.round(k);
  if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return 0;
  return label.values[z * nx * ny + y * nx + x];
}

/** Voxel indices to patient world coordinates in mm (RAS). */
export function voxelToWorld(volume: Volume, voxel: Vec3): Vec3 {
  return applyMat4(volume.affine, voxel);
}

export function worldToVoxel(volume: Volume, world: Vec3): Vec3 {
  return applyMat4(mat4Invert(volume.affine), world);
}

/** Volume local millimetres, the space the renderer works in. */
export function voxelToLocalMm(volume: Volume, voxel: Vec3): Vec3 {
  return [
    (voxel[0] + 0.5) * volume.spacing[0],
    (voxel[1] + 0.5) * volume.spacing[1],
    (voxel[2] + 0.5) * volume.spacing[2],
  ];
}

export function clampVoxel(volume: Volume, voxel: Vec3): Vec3 {
  return [
    Math.max(0, Math.min(volume.dims[0] - 1, voxel[0])),
    Math.max(0, Math.min(volume.dims[1] - 1, voxel[1])),
    Math.max(0, Math.min(volume.dims[2] - 1, voxel[2])),
  ];
}

/**
 * Position readout, e.g. "R 42.1  A 13.7  S 88.0". Direction letters rather
 * than signed axes: "L 20" is unambiguous where "x = -20" needs the convention.
 */
export function formatPatientPosition(world: Vec3): string {
  const axis = (v: number, pos: string, neg: string) =>
    `${v >= 0 ? pos : neg} ${Math.abs(v).toFixed(1)}`;
  return `${axis(world[0], 'R', 'L')}   ${axis(world[1], 'A', 'P')}   ${axis(world[2], 'S', 'I')}`;
}

export function describeAxCodes(codes: AxCodes): string {
  return codes.join('');
}

/**
 * Histogram of the rescaled values, for the window/level widget. Air dominates
 * any CT by a wide margin, so callers plot this on a log scale.
 */
export function histogram(volume: Volume, bins = 256): { counts: Uint32Array; min: number; max: number } {
  const counts = new Uint32Array(bins);
  const { values, slope, intercept, min, max } = volume;
  const range = max - min;
  if (range <= 0) return { counts, min, max };
  const scale = (bins - 1) / range;
  // 12 million samples and 1 million give the same shape, and shape is all the widget wants.
  const stride = Math.max(1, Math.floor(values.length / 2_000_000));
  for (let i = 0; i < values.length; i += stride) {
    const v = values[i] * slope + intercept;
    const bin = (v - min) * scale;
    counts[bin < 0 ? 0 : bin > bins - 1 ? bins - 1 : bin | 0]++;
  }
  return { counts, min, max };
}
