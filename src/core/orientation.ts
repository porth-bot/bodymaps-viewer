/**
 * Voxel-order canonicalisation.
 *
 * NIfTI world space is RAS+ (+x Right, +y Anterior, +z Superior), but the
 * voxel axes can point any which way. Everything downstream assumes canonical
 * RAS voxel order, so exactly one place in the app has to think about this:
 * here.
 *
 * The reorientation is a pure permute-and-flip. It never resamples, so voxel
 * values are preserved bit for bit and an oblique affine stays oblique (its
 * columns are only reordered and negated).
 */

import { mat4Multiply } from './mat4';
import type { AxCodes, AxisCode, Mat4, NiftiImage, TypedNumberArray, Volume } from './types';

/** Letter for increasing index along world axis 0/1/2, positive then negative. */
const POSITIVE_CODES: readonly AxisCode[] = ['R', 'A', 'S'];
const NEGATIVE_CODES: readonly AxisCode[] = ['L', 'P', 'I'];

interface AxisMapping {
  /** worldOf[voxelAxis] = the world axis (0=x, 1=y, 2=z) it aligns with. */
  worldOf: [number, number, number];
  /** +1 when increasing the voxel index moves along +world, -1 otherwise. */
  signOf: [number, number, number];
}

/**
 * Match each voxel axis to a distinct world axis.
 *
 * Taking the per-column argmax independently can hand two voxel axes the same
 * world axis on an oblique affine, which would produce a nonsense code like
 * ['R','R','S']. Assigning the strongest correspondences first and letting the
 * weaker ones take what is left makes the result a permutation by
 * construction.
 */
function axisMapping(affine: Mat4): AxisMapping {
  const candidates: Array<{ voxel: number; world: number; value: number }> = [];
  for (let voxel = 0; voxel < 3; voxel++) {
    for (let world = 0; world < 3; world++) {
      candidates.push({ voxel, world, value: affine[world * 4 + voxel] });
    }
  }
  candidates.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  const worldOf: [number, number, number] = [-1, -1, -1];
  const signOf: [number, number, number] = [1, 1, 1];
  const worldTaken = [false, false, false];
  for (const c of candidates) {
    if (worldOf[c.voxel] >= 0 || worldTaken[c.world]) continue;
    worldOf[c.voxel] = c.world;
    signOf[c.voxel] = c.value < 0 ? -1 : 1;
    worldTaken[c.world] = true;
  }
  return { worldOf, signOf };
}

export function affineToAxCodes(affine: Mat4): AxCodes {
  const { worldOf, signOf } = axisMapping(affine);
  const code = (voxel: number): AxisCode =>
    signOf[voxel] > 0 ? POSITIVE_CODES[worldOf[voxel]] : NEGATIVE_CODES[worldOf[voxel]];
  return [code(0), code(1), code(2)];
}

interface ReorientPlan {
  /** Source voxel axis feeding output axis 0 (R), 1 (A), 2 (S). */
  srcAxis: [number, number, number];
  /** Whether that source axis is traversed backwards. */
  flip: [boolean, boolean, boolean];
  inDims: [number, number, number];
  outDims: [number, number, number];
  axCodes: AxCodes;
  affine: Mat4;
  /** True when the file is already RAS and no data movement is needed. */
  identity: boolean;
}

function planReorient(image: NiftiImage): ReorientPlan {
  const { worldOf, signOf } = axisMapping(image.header.affine);
  const inDims: [number, number, number] = [
    image.header.dim[1] || 1,
    image.header.dim[2] || 1,
    image.header.dim[3] || 1,
  ];

  const srcAxis: [number, number, number] = [0, 0, 0];
  const flip: [boolean, boolean, boolean] = [false, false, false];
  for (let voxel = 0; voxel < 3; voxel++) {
    const out = worldOf[voxel];
    srcAxis[out] = voxel;
    flip[out] = signOf[voxel] < 0;
  }

  const outDims: [number, number, number] = [
    inDims[srcAxis[0]],
    inDims[srcAxis[1]],
    inDims[srcAxis[2]],
  ];

  // Voxel-space change of basis: new index -> old index. Composing it on the
  // right of the old affine keeps every voxel mapping to the same world point.
  const t = new Float64Array(16);
  for (let out = 0; out < 3; out++) {
    const src = srcAxis[out];
    t[src * 4 + out] = flip[out] ? -1 : 1;
    t[src * 4 + 3] = flip[out] ? inDims[src] - 1 : 0;
  }
  t[15] = 1;

  const identity = srcAxis[0] === 0 && srcAxis[1] === 1 && srcAxis[2] === 2 && !flip[0] && !flip[1] && !flip[2];

  return {
    srcAxis,
    flip,
    inDims,
    outDims,
    axCodes: affineToAxCodes(image.header.affine),
    affine: identity ? Float64Array.from(image.header.affine) : mat4Multiply(image.header.affine, t),
    identity,
  };
}

function allocateLike(source: TypedNumberArray, length: number): TypedNumberArray {
  // TypedArray constructors are not related by any TS-visible interface, so
  // the concrete one is taken off the instance.
  const ctor = source.constructor as new (n: number) => TypedNumberArray;
  return new ctor(length);
}

/**
 * Single pass over the output in memory order. The source index is advanced by
 * precomputed per-axis steps, so the inner loop is one add and one copy with
 * no per-voxel branching or multiplication.
 */
function applyPlan(data: TypedNumberArray, plan: ReorientPlan): TypedNumberArray {
  if (plan.identity) return data;

  const [ni, nj] = plan.inDims;
  const inStride = [1, ni, ni * nj];
  const step = [0, 0, 0];
  let origin = 0;
  for (let out = 0; out < 3; out++) {
    const src = plan.srcAxis[out];
    step[out] = plan.flip[out] ? -inStride[src] : inStride[src];
    if (plan.flip[out]) origin += (plan.inDims[src] - 1) * inStride[src];
  }

  const [oi, oj, ok] = plan.outDims;
  const out = allocateLike(data, oi * oj * ok);
  let w = 0;
  for (let k = 0; k < ok; k++) {
    const kBase = origin + k * step[2];
    for (let j = 0; j < oj; j++) {
      let src = kBase + j * step[1];
      for (let i = 0; i < oi; i++) {
        out[w++] = data[src];
        src += step[0];
      }
    }
  }
  return out;
}

/** mm per voxel along each output axis, i.e. the affine's column norms. */
function columnNorms(affine: Mat4): [number, number, number] {
  const norm = (col: number): number =>
    Math.hypot(affine[col], affine[4 + col], affine[8 + col]);
  return [norm(0), norm(1), norm(2)];
}

function rescaledRange(values: TypedNumberArray, slope: number, intercept: number): [number, number] {
  let rawMin = Infinity;
  let rawMax = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    // Derived float maps carry both NaN and +/-Infinity, and both have to go:
    // min/max feed the window/level autoscaling, so one infinity there flattens
    // the whole display. Comparison alone would only drop the NaN.
    if (!Number.isFinite(v)) continue;
    if (v < rawMin) rawMin = v;
    if (v > rawMax) rawMax = v;
  }
  if (rawMin > rawMax) return [0, 0];
  const a = rawMin * slope + intercept;
  const b = rawMax * slope + intercept;
  // A negative slope inverts the ordering, which happens in some PET files.
  return a <= b ? [a, b] : [b, a];
}

export function reorientToRAS(image: NiftiImage): Volume {
  const plan = planReorient(image);
  const values = applyPlan(image.data, plan);
  const { sclSlope: slope, sclInter: intercept } = image.header;
  const spacing = columnNorms(plan.affine);
  const [min, max] = rescaledRange(values, slope, intercept);

  return {
    dims: plan.outDims,
    spacing,
    affine: plan.affine,
    originalAxCodes: plan.axCodes,
    values,
    slope,
    intercept,
    min,
    max,
    // Field of view, so a bounding box drawn from it encloses whole voxels.
    extent: [plan.outDims[0] * spacing[0], plan.outDims[1] * spacing[1], plan.outDims[2] * spacing[2]],
  };
}

const GRID_TOLERANCE_MM = 1e-3;

/**
 * Reorient a mask onto the grid of an already-canonical reference volume.
 *
 * Returns bare values rather than a Volume because a mask has no meaningful
 * window, spacing of its own, or intensity range: it borrows all of that from
 * the reference it is being drawn over.
 */
export function reorientLike(image: NiftiImage, reference: Volume): TypedNumberArray {
  const plan = planReorient(image);

  if (
    plan.outDims[0] !== reference.dims[0] ||
    plan.outDims[1] !== reference.dims[1] ||
    plan.outDims[2] !== reference.dims[2]
  ) {
    throw new Error(
      `Grid mismatch: mask is ${plan.outDims.join('x')} after reorientation but the reference ` +
        `volume is ${reference.dims.join('x')}`,
    );
  }

  let worst = 0;
  let worstIndex = -1;
  for (let i = 0; i < 16; i++) {
    const diff = Math.abs(plan.affine[i] - reference.affine[i]);
    if (Number.isNaN(diff)) {
      // NaN loses every comparison, so it needs its own arm: left to the
      // accumulator below, the affine that should be rejected hardest (one that
      // cannot be compared at all) would leave `worst` at 0 and pass as an
      // exact match, and the mask would be drawn at an unknown position.
      throw new Error(
        `Grid mismatch: mask affine holds ${plan.affine[i]} at row ${Math.floor(i / 4)} col ` +
          `${i % 4} against the reference's ${reference.affine[i]}, so the mask's world ` +
          `placement cannot be compared. The mask header is malformed.`,
      );
    }
    if (diff > worst) {
      worst = diff;
      worstIndex = i;
    }
  }
  if (worst > GRID_TOLERANCE_MM) {
    throw new Error(
      `Grid mismatch: mask affine differs from the reference by ${worst.toExponential(3)} at ` +
        `row ${Math.floor(worstIndex / 4)} col ${worstIndex % 4}, above the ${GRID_TOLERANCE_MM} mm ` +
        `tolerance. The mask was probably produced on a different scan.`,
    );
  }

  return applyPlan(image.data, plan);
}
