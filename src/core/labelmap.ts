/**
 * Packs per-structure binary masks into one uint8 label volume. A 3D texture
 * per structure does not scale: this case alone would be nine 12 MB textures
 * and nine fetches per raycast sample, against one fetch plus a 256-entry
 * lookup table, where toggling a structure rewrites 1 KB.
 */

import type { LabelVolume, Structure, TypedNumberArray, Volume } from './types';
import { lookupAnatomy } from './anatomy';

export interface MaskInput {
  /** Machine name, e.g. "kidney_left". Used to look up colour and display name. */
  key: string;
  /** Already reoriented to the reference volume's RAS grid. */
  values: TypedNumberArray;
  /**
   * Which value counts as inside; undefined means any non-zero, the binary
   * mask case. A combined multi-label file (values 1..N in one volume, what
   * TotalSegmentator emits by default) becomes N MaskInputs sharing one
   * `values` array. Splitting into N real arrays would cost 12 MB each, so 117
   * structures would need well over a gigabyte to say the same thing.
   */
  matchValue?: number;
}

/** Distinct non-zero values. Stops past `limit`: a continuous-valued volume handed
 *  in by mistake would otherwise build a set with millions of entries. */
export function distinctLabels(values: TypedNumberArray, limit = 255): number[] {
  const seen = new Set<number>();
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === 0) continue;
    if (!Number.isInteger(v)) return [];
    seen.add(v);
    if (seen.size > limit) return [];
  }
  return [...seen].sort((a, b) => a - b);
}

/**
 * One non-zero value is a binary mask named after the file, several is a
 * combined label map. Names fall back to the value itself, so an unrecognised
 * label map still toggles per structure rather than collapsing into one blob.
 */
export function expandMaskFile(
  key: string,
  values: TypedNumberArray,
  labelNames?: Record<number, string>,
): MaskInput[] {
  const labels = distinctLabels(values);
  if (labels.length <= 1) return [{ key, values }];
  return labels.map((value) => ({
    key: labelNames?.[value] ?? `label_${value}`,
    values,
    matchValue: value,
  }));
}

export interface BuildLabelOptions {
  reference: Volume;
  masks: MaskInput[];
}

/**
 * Stats come from each mask before packing, so the UI numbers describe the
 * segmentation rather than what survived overlap resolution: a liver reported
 * at 1573 mL stays 1573 mL even with a vessel drawn on top of it.
 */
export function buildLabelVolume({ reference, masks }: BuildLabelOptions): LabelVolume {
  const [nx, ny, nz] = reference.dims;
  const n = nx * ny * nz;

  if (masks.length > 255) {
    throw new Error(`A label volume holds at most 255 structures, got ${masks.length}`);
  }
  for (const m of masks) {
    if (m.values.length !== n) {
      throw new Error(
        `Mask "${m.key}" has ${m.values.length} voxels but the CT has ${n}. ` +
        `Masks must be on the same grid as the scan.`,
      );
    }
  }

  const values = new Uint8Array(n);
  const structures: Structure[] = [];
  const slope = reference.slope;
  const intercept = reference.intercept;
  const scalar = reference.values;

  const voxelMl = (reference.spacing[0] * reference.spacing[1] * reference.spacing[2]) / 1000;

  const stats = masks.map((mask, i) => {
    let count = 0;
    let sum = 0;
    let i0 = nx, j0 = ny, k0 = nz, i1 = -1, j1 = -1, k1 = -1;
    let ci = 0, cj = 0, ck = 0;

    const v = mask.values;
    const target = mask.matchValue;
    const inside = target === undefined
      ? (x: number) => x !== 0
      : (x: number) => x === target;

    for (let k = 0; k < nz; k++) {
      const kOff = k * nx * ny;
      for (let j = 0; j < ny; j++) {
        const jOff = kOff + j * nx;
        for (let i2 = 0; i2 < nx; i2++) {
          if (!inside(v[jOff + i2])) continue;
          count++;
          sum += scalar[jOff + i2] * slope + intercept;
          ci += i2; cj += j; ck += k;
          if (i2 < i0) i0 = i2;
          if (i2 > i1) i1 = i2;
          if (j < j0) j0 = j;
          if (j > j1) j1 = j;
          if (k < k0) k0 = k;
          if (k > k1) k1 = k;
        }
      }
    }

    const anatomy = lookupAnatomy(mask.key);
    return {
      order: i,
      mask,
      count,
      structure: {
        index: 0,
        key: mask.key,
        name: anatomy.name,
        color: anatomy.color,
        visible: true,
        voxelCount: count,
        volumeMl: count * voxelMl,
        meanHu: count > 0 ? sum / count : 0,
        bounds: (count > 0 ? [i0, j0, k0, i1, j1, k1] : [0, 0, 0, -1, -1, -1]) as Structure['bounds'],
        centroid: (count > 0 ? [ci / count, cj / count, ck / count] : [0, 0, 0]) as Structure['centroid'],
      } satisfies Structure,
    };
  });

  // Indices follow the caller's order so the UI list is stable, but painting
  // goes largest first: a 29 mL aorta swallowed by a 1573 mL liver would
  // vanish, while the liver losing a few hundred voxels to it is invisible.
  stats.forEach((s, i) => { s.structure.index = i + 1; });
  const paintOrder = [...stats].sort((a, b) => b.count - a.count);

  let overlapVoxels = 0;
  for (const s of paintOrder) {
    if (s.count === 0) continue;
    const v = s.mask.values;
    const label = s.structure.index;
    const target = s.mask.matchValue;
    // Bounding box only. Structures are tiny next to the scan (the nine here
    // fill 1.5 of 12.4 million voxels), so a full pass per structure would be
    // most of the work in this function for none of the result.
    const [bi0, bj0, bk0, bi1, bj1, bk1] = s.structure.bounds;
    for (let k = bk0; k <= bk1; k++) {
      const kOff = k * nx * ny;
      for (let j = bj0; j <= bj1; j++) {
        const row = kOff + j * nx;
        for (let i = bi0; i <= bi1; i++) {
          const idx = row + i;
          const raw = v[idx];
          if (target === undefined ? raw === 0 : raw !== target) continue;
          if (values[idx] !== 0) overlapVoxels++;
          values[idx] = label;
        }
      }
    }
  }

  for (const s of stats) structures.push(s.structure);

  return {
    dims: reference.dims,
    spacing: reference.spacing,
    affine: reference.affine,
    values,
    structures,
    overlapVoxels,
  };
}

/** 256x1 RGBA table the shaders index by label value. Alpha is visibility times
 *  opacity, so a hidden structure costs nothing at draw time. */
export function buildLut(structures: Structure[], perStructureOpacity = 1): Uint8Array {
  const lut = new Uint8Array(256 * 4);
  for (const s of structures) {
    if (s.index < 1 || s.index > 255) continue;
    const o = s.index * 4;
    lut[o] = s.color[0];
    lut[o + 1] = s.color[1];
    lut[o + 2] = s.color[2];
    lut[o + 3] = s.visible ? Math.round(255 * perStructureOpacity) : 0;
  }
  return lut;
}

/** Extract a single structure's binary mask from the packed volume. */
export function extractMask(label: LabelVolume, index: number): Uint8Array {
  const out = new Uint8Array(label.values.length);
  const v = label.values;
  for (let i = 0; i < v.length; i++) out[i] = v[i] === index ? 1 : 0;
  return out;
}
