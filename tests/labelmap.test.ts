import { describe, expect, it } from 'vitest';

import {
  buildLabelVolume, buildLut, distinctLabels, expandMaskFile, extractMask,
  type MaskInput,
} from '../src/core/labelmap';
import { mat4Identity } from '../src/core/mat4';
import type { Volume } from '../src/core/types';

/** A 10x10x10 reference volume with a known linear ramp, 1 mm isotropic. */
function makeReference(spacing: [number, number, number] = [1, 1, 1]): Volume {
  const dims: [number, number, number] = [10, 10, 10];
  const n = dims[0] * dims[1] * dims[2];
  const values = new Int16Array(n);
  for (let i = 0; i < n; i++) values[i] = i % 200;
  const affine = mat4Identity();
  affine[0] = spacing[0];
  affine[5] = spacing[1];
  affine[10] = spacing[2];
  return {
    dims,
    spacing,
    affine,
    originalAxCodes: ['R', 'A', 'S'],
    values,
    slope: 1,
    intercept: 0,
    min: 0,
    max: 199,
    extent: [dims[0] * spacing[0], dims[1] * spacing[1], dims[2] * spacing[2]],
  };
}

const idx = (i: number, j: number, k: number) => k * 100 + j * 10 + i;

describe('distinctLabels', () => {
  it('finds the distinct non-zero values, sorted', () => {
    const v = new Uint8Array([0, 3, 1, 3, 0, 7, 1]);
    expect(distinctLabels(v)).toEqual([1, 3, 7]);
  });

  it('treats an all-zero volume as having no labels', () => {
    expect(distinctLabels(new Uint8Array(50))).toEqual([]);
  });

  it('bails out on non-integer data rather than building a huge set', () => {
    // A continuous-valued volume handed in by mistake must not be mistaken for
    // a label map with millions of structures.
    const v = new Float32Array([0, 0.5, 1.25, 3.75]);
    expect(distinctLabels(v)).toEqual([]);
  });

  it('bails out once the label count exceeds the limit', () => {
    const v = new Uint16Array(600);
    for (let i = 0; i < v.length; i++) v[i] = i + 1;
    expect(distinctLabels(v, 255)).toEqual([]);
  });
});

describe('expandMaskFile', () => {
  it('leaves a binary mask as a single structure named after the file', () => {
    const v = new Uint8Array(1000);
    v[idx(1, 1, 1)] = 1;
    const out = expandMaskFile('liver', v);
    expect(out).toHaveLength(1);
    expect(out[0].key).toBe('liver');
    expect(out[0].matchValue).toBeUndefined();
  });

  it('splits a combined label map into one structure per value', () => {
    const v = new Uint8Array(1000);
    v[idx(1, 1, 1)] = 1;
    v[idx(2, 2, 2)] = 5;
    v[idx(3, 3, 3)] = 5;
    const out = expandMaskFile('segmentations', v);
    expect(out.map((m) => [m.key, m.matchValue])).toEqual([
      ['label_1', 1],
      ['label_5', 5],
    ]);
  });

  it('shares one array across the split rather than copying it per label', () => {
    // The whole point of matchValue: a 117-structure TotalSegmentator output
    // must not allocate 117 full-size arrays.
    const v = new Uint8Array(1000);
    v[idx(1, 1, 1)] = 1;
    v[idx(2, 2, 2)] = 2;
    const out = expandMaskFile('combined', v);
    expect(out).toHaveLength(2);
    expect(out[0].values).toBe(v);
    expect(out[1].values).toBe(out[0].values);
  });

  it('uses supplied names when it has them', () => {
    const v = new Uint8Array(1000);
    v[idx(0, 0, 0)] = 1;
    v[idx(1, 0, 0)] = 2;
    const out = expandMaskFile('combined', v, { 1: 'liver', 2: 'spleen' });
    expect(out.map((m) => m.key)).toEqual(['liver', 'spleen']);
  });
});

describe('buildLabelVolume', () => {
  function fill(v: Uint8Array, value: number, i0: number, i1: number): void {
    for (let k = 0; k < 10; k++) {
      for (let j = 0; j < 10; j++) {
        for (let i = i0; i <= i1; i++) v[idx(i, j, k)] = value;
      }
    }
  }

  it('packs binary masks and computes volume in millilitres', () => {
    const reference = makeReference([2, 2, 2]);
    const a = new Uint8Array(1000);
    fill(a, 1, 0, 1); // 2 x 10 x 10 = 200 voxels
    const masks: MaskInput[] = [{ key: 'liver', values: a }];
    const out = buildLabelVolume({ reference, masks });

    expect(out.structures).toHaveLength(1);
    const s = out.structures[0];
    expect(s.voxelCount).toBe(200);
    // 200 voxels at 8 mm^3 each = 1600 mm^3 = 1.6 mL
    expect(s.volumeMl).toBeCloseTo(1.6, 6);
    expect(s.bounds).toEqual([0, 0, 0, 1, 9, 9]);
    expect(s.centroid[0]).toBeCloseTo(0.5, 6);
  });

  it('handles a combined label map end to end', () => {
    const reference = makeReference();
    const combined = new Uint8Array(1000);
    fill(combined, 1, 0, 2); // 300 voxels
    fill(combined, 4, 5, 5); // 100 voxels
    const out = buildLabelVolume({
      reference,
      masks: expandMaskFile('combined', combined),
    });

    expect(out.structures.map((s) => [s.key, s.voxelCount])).toEqual([
      ['label_1', 300],
      ['label_4', 100],
    ]);
    // Packed indices are 1-based and independent of the source values, so the
    // label written into the volume is the structure index, not the file's 4.
    expect(out.values[idx(5, 0, 0)]).toBe(2);
    expect(out.values[idx(0, 0, 0)]).toBe(1);
    expect(out.overlapVoxels).toBe(0);
  });

  it('lets the smaller structure win where two overlap', () => {
    const reference = makeReference();
    const big = new Uint8Array(1000);
    const small = new Uint8Array(1000);
    fill(big, 1, 0, 8);   // 900 voxels
    fill(small, 1, 4, 4); // 100 voxels, entirely inside big

    const out = buildLabelVolume({
      reference,
      masks: [{ key: 'liver', values: big }, { key: 'aorta', values: small }],
    });

    // Reported statistics describe each mask as authored, not what survived.
    expect(out.structures[0].voxelCount).toBe(900);
    expect(out.structures[1].voxelCount).toBe(100);
    expect(out.overlapVoxels).toBe(100);
    // A 100 voxel vessel swallowed by a 900 voxel organ would vanish, so the
    // small one is painted last and wins the shared voxels.
    expect(out.values[idx(4, 0, 0)]).toBe(2);
    expect(out.values[idx(0, 0, 0)]).toBe(1);
  });

  it('computes mean intensity over the mask only', () => {
    const reference = makeReference();
    const m = new Uint8Array(1000);
    m[idx(0, 0, 0)] = 1; // reference value 0
    m[idx(1, 0, 0)] = 1; // reference value 1
    m[idx(2, 0, 0)] = 1; // reference value 2
    const out = buildLabelVolume({ reference, masks: [{ key: 'x', values: m }] });
    expect(out.structures[0].meanHu).toBeCloseTo(1, 6);
  });

  it('applies slope and intercept when computing mean intensity', () => {
    const reference = makeReference();
    reference.slope = 2;
    reference.intercept = -10;
    const m = new Uint8Array(1000);
    m[idx(1, 0, 0)] = 1; // stored 1 -> 2*1 - 10 = -8
    const out = buildLabelVolume({ reference, masks: [{ key: 'x', values: m }] });
    expect(out.structures[0].meanHu).toBeCloseTo(-8, 6);
  });

  it('reports an empty structure without inventing a bounding box', () => {
    const reference = makeReference();
    const out = buildLabelVolume({ reference, masks: [{ key: 'ghost', values: new Uint8Array(1000) }] });
    const s = out.structures[0];
    expect(s.voxelCount).toBe(0);
    expect(s.volumeMl).toBe(0);
    // An inverted box, so downstream loops iterate zero times rather than
    // meshing a phantom voxel at the origin.
    expect(s.bounds[3]).toBeLessThan(s.bounds[0]);
  });

  it('rejects a mask that is not on the reference grid, with a useful message', () => {
    const reference = makeReference();
    expect(() =>
      buildLabelVolume({ reference, masks: [{ key: 'wrong', values: new Uint8Array(64) }] }),
    ).toThrow(/same grid/i);
  });

  it('rejects more structures than a single-byte label volume can hold', () => {
    const reference = makeReference();
    const masks = Array.from({ length: 256 }, (_, i) => ({
      key: `s${i}`,
      values: new Uint8Array(1000),
    }));
    expect(() => buildLabelVolume({ reference, masks })).toThrow(/255/);
  });
});

describe('buildLut', () => {
  it('writes colour at the structure index and zero alpha when hidden', () => {
    const reference = makeReference();
    const a = new Uint8Array(1000);
    a[idx(0, 0, 0)] = 1;
    const out = buildLabelVolume({ reference, masks: [{ key: 'liver', values: a }] });
    out.structures[0].color = [10, 20, 30];

    const visible = buildLut(out.structures);
    expect([...visible.slice(4, 8)]).toEqual([10, 20, 30, 255]);
    // Index 0 is background and must stay fully transparent.
    expect([...visible.slice(0, 4)]).toEqual([0, 0, 0, 0]);

    out.structures[0].visible = false;
    expect(buildLut(out.structures)[7]).toBe(0);
  });
});

describe('extractMask', () => {
  it('pulls one structure back out of the packed volume', () => {
    const reference = makeReference();
    const combined = new Uint8Array(1000);
    combined[idx(0, 0, 0)] = 1;
    combined[idx(1, 0, 0)] = 2;
    const out = buildLabelVolume({ reference, masks: expandMaskFile('c', combined) });

    const m = extractMask(out, 2);
    expect(m[idx(1, 0, 0)]).toBe(1);
    expect(m[idx(0, 0, 0)]).toBe(0);
    expect(m.reduce((a, b) => a + b, 0)).toBe(1);
  });
});
