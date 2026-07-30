import { describe, expect, it } from 'vitest';

import { applyMat4, mat4Invert } from '../src/core/mat4';
import { gunzipIfNeeded, parseNifti } from '../src/core/nifti';
import { affineToAxCodes, reorientLike, reorientToRAS } from '../src/core/orientation';
import { NiftiDataType } from '../src/core/types';
import type { AxCodes, AxisCode, Mat4, NiftiImage, TypedNumberArray } from '../src/core/types';

// See the note in nifti.test.ts: `node:fs` is untyped here, so the specifier
// is hidden from the type graph.
interface NodeFs {
  readFileSync(path: URL): Uint8Array;
}
const FS_MODULE = 'node:fs';
const fs = (await import(FS_MODULE)) as NodeFs;

const CASE_DIR = new URL('../public/data/BDMAP_00000338/', import.meta.url);

function readCaseFile(relative: string): ArrayBuffer {
  const bytes = fs.readFileSync(new URL(relative, CASE_DIR));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

// --- fixtures --------------------------------------------------------------

/** World axis index and sign that each orientation letter names. */
const AXIS_OF: Record<AxisCode, { world: number; sign: number }> = {
  R: { world: 0, sign: 1 },
  L: { world: 0, sign: -1 },
  A: { world: 1, sign: 1 },
  P: { world: 1, sign: -1 },
  S: { world: 2, sign: 1 },
  I: { world: 2, sign: -1 },
};

/** Build the affine of a volume whose voxel axes point along `codes`. */
function affineFor(
  codes: AxCodes,
  spacing: [number, number, number] = [1, 1, 1],
  origin: [number, number, number] = [0, 0, 0],
): Mat4 {
  const m = new Float64Array(16);
  for (let voxel = 0; voxel < 3; voxel++) {
    const { world, sign } = AXIS_OF[codes[voxel]];
    m[world * 4 + voxel] = sign * spacing[voxel];
  }
  for (let world = 0; world < 3; world++) m[world * 4 + 3] = origin[world];
  m[15] = 1;
  return m;
}

function makeImage(
  dims: [number, number, number],
  affine: Mat4,
  data: TypedNumberArray,
  scl: { slope?: number; inter?: number } = {},
): NiftiImage {
  return {
    header: {
      version: 1,
      littleEndian: true,
      dim: [3, dims[0], dims[1], dims[2], 1, 1, 1, 1],
      datatype: NiftiDataType.INT32,
      bitpix: 32,
      pixdim: [1, 1, 1, 1, 1, 1, 1, 1],
      voxOffset: 352,
      sclSlope: scl.slope ?? 1,
      sclInter: scl.inter ?? 0,
      calMin: 0,
      calMax: 0,
      qformCode: 0,
      sformCode: 1,
      spatialUnits: 2,
      description: '',
      intentName: '',
      affine,
      affineSource: 'sform',
    volumeCount: 1,
    },
    data,
  };
}

/** value = i*100 + j*10 + k, so every voxel names its own source index. */
function indexPattern(dims: [number, number, number]): Int32Array {
  const [ni, nj, nk] = dims;
  const out = new Int32Array(ni * nj * nk);
  for (let k = 0; k < nk; k++) {
    for (let j = 0; j < nj; j++) {
      for (let i = 0; i < ni; i++) {
        out[i + ni * (j + nj * k)] = i * 100 + j * 10 + k;
      }
    }
  }
  return out;
}

// --- axis codes ------------------------------------------------------------

describe('affineToAxCodes', () => {
  it('reads the canonical orientations', () => {
    expect(affineToAxCodes(affineFor(['R', 'A', 'S']))).toEqual(['R', 'A', 'S']);
    expect(affineToAxCodes(affineFor(['L', 'P', 'S']))).toEqual(['L', 'P', 'S']);
    expect(affineToAxCodes(affineFor(['L', 'A', 'S']))).toEqual(['L', 'A', 'S']);
    expect(affineToAxCodes(affineFor(['P', 'I', 'L']))).toEqual(['P', 'I', 'L']);
    expect(affineToAxCodes(affineFor(['I', 'R', 'P']))).toEqual(['I', 'R', 'P']);
  });

  it('ignores spacing and origin', () => {
    const affine = affineFor(['L', 'P', 'S'], [0.8, 0.8, 2.5], [-417, -417, 0]);
    expect(affineToAxCodes(affine)).toEqual(['L', 'P', 'S']);
  });

  it('handles an oblique affine whose columns would otherwise collide', () => {
    // Both of the first two columns are dominated by world x. The weaker of
    // the two has to give way, or the code would come out as ['R','R','S'].
    const m = Float64Array.from([
      0.9, 0.8, 0, 0,
      0.1, -0.6, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    expect(affineToAxCodes(m)).toEqual(['R', 'P', 'S']);
  });

  it('resolves a rotated but still axis-dominant affine', () => {
    const theta = 0.15;
    const m = Float64Array.from([
      Math.cos(theta), -Math.sin(theta), 0, 12,
      Math.sin(theta), Math.cos(theta), 0, -4,
      0, 0, -3, 8,
      0, 0, 0, 1,
    ]);
    expect(affineToAxCodes(m)).toEqual(['R', 'A', 'I']);
  });

  it('still returns a permutation for a degenerate affine', () => {
    const codes = affineToAxCodes(Float64Array.from([
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 1,
    ]));
    expect(new Set(codes.map((c) => AXIS_OF[c].world)).size).toBe(3);
  });
});

// --- reorientToRAS ---------------------------------------------------------

describe('reorientToRAS', () => {
  it('leaves an RAS volume alone without copying', () => {
    const dims: [number, number, number] = [2, 3, 4];
    const affine = affineFor(['R', 'A', 'S'], [0.5, 0.5, 2], [-10, -20, -30]);
    const data = indexPattern(dims);
    const volume = reorientToRAS(makeImage(dims, affine, data));

    expect(volume.originalAxCodes).toEqual(['R', 'A', 'S']);
    expect(volume.values).toBe(data);
    expect(volume.dims).toEqual(dims);
    expect(volume.spacing).toEqual([0.5, 0.5, 2]);
    expect(volume.extent).toEqual([1, 1.5, 8]);
    expect(Array.from(volume.affine)).toEqual(Array.from(affine));
    // The affine is a copy, so a later in-place edit cannot corrupt the header.
    expect(volume.affine).not.toBe(affine);
  });

  it('flips an LPS volume into RAS and keeps world positions fixed', () => {
    const dims: [number, number, number] = [2, 3, 4];
    const affine = affineFor(['L', 'P', 'S'], [1, 2, 3], [10, 20, 30]);
    const image = makeImage(dims, affine, indexPattern(dims));
    const volume = reorientToRAS(image);

    expect(volume.originalAxCodes).toEqual(['L', 'P', 'S']);
    expect(affineToAxCodes(volume.affine)).toEqual(['R', 'A', 'S']);
    expect(volume.dims).toEqual([2, 3, 4]);
    expect(volume.spacing).toEqual([1, 2, 3]);
    expect(volume.extent).toEqual([2, 6, 12]);
    expect(Array.from(volume.affine)).toEqual([
      1, 0, 0, 9,
      0, 2, 0, 16,
      0, 0, 3, 30,
      0, 0, 0, 1,
    ]);

    // i and j are reversed, k is untouched.
    const at = (i: number, j: number, k: number): number => volume.values[i + 2 * (j + 3 * k)];
    expect(at(0, 0, 0)).toBe(120); // source voxel (1,2,0)
    expect(at(1, 0, 0)).toBe(20); // source voxel (0,2,0)
    expect(at(0, 2, 0)).toBe(100); // source voxel (1,0,0)
    expect(at(1, 2, 3)).toBe(3); // source voxel (0,0,3)
    expect(at(0, 1, 2)).toBe(112); // source voxel (1,1,2)
  });

  it('permutes axes and preserves the world mapping voxel for voxel', () => {
    // i -> Posterior, j -> Inferior, k -> Left: a full permutation with flips.
    const dims: [number, number, number] = [2, 3, 4];
    const affine = affineFor(['P', 'I', 'L'], [1.5, 2.5, 3.5], [7, -11, 23]);
    const source = indexPattern(dims);
    const volume = reorientToRAS(makeImage(dims, affine, source));

    expect(volume.originalAxCodes).toEqual(['P', 'I', 'L']);
    expect(affineToAxCodes(volume.affine)).toEqual(['R', 'A', 'S']);
    // k (Left) becomes the R axis, i (Posterior) becomes A, j (Inferior) is S.
    expect(volume.dims).toEqual([4, 2, 3]);
    expect(volume.spacing).toEqual([3.5, 1.5, 2.5]);

    // Every output voxel must sit at the same world point as the source voxel
    // it came from, and carry the same value.
    const inverse = mat4Invert(affine);
    const [oi, oj, ok] = volume.dims;
    for (let k = 0; k < ok; k++) {
      for (let j = 0; j < oj; j++) {
        for (let i = 0; i < oi; i++) {
          const world = applyMat4(volume.affine, [i, j, k]);
          const src = applyMat4(inverse, world).map(Math.round) as [number, number, number];
          expect(src.every((v, axis) => v >= 0 && v < dims[axis])).toBe(true);
          const srcIndex = src[0] + dims[0] * (src[1] + dims[1] * src[2]);
          expect(volume.values[i + oi * (j + oj * k)]).toBe(source[srcIndex]);
        }
      }
    }
  });

  it('preserves the source array type', () => {
    const dims: [number, number, number] = [2, 2, 2];
    const data = Int8Array.from([0, 1, 2, 3, 4, 5, 6, 7]);
    const volume = reorientToRAS(makeImage(dims, affineFor(['L', 'A', 'S']), data));
    expect(volume.values).toBeInstanceOf(Int8Array);
    expect(Array.from(volume.values)).toEqual([1, 0, 3, 2, 5, 4, 7, 6]);
  });

  it('reports the rescaled range, not the stored one', () => {
    const dims: [number, number, number] = [2, 2, 2];
    const data = Int32Array.from([-100, 0, 5, 20, 3, 3, 3, 400]);
    const volume = reorientToRAS(
      makeImage(dims, affineFor(['R', 'A', 'S']), data, { slope: 2, inter: 10 }),
    );
    expect(volume.slope).toBe(2);
    expect(volume.intercept).toBe(10);
    expect(volume.min).toBe(-190);
    expect(volume.max).toBe(810);
  });

  it('orders the range correctly for a negative slope', () => {
    const data = Int32Array.from([0, 10, 20, 30, 40, 50, 60, 70]);
    const volume = reorientToRAS(
      makeImage([2, 2, 2], affineFor(['R', 'A', 'S']), data, { slope: -1, inter: 5 }),
    );
    expect(volume.min).toBe(-65);
    expect(volume.max).toBe(5);
  });

  it('skips NaN voxels when ranging float data', () => {
    const data = Float32Array.from([NaN, -3, 7, NaN, 0, 1, 2, 3]);
    const volume = reorientToRAS(makeImage([2, 2, 2], affineFor(['R', 'A', 'S']), data));
    expect(volume.min).toBe(-3);
    expect(volume.max).toBe(7);
  });

  it('skips infinite voxels too, which comparison alone would keep', () => {
    // Derived float maps carry these. An infinity reaching min/max flattens the
    // window/level autoscaling that reads them.
    const data = Float32Array.from([1, 2, Infinity, -Infinity, 3, 4, NaN, 5]);
    const volume = reorientToRAS(makeImage([2, 2, 2], affineFor(['R', 'A', 'S']), data));
    expect(volume.min).toBe(1);
    expect(volume.max).toBe(5);
  });

  it('reports a zero range for an all-NaN volume', () => {
    const data = Float32Array.from([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]);
    const volume = reorientToRAS(makeImage([2, 2, 2], affineFor(['R', 'A', 'S']), data));
    expect([volume.min, volume.max]).toEqual([0, 0]);
  });
});

// --- reorientLike ----------------------------------------------------------

describe('reorientLike', () => {
  const dims: [number, number, number] = [2, 3, 4];
  const affine = affineFor(['L', 'P', 'S'], [1, 2, 3], [10, 20, 30]);
  const reference = reorientToRAS(makeImage(dims, affine, indexPattern(dims)));

  it('lands a mask on the reference grid', () => {
    const mask = new Uint8Array(2 * 3 * 4);
    mask[1 + 2 * (2 + 3 * 0)] = 1; // source voxel (1,2,0), which becomes (0,0,0)
    const out = reorientLike(makeImage(dims, affine, mask), reference);

    expect(out).toBeInstanceOf(Uint8Array);
    expect(out.length).toBe(reference.values.length);
    expect(out[0]).toBe(1);
    expect(Array.from(out).reduce((a, b) => a + b, 0)).toBe(1);
  });

  it('agrees with reorientToRAS on the same data', () => {
    const image = makeImage(dims, affine, indexPattern(dims));
    expect(Array.from(reorientLike(image, reference))).toEqual(Array.from(reference.values));
  });

  it('rejects a mask with different dimensions', () => {
    const other: [number, number, number] = [2, 3, 5];
    const image = makeImage(other, affineFor(['L', 'P', 'S'], [1, 2, 3], [10, 20, 30]), indexPattern(other));
    expect(() => reorientLike(image, reference)).toThrow(/Grid mismatch: mask is 2x3x5/);
  });

  it('rejects a mask from a different scan position', () => {
    const shifted = affineFor(['L', 'P', 'S'], [1, 2, 3], [10, 20, 31]);
    const image = makeImage(dims, shifted, indexPattern(dims));
    expect(() => reorientLike(image, reference)).toThrow(/mask affine differs from the reference/);
  });

  it('rejects a mask affine that cannot be compared to the reference', () => {
    // A worst-difference accumulator never sees a NaN difference, so the one
    // affine that should be rejected hardest (a mask whose world placement is
    // unknown) would otherwise pass as an exact match and be drawn anyway.
    const broken = Float64Array.from(affine);
    broken[3] = NaN;
    expect(() => reorientLike(makeImage(dims, broken, indexPattern(dims)), reference)).toThrow(
      /cannot be compared/,
    );
  });

  it('rejects a mask affine with an infinite origin', () => {
    const broken = Float64Array.from(affine);
    broken[7] = Infinity;
    expect(() => reorientLike(makeImage(dims, broken, indexPattern(dims)), reference)).toThrow(
      /Grid mismatch/,
    );
  });

  it('tolerates sub-micron affine noise', () => {
    const noisy = Float64Array.from(affine);
    noisy[3] += 1e-6;
    const image = makeImage(dims, noisy, indexPattern(dims));
    expect(() => reorientLike(image, reference)).not.toThrow();
  });
});

// --- real data -------------------------------------------------------------

describe('real BDMAP_00000338 case', () => {
  async function loadVolume(relative: string): Promise<NiftiImage> {
    return parseNifti(await gunzipIfNeeded(readCaseFile(relative)));
  }

  it('loads the CT as an RAS volume with the right geometry and range', async () => {
    const image = await loadVolume('ct.nii.gz');
    expect(image.header.dim.slice(1, 4)).toEqual([502, 348, 71]);
    expect(image.header.datatype).toBe(NiftiDataType.INT16);

    const volume = reorientToRAS(image);
    expect(volume.originalAxCodes).toEqual(['R', 'A', 'S']);
    expect(affineToAxCodes(volume.affine)).toEqual(['R', 'A', 'S']);
    expect(volume.dims).toEqual([502, 348, 71]);
    expect(volume.values).toBeInstanceOf(Int16Array);
    expect(volume.values.length).toBe(502 * 348 * 71);

    expect(volume.spacing[0]).toBeCloseTo(0.81641, 5);
    expect(volume.spacing[1]).toBeCloseTo(0.81641, 5);
    expect(volume.spacing[2]).toBeCloseTo(2.5, 9);
    expect(volume.extent[0]).toBeCloseTo(409.84, 1); // 502 voxels of 0.816406 mm
    expect(volume.extent[2]).toBeCloseTo(177.5, 9);

    // The file's slope and intercept map the full int16 range onto exactly
    // -1000..1000 HU.
    expect(volume.min).toBeCloseTo(-1000, 5);
    expect(volume.max).toBeCloseTo(1000, 5);
  });

  it('lines the liver mask up with the CT and reproduces its centroid', async () => {
    const ct = reorientToRAS(await loadVolume('ct.nii.gz'));
    const mask = reorientLike(await loadVolume('segmentations/liver.nii.gz'), ct);

    expect(mask.length).toBe(ct.values.length);

    const [ni, nj] = ct.dims;
    let count = 0;
    let si = 0;
    let sj = 0;
    let sk = 0;
    for (let index = 0; index < mask.length; index++) {
      if (mask[index] === 0) continue;
      count++;
      const i = index % ni;
      const j = Math.floor(index / ni) % nj;
      const k = Math.floor(index / (ni * nj));
      si += i;
      sj += j;
      sk += k;
    }

    expect(count).toBe(944325);
    expect(si / count).toBeCloseTo(344.2, 1);
    expect(sj / count).toBeCloseTo(200.3, 1);
    expect(sk / count).toBeCloseTo(35.2, 1);
  });

  it('lines up every segmentation mask', async () => {
    const ct = reorientToRAS(await loadVolume('ct.nii.gz'));
    const expected: Record<string, number> = {
      aorta: 17600,
      gall_bladder: 10088,
      kidney_left: 64746,
      kidney_right: 65365,
      pancreas: 63802,
      postcava: 27585,
      spleen: 109422,
      stomach: 236819,
    };
    for (const [name, voxels] of Object.entries(expected)) {
      const mask = reorientLike(await loadVolume(`segmentations/${name}.nii.gz`), ct);
      let count = 0;
      for (let i = 0; i < mask.length; i++) if (mask[i] !== 0) count++;
      expect(count, name).toBe(voxels);
    }
  });
});
