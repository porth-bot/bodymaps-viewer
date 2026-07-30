import { gunzipSync } from 'fflate';
import { describe, it, expect } from 'vitest';

import { surfaceNets, computeNormals, meshBounds, extractOrganMesh } from '../src/mesh/surfaceNets';
import { taubinSmooth } from '../src/mesh/smooth';
import { boxBlur3 } from '../src/mesh/blur';
import type { RawMesh } from '../src/mesh/surfaceNets';

const SEGMENTATIONS = '../public/data/BDMAP_00000338/segmentations';
const LIVER_NII = new URL(`${SEGMENTATIONS}/liver.nii.gz`, import.meta.url);
const LIVER_SPACING: [number, number, number] = [0.81641, 0.81641, 2.5];
const LIVER_TRUTH_ML = 1573.5;

/** Voxel counts verified against nibabel, used to prove the reader agrees. */
const ORGAN_VOXELS: ReadonlyArray<readonly [string, number]> = [
  ['aorta', 17600],
  ['gall_bladder', 10088],
  ['kidney_left', 64746],
  ['kidney_right', 65365],
  ['liver', 944325],
  ['pancreas', 63802],
  ['postcava', 27585],
  ['spleen', 109422],
  ['stomach', 236819],
];

// --- helpers --------------------------------------------------------------

/**
 * Signed sphere field: positive inside, so the zero isosurface is the sphere
 * itself and every crossing is exact to first order. Using this instead of a
 * blurred binary mask isolates the meshing from the blur when checking
 * geometric accuracy.
 */
function sphereField(
  n: number,
  radius: number,
  centre: [number, number, number],
  noise = 0,
  seed = 1,
): Float32Array {
  const rng = mulberry32(seed);
  const field = new Float32Array(n * n * n);
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const d = Math.hypot(i - centre[0], j - centre[1], k - centre[2]);
        const jitter = noise > 0 ? (rng() * 2 - 1) * noise : 0;
        field[(k * n + j) * n + i] = radius - d + jitter;
      }
    }
  }
  return field;
}

function sphereMask(n: number, radius: number, centre: [number, number, number]): Uint8Array {
  const mask = new Uint8Array(n * n * n);
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const d = Math.hypot(i - centre[0], j - centre[1], k - centre[2]);
        if (d <= radius) mask[(k * n + j) * n + i] = 1;
      }
    }
  }
  return mask;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Signed volume enclosed by a closed triangle mesh, by the divergence theorem.
 * The sign is the payoff: it is positive only when the winding is outward, so
 * this doubles as the orientation check.
 */
function enclosedVolume(positions: Float32Array, indices: Uint32Array): number {
  let total = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t] * 3;
    const b = indices[t + 1] * 3;
    const c = indices[t + 2] * 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    total += positions[a] * nx + positions[a + 1] * ny + positions[a + 2] * nz;
  }
  return total / 6;
}

interface ManifoldReport {
  edgeCount: number;
  /** Undirected edges not used by exactly one triangle in each direction. */
  badEdges: number;
  /** Edges whose two directions are used a different number of times. */
  unbalancedEdges: number;
  firstBad: string | null;
}

/**
 * The definitive closed-and-consistently-oriented test: build every directed
 * triangle edge and require each undirected edge to be traversed exactly once
 * forwards and once backwards. A boundary shows up as a missing direction, a
 * flipped triangle as two same-direction uses, and a pinch where two surface
 * sheets meet as a balanced count above one.
 *
 * `unbalancedEdges` is the strictly weaker property (closed and consistently
 * oriented, pinches allowed), which is all a decimated preview promises and all
 * that normals and the divergence-theorem volume actually need.
 */
function checkManifold(indices: Uint32Array, vertexCount: number): ManifoldReport {
  // Pack the two direction counts into one number so a single Map carries the
  // whole tally; anything other than 1001 at the end is a defect.
  const uses = new Map<number, number>();
  const add = (u: number, v: number): void => {
    const forward = u < v;
    const key = (forward ? u : v) * vertexCount + (forward ? v : u);
    const prev = uses.get(key) ?? 0;
    uses.set(key, prev + (forward ? 1000 : 1));
  };
  for (let t = 0; t < indices.length; t += 3) {
    add(indices[t], indices[t + 1]);
    add(indices[t + 1], indices[t + 2]);
    add(indices[t + 2], indices[t]);
  }
  let badEdges = 0;
  let unbalancedEdges = 0;
  let firstBad: string | null = null;
  for (const [key, value] of uses) {
    const forward = Math.floor(value / 1000);
    const backward = value % 1000;
    if (forward !== backward) unbalancedEdges++;
    if (value !== 1001) {
      badEdges++;
      if (firstBad === null) {
        const u = Math.floor(key / vertexCount);
        firstBad = `edge ${u}-${key - u * vertexCount} used fwd=${forward} bwd=${backward}`;
      }
    }
  }
  return { edgeCount: uses.size, badEdges, unbalancedEdges, firstBad };
}

function radiusStats(
  positions: Float32Array,
  centre: [number, number, number],
): { mean: number; std: number; min: number; max: number } {
  const n = positions.length / 3;
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  const radii = new Float64Array(n);
  for (let v = 0; v < n; v++) {
    const r = Math.hypot(
      positions[v * 3] - centre[0],
      positions[v * 3 + 1] - centre[1],
      positions[v * 3 + 2] - centre[2],
    );
    radii[v] = r;
    sum += r;
    if (r < min) min = r;
    if (r > max) max = r;
  }
  const mean = sum / n;
  let acc = 0;
  for (let v = 0; v < n; v++) acc += (radii[v] - mean) ** 2;
  return { mean, std: Math.sqrt(acc / n), min, max };
}

function unitSphereMesh(n = 48, radius = 14): { mesh: RawMesh; centre: [number, number, number] } {
  const centre: [number, number, number] = [(n - 1) / 2, (n - 1) / 2, (n - 1) / 2];
  const mesh = surfaceNets({
    field: sphereField(n, radius, centre),
    dims: [n, n, n],
    spacing: [1, 1, 1],
    isoValue: 0,
    origin: [0, 0, 0],
  });
  return { mesh, centre };
}

interface NodeFs {
  readFileSync(path: URL): Uint8Array;
}

/**
 * `@types/node` is not a dependency of this project (the app itself only ever
 * runs in a browser), so the one built-in this integration test needs is
 * reached through a non-literal specifier, which the compiler does not try to
 * resolve, and typed by hand at the boundary.
 */
async function nodeFs(): Promise<NodeFs> {
  const specifier: string = 'node:fs';
  return (await import(specifier)) as NodeFs;
}

/** Non-zero voxel count and inclusive bounding box of a mask. */
function maskExtent(
  mask: Uint8Array,
  dims: [number, number, number],
): { voxels: number; bounds: [number, number, number, number, number, number] } {
  let voxels = 0;
  let i0 = dims[0], j0 = dims[1], k0 = dims[2];
  let i1 = -1, j1 = -1, k1 = -1;
  for (let k = 0; k < dims[2]; k++) {
    for (let j = 0; j < dims[1]; j++) {
      const row = (k * dims[1] + j) * dims[0];
      for (let i = 0; i < dims[0]; i++) {
        if (mask[row + i] === 0) continue;
        voxels++;
        if (i < i0) i0 = i;
        if (j < j0) j0 = j;
        if (k < k0) k0 = k;
        if (i > i1) i1 = i;
        if (j > j1) j1 = j;
        if (k > k1) k1 = k;
      }
    }
  }
  return { voxels, bounds: [i0, j0, k0, i1, j1, k1] };
}

/**
 * Minimal NIfTI-1 reader, enough for the single-file uncompressed-data case
 * the segmentations use. Deliberately standalone so this test never depends on
 * the loader another part of the app owns.
 */
async function readNiftiInt8(url: URL): Promise<{
  dims: [number, number, number];
  data: Uint8Array;
}> {
  const fs = await nodeFs();
  const raw = gunzipSync(fs.readFileSync(url));
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  let little = true;
  if (view.getInt32(0, true) !== 348) {
    little = false;
    if (view.getInt32(0, false) !== 348) throw new Error('not a NIfTI-1 header');
  }
  const datatype = view.getInt16(70, little);
  if (datatype !== 256 && datatype !== 2) throw new Error(`unexpected datatype ${datatype}`);
  const dims: [number, number, number] = [
    view.getInt16(42, little),
    view.getInt16(44, little),
    view.getInt16(46, little),
  ];
  const voxOffset = Math.round(view.getFloat32(108, little)) || 352;
  const count = dims[0] * dims[1] * dims[2];
  // int8 and uint8 share a byte layout and these masks are 0/1, so a Uint8
  // view is correct for both without a copy.
  const data = new Uint8Array(raw.buffer, raw.byteOffset + voxOffset, count);
  return { dims, data };
}

// --- tests ----------------------------------------------------------------

describe('surfaceNets on an analytic sphere', () => {
  const radius = 14;
  const { mesh, centre } = unitSphereMesh(48, radius);

  it('produces a non-empty mesh', () => {
    expect(mesh.positions.length).toBeGreaterThan(0);
    expect(mesh.indices.length % 3).toBe(0);
    expect(mesh.indices.length / 3).toBeGreaterThan(1000);
  });

  it('places every vertex on the sphere surface', () => {
    const stats = radiusStats(mesh.positions, centre);
    // Surface nets averages the crossings of a cell, which pulls the vertex a
    // fraction of a voxel inside a convex surface. Half a voxel covers that.
    expect(Math.abs(stats.mean - radius)).toBeLessThan(0.25);
    expect(radius - stats.min).toBeLessThan(0.6);
    expect(stats.max - radius).toBeLessThan(0.6);
  });

  it('encloses the analytic volume, with positive (outward) winding', () => {
    const expected = (4 / 3) * Math.PI * radius ** 3;
    const actual = enclosedVolume(mesh.positions, mesh.indices);
    expect(actual).toBeGreaterThan(0);
    expect(Math.abs(actual - expected) / expected).toBeLessThan(0.03);
  });

  it('is a closed, consistently oriented manifold', () => {
    const report = checkManifold(mesh.indices, mesh.positions.length / 3);
    expect(report.firstBad).toBeNull();
    expect(report.badEdges).toBe(0);
    // Euler characteristic of a sphere: V - E + F = 2 with F triangles.
    const v = mesh.positions.length / 3;
    const f = mesh.indices.length / 3;
    expect(v - report.edgeCount + f).toBe(2);
  });

  it('orients normals outward', () => {
    const normals = computeNormals(mesh.positions, mesh.indices);
    let inward = 0;
    for (let v = 0; v < mesh.positions.length; v += 3) {
      const dot =
        normals[v] * (mesh.positions[v] - centre[0]) +
        normals[v + 1] * (mesh.positions[v + 1] - centre[1]) +
        normals[v + 2] * (mesh.positions[v + 2] - centre[2]);
      if (dot <= 0) inward++;
    }
    expect(inward).toBe(0);
  });

  it('reports bounds that tightly contain the sphere', () => {
    const [minX, minY, minZ, maxX, maxY, maxZ] = meshBounds(mesh.positions);
    for (const lo of [minX, minY, minZ]) expect(lo).toBeCloseTo(centre[0] - radius, 0);
    for (const hi of [maxX, maxY, maxZ]) expect(hi).toBeCloseTo(centre[0] + radius, 0);
  });
});

describe('anisotropic spacing', () => {
  it('stretches a voxel-space sphere into an ellipsoid along z', () => {
    const n = 40;
    const radius = 12;
    const centre: [number, number, number] = [(n - 1) / 2, (n - 1) / 2, (n - 1) / 2];
    const mesh = surfaceNets({
      field: sphereField(n, radius, centre),
      dims: [n, n, n],
      spacing: [1, 1, 4],
      isoValue: 0,
      origin: [0, 0, 0],
    });
    const [minX, minY, minZ, maxX, maxY, maxZ] = meshBounds(mesh.positions);
    const ex = maxX - minX;
    const ey = maxY - minY;
    const ez = maxZ - minZ;
    expect(ez / ex).toBeCloseTo(4, 1);
    expect(ey / ex).toBeCloseTo(1, 2);
  });

  it('offsets positions by the field origin', () => {
    const n = 24;
    const radius = 7;
    const centre: [number, number, number] = [(n - 1) / 2, (n - 1) / 2, (n - 1) / 2];
    const field = sphereField(n, radius, centre);
    const at = (origin: [number, number, number]): number[] =>
      meshBounds(
        surfaceNets({ field, dims: [n, n, n], spacing: [2, 2, 2], isoValue: 0, origin }).positions,
      );
    const base = at([0, 0, 0]);
    const shifted = at([10, -5, 3]);
    expect(shifted[0]).toBeCloseTo(base[0] + 20, 4);
    expect(shifted[1]).toBeCloseTo(base[1] - 10, 4);
    expect(shifted[2]).toBeCloseTo(base[2] + 6, 4);
  });
});

describe('fields with no isosurface', () => {
  const dims: [number, number, number] = [16, 16, 16];

  it('emits nothing for an all-zero field', () => {
    const mesh = surfaceNets({
      field: new Float32Array(16 * 16 * 16),
      dims,
      spacing: [1, 1, 1],
      isoValue: 0.5,
      origin: [0, 0, 0],
    });
    expect(mesh.indices.length).toBe(0);
    expect(mesh.positions.length).toBe(0);
  });

  it('emits nothing for an all-one field', () => {
    const mesh = surfaceNets({
      field: new Float32Array(16 * 16 * 16).fill(1),
      dims,
      spacing: [1, 1, 1],
      isoValue: 0.5,
      origin: [0, 0, 0],
    });
    expect(mesh.indices.length).toBe(0);
    expect(mesh.positions.length).toBe(0);
  });

  it('emits nothing for an all-zero mask through extractOrganMesh', () => {
    const out = extractOrganMesh({
      mask: new Uint8Array(16 * 16 * 16),
      dims,
      spacing: [1, 1, 1],
      bounds: [0, 0, 0, 15, 15, 15],
    });
    expect(out.triangleCount).toBe(0);
    expect(out.indices.length).toBe(0);
  });

  it('emits nothing for an empty bounding box', () => {
    const out = extractOrganMesh({
      mask: new Uint8Array(16 * 16 * 16),
      dims,
      spacing: [1, 1, 1],
      bounds: [8, 8, 8, 7, 7, 7],
    });
    expect(out.triangleCount).toBe(0);
  });
});

describe('boxBlur3', () => {
  it('leaves a constant field constant and does not touch its input', () => {
    const dims: [number, number, number] = [9, 8, 7];
    const field = new Float32Array(9 * 8 * 7).fill(3);
    const out = boxBlur3(field, dims, 2);
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(3, 5);
    expect(field.every((v) => v === 3)).toBe(true);
  });

  it('conserves total mass away from the border and spreads a delta', () => {
    const n = 21;
    const dims: [number, number, number] = [n, n, n];
    const field = new Float32Array(n * n * n);
    const centre = ((10 * n + 10) * n) + 10;
    field[centre] = 1;
    const out = boxBlur3(field, dims, 2);
    let sum = 0;
    for (let i = 0; i < out.length; i++) sum += out[i];
    expect(sum).toBeCloseTo(1, 4);
    expect(out[centre]).toBeLessThan(1);
    expect(out[centre]).toBeGreaterThan(0);
    expect(out[centre + 1]).toBeGreaterThan(0);
    expect(out[centre + n]).toBeGreaterThan(0);
    expect(out[centre + n * n]).toBeGreaterThan(0);
    // Radius 2 support per pass pair: nothing three voxels out.
    expect(out[centre + 3]).toBe(0);
  });
});

describe('taubinSmooth', () => {
  const n = 48;
  const radius = 14;
  const centre: [number, number, number] = [(n - 1) / 2, (n - 1) / 2, (n - 1) / 2];
  const mesh = surfaceNets({
    field: sphereField(n, radius, centre, 0.35, 7),
    dims: [n, n, n],
    spacing: [1, 1, 1],
    isoValue: 0,
    origin: [0, 0, 0],
  });

  it('flattens radial noise while preserving enclosed volume', () => {
    const before = radiusStats(mesh.positions, centre);
    const volumeBefore = enclosedVolume(mesh.positions, mesh.indices);

    const smoothed = taubinSmooth(mesh.positions, mesh.indices, 20);
    const after = radiusStats(smoothed, centre);
    const volumeAfter = enclosedVolume(smoothed, mesh.indices);

    expect(after.std).toBeLessThan(before.std * 0.85);
    expect(Math.abs(volumeAfter - volumeBefore) / volumeBefore).toBeLessThan(0.01);
  });

  it('holds volume where a plain Laplacian would collapse it', () => {
    // Same filter with mu forced positive is exactly Laplacian smoothing, and
    // it is the one comparison that shows what the mu step buys: the shrinkage
    // it removes is two orders of magnitude larger than what Taubin leaves.
    const volumeBefore = enclosedVolume(mesh.positions, mesh.indices);
    const laplacian = taubinSmooth(mesh.positions, mesh.indices, 20, 0.5, 0.5);
    const taubin = taubinSmooth(mesh.positions, mesh.indices, 20);

    const laplacianShrink =
      (volumeBefore - enclosedVolume(laplacian, mesh.indices)) / volumeBefore;
    const taubinShrink = (volumeBefore - enclosedVolume(taubin, mesh.indices)) / volumeBefore;

    expect(laplacianShrink).toBeGreaterThan(0.03);
    expect(Math.abs(taubinShrink)).toBeLessThan(laplacianShrink / 10);
  });

  it('does not modify the input positions', () => {
    const copy = mesh.positions.slice();
    taubinSmooth(mesh.positions, mesh.indices, 6);
    expect(mesh.positions).toEqual(copy);
  });

  it('is a no-op for zero iterations', () => {
    const out = taubinSmooth(mesh.positions, mesh.indices, 0);
    expect(out).toEqual(mesh.positions);
  });

  it('keeps the mesh manifold', () => {
    const smoothed = taubinSmooth(mesh.positions, mesh.indices, 12);
    expect(smoothed.length).toBe(mesh.positions.length);
    const report = checkManifold(mesh.indices, mesh.positions.length / 3);
    expect(report.badEdges).toBe(0);
  });
});

describe('extractOrganMesh on a synthetic sphere mask', () => {
  const n = 64;
  const radius = 18;
  const centre: [number, number, number] = [32, 32, 32];
  const mask = sphereMask(n, radius, centre);

  it('recovers the sphere volume from a binary mask', () => {
    const out = extractOrganMesh({
      mask,
      dims: [n, n, n],
      spacing: [1, 1, 1],
      bounds: [
        centre[0] - radius, centre[1] - radius, centre[2] - radius,
        centre[0] + radius, centre[1] + radius, centre[2] + radius,
      ],
    });
    expect(out.triangleCount).toBeGreaterThan(1000);
    expect(out.normals.length).toBe(out.positions.length);
    const report = checkManifold(out.indices, out.positions.length / 3);
    expect(report.badEdges).toBe(0);
    const expected = (4 / 3) * Math.PI * radius ** 3;
    const actual = enclosedVolume(out.positions, out.indices);
    expect(Math.abs(actual - expected) / expected).toBeLessThan(0.05);
  });

  it('closes a mask that runs off the edge of the volume', () => {
    // Half a sphere, cut flat by the k = 0 plane. Clamping the padded crop to
    // the volume would leave the flat face open; treating out-of-volume as
    // background keeps the mesh watertight, which is what the real liver needs.
    const half = new Uint8Array(n * n * n);
    for (let k = 0; k < n; k++) {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          const d = Math.hypot(i - centre[0], j - centre[1], k);
          if (d <= radius) half[(k * n + j) * n + i] = 1;
        }
      }
    }
    const out = extractOrganMesh({
      mask: half,
      dims: [n, n, n],
      spacing: [1, 1, 1],
      bounds: [
        centre[0] - radius, centre[1] - radius, 0,
        centre[0] + radius, centre[1] + radius, radius,
      ],
    });
    const report = checkManifold(out.indices, out.positions.length / 3);
    expect(report.firstBad).toBeNull();
    expect(enclosedVolume(out.positions, out.indices)).toBeGreaterThan(0);
  });

  it('produces a lighter mesh at decimateStride 3 with the same volume', () => {
    const bounds: [number, number, number, number, number, number] = [
      centre[0] - radius, centre[1] - radius, centre[2] - radius,
      centre[0] + radius, centre[1] + radius, centre[2] + radius,
    ];
    const full = extractOrganMesh({ mask, dims: [n, n, n], spacing: [1, 1, 1], bounds });
    const preview = extractOrganMesh({
      mask,
      dims: [n, n, n],
      spacing: [1, 1, 1],
      bounds,
      decimateStride: 3,
    });
    expect(preview.triangleCount).toBeGreaterThan(0);
    expect(preview.triangleCount).toBeLessThan(full.triangleCount / 4);
    // A preview only promises closed and consistently oriented; see the note on
    // pinch cells in surfaceNets.ts.
    expect(checkManifold(preview.indices, preview.positions.length / 3).unbalancedEdges).toBe(0);
    // Blurring at the coarse scale shifts a curved boundary inward by roughly
    // the blur variance times the curvature, so a preview reads a few percent
    // small on anything this tightly curved. That is the speed/accuracy trade.
    const expected = (4 / 3) * Math.PI * radius ** 3;
    const actual = enclosedVolume(preview.positions, preview.indices);
    expect(actual).toBeLessThan(expected);
    expect(Math.abs(actual - expected) / expected).toBeLessThan(0.2);
    // The preview must sit in the same place, not just be the same size.
    const fb = full.bounds;
    const pb = preview.bounds;
    for (let a = 0; a < 6; a++) expect(Math.abs(pb[a] - fb[a])).toBeLessThan(2.5);
  });
});

describe('integration: real liver segmentation', () => {
  it('meshes BDMAP_00000338 liver to within 10% of the reference volume', async () => {
    const readStart = performance.now();
    const { dims, data } = await readNiftiInt8(LIVER_NII);
    const readMs = performance.now() - readStart;
    expect(dims).toEqual([502, 348, 71]);

    // Voxel count and bbox straight from the mask, so the mesh is compared
    // against the same ground truth nibabel reports.
    const { voxels, bounds } = maskExtent(data, dims);
    const [i0, j0, k0, i1, j1, k1] = bounds;
    expect(voxels).toBe(944325);

    const meshStart = performance.now();
    const out = extractOrganMesh({ mask: data, dims, spacing: LIVER_SPACING, bounds });
    const meshMs = performance.now() - meshStart;

    const voxelMl = (voxels * LIVER_SPACING[0] * LIVER_SPACING[1] * LIVER_SPACING[2]) / 1000;
    const meshMl = enclosedVolume(out.positions, out.indices) / 1000;

    const manifoldStart = performance.now();
    const report = checkManifold(out.indices, out.positions.length / 3);
    const manifoldMs = performance.now() - manifoldStart;

    console.log(
      [
        `liver bbox            [${i0},${j0},${k0}]..[${i1},${j1},${k1}]`,
        `vertices              ${out.positions.length / 3}`,
        `triangles             ${out.triangleCount}`,
        `mesh bounds mm        ${out.bounds.map((v) => v.toFixed(1)).join(', ')}`,
        `voxel volume          ${voxelMl.toFixed(1)} mL (truth ${LIVER_TRUTH_ML} mL)`,
        `mesh volume           ${meshMl.toFixed(1)} mL`,
        `error vs truth        ${(((meshMl - LIVER_TRUTH_ML) / LIVER_TRUTH_ML) * 100).toFixed(2)} %`,
        `timing                read ${readMs.toFixed(0)} ms, extract ${meshMs.toFixed(0)} ms, manifold check ${manifoldMs.toFixed(0)} ms`,
      ].join('\n  '),
    );

    expect(voxelMl).toBeCloseTo(LIVER_TRUTH_ML, 0);
    expect(report.firstBad).toBeNull();
    expect(report.badEdges).toBe(0);
    const v = out.positions.length / 3;
    const f = out.triangleCount;
    expect(v - report.edgeCount + f).toBe(2);

    expect(out.triangleCount).toBeGreaterThan(50_000);
    expect(out.triangleCount).toBeLessThan(2_000_000);
    expect(out.normals.length).toBe(out.positions.length);

    expect(Math.abs(meshMl - LIVER_TRUTH_ML) / LIVER_TRUTH_ML).toBeLessThan(0.1);

    // Local mm space: bounds must line up with voxelIndex * spacing, no affine.
    expect(out.bounds[0]).toBeGreaterThan((i0 - 3) * LIVER_SPACING[0]);
    expect(out.bounds[3]).toBeLessThan((i1 + 3) * LIVER_SPACING[0]);
    // The liver mask starts at k = 0, so the closing cap sits just outside the
    // scan. Anything much below that would mean the padding had run away.
    expect(k0).toBe(0);
    expect(out.bounds[2]).toBeGreaterThan(-LIVER_SPACING[2]);
    expect(out.bounds[2]).toBeLessThan(0);
  }, 120_000);

  it('meshes all nine organs closed, manifold and volume-accurate', async () => {
    const rows: string[] = [];
    let totalMs = 0;

    for (const [organ, truthVoxels] of ORGAN_VOXELS) {
      const { dims, data } = await readNiftiInt8(
        new URL(`${SEGMENTATIONS}/${organ}.nii.gz`, import.meta.url),
      );
      const { voxels, bounds } = maskExtent(data, dims);
      expect(voxels, organ).toBe(truthVoxels);

      const start = performance.now();
      const out = extractOrganMesh({ mask: data, dims, spacing: LIVER_SPACING, bounds });
      const ms = performance.now() - start;
      totalMs += ms;

      const report = checkManifold(out.indices, out.positions.length / 3);
      const voxelMl = (voxels * LIVER_SPACING[0] * LIVER_SPACING[1] * LIVER_SPACING[2]) / 1000;
      const meshMl = enclosedVolume(out.positions, out.indices) / 1000;
      const error = (meshMl - voxelMl) / voxelMl;

      rows.push(
        `${organ.padEnd(13)} T=${String(out.triangleCount).padStart(6)}` +
          ` ${ms.toFixed(0).padStart(4)} ms  ${meshMl.toFixed(1).padStart(7)} mL` +
          ` vs ${voxelMl.toFixed(1).padStart(7)} mL (${(error * 100).toFixed(2).padStart(6)} %)`,
      );

      expect(report.firstBad, organ).toBeNull();
      expect(out.triangleCount, organ).toBeGreaterThan(1000);
      // Small organs read low because the surface bias is a fixed fraction of a
      // voxel and they have far more surface per unit volume; the gall bladder
      // is the worst of the nine at about -5%.
      expect(Math.abs(error), organ).toBeLessThan(0.06);
      expect(meshMl, organ).toBeLessThan(voxelMl);
    }

    console.log(`\n  ${rows.join('\n  ')}\n  total extract ${totalMs.toFixed(0)} ms`);
  }, 120_000);
});
