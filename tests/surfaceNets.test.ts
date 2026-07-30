import { gunzipSync } from 'fflate';
import { describe, it, expect } from 'vitest';

import { surfaceNets, computeNormals, meshBounds, extractOrganMesh } from '../src/mesh/surfaceNets';
import { taubinSmooth } from '../src/mesh/smooth';
import { boxBlur3 } from '../src/mesh/blur';
import type { RawMesh, OrganMesh } from '../src/mesh/surfaceNets';

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

/**
 * Rebuild an index set from vertex positions, merging vertices that land on the
 * same point. Used to check a mesh whose coordinates moved: the original index
 * array is unchanged by construction and so proves nothing about them.
 */
function weldByPosition(
  positions: Float32Array,
  indices: Uint32Array,
): { indices: Uint32Array; vertexCount: number } {
  // A tenth of a micron, far below anything meshing at millimetre scale
  // produces, so only genuinely coincident vertices collapse.
  const quantise = (v: number): number => Math.round(v * 1e4);
  const ids = new Map<string, number>();
  const remap = new Uint32Array(positions.length / 3);
  for (let v = 0; v < remap.length; v++) {
    const key = `${quantise(positions[v * 3])},${quantise(positions[v * 3 + 1])},${quantise(positions[v * 3 + 2])}`;
    let id = ids.get(key);
    if (id === undefined) {
      id = ids.size;
      ids.set(key, id);
    }
    remap[v] = id;
  }
  const welded = new Uint32Array(indices.length);
  for (let i = 0; i < indices.length; i++) welded[i] = remap[indices[i]];
  return { indices: welded, vertexCount: ids.size };
}

/** Solid axis-aligned box mask in an n^3 volume, inclusive of both corners. */
function boxMask(n: number, box: [number, number, number, number, number, number]): Uint8Array {
  const mask = new Uint8Array(n * n * n);
  for (let k = box[2]; k <= box[5]; k++) {
    for (let j = box[1]; j <= box[4]; j++) {
      for (let i = box[0]; i <= box[3]; i++) mask[(k * n + j) * n + i] = 1;
    }
  }
  return mask;
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

  it('keeps the smoothed mesh a closed manifold', () => {
    const smoothed = taubinSmooth(mesh.positions, mesh.indices, 12);
    expect(smoothed.length).toBe(mesh.positions.length);
    expect(smoothed.every(Number.isFinite)).toBe(true);

    // Re-index by position instead of reusing the index array, which the
    // smoother cannot touch and which therefore says nothing about its output.
    // Welding is what notices vertices that were fused, collapsed or filled
    // with NaN, and only then does the manifold check describe the mesh that
    // actually came out of the smoother.
    const welded = weldByPosition(smoothed, mesh.indices);
    expect(welded.vertexCount).toBe(mesh.positions.length / 3);
    const report = checkManifold(welded.indices, welded.vertexCount);
    expect(report.firstBad).toBeNull();
    expect(report.badEdges).toBe(0);
    expect(enclosedVolume(smoothed, welded.indices)).toBeGreaterThan(0);
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
    // Blurring shifts a curved boundary inward by roughly the blur variance
    // times the curvature, so a preview reads small, and it is the decimation
    // that sets how small: this radius-18 sphere is the gentle case. See the
    // real-mask preview test for the figures that actually bound the trade.
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

/**
 * Both guards below compare the shipped code against a local reimplementation
 * that differs from it in exactly one way, the defect being guarded, and run
 * the two interleaved so machine speed, thermal drift and JIT state cancel out
 * of the ratio. That is what makes a timing assertion here reproducible: the
 * threshold is a shape of the code, not a number of milliseconds.
 */
function raceBestOf(trials: number, a: () => void, b: () => void): [number, number] {
  let bestA = Infinity;
  let bestB = Infinity;
  for (let t = 0; t < trials; t++) {
    let start = performance.now();
    a();
    const elapsedA = performance.now() - start;
    start = performance.now();
    b();
    const elapsedB = performance.now() - start;
    if (elapsedA < bestA) bestA = elapsedA;
    if (elapsedB < bestB) bestB = elapsedB;
  }
  return [bestA, bestB];
}

/**
 * The cell scan as it was before the sign test came first: all eight corner
 * values written into a scratch array, only for the early-out to throw 97% of
 * them away on the next line.
 */
function scanFillingScratch(field: Float32Array, dims: [number, number, number], isoValue: number): number {
  const [nx, ny, nz] = dims;
  const strideY = nx;
  const strideZ = nx * ny;
  const cornerOffset = new Int32Array([
    0, 1, strideY, strideY + 1, strideZ, strideZ + 1, strideZ + strideY, strideZ + strideY + 1,
  ]);
  const cnx = nx - 1;
  const cny = ny - 1;
  const g = new Float64Array(8);
  let below = new Int32Array(cnx * cny);
  let cur = new Int32Array(cnx * cny);
  let straddling = 0;
  for (let k = 0; k < nz - 1; k++) {
    for (let j = 0; j < cny; j++) {
      const rowBase = k * strideZ + j * strideY;
      const cellRow = j * cnx;
      for (let i = 0; i < cnx; i++) {
        const base = rowBase + i;
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const v = field[base + cornerOffset[c]] - isoValue;
          g[c] = v;
          if (v > 0) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) {
          cur[cellRow + i] = -1;
          continue;
        }
        straddling++;
      }
    }
    const swap = below;
    below = cur;
    cur = swap;
  }
  return straddling + (below[0] & 0);
}

/** The crop as it was with the stride-block loops nested inside the x loop. */
function cropWithNestedBlockLoops(
  mask: Uint8Array,
  dims: [number, number, number],
  clip: [number, number, number, number, number, number],
  lo: [number, number, number],
  coarse: [number, number, number],
  stride: number,
): Float32Array {
  const [nx, ny] = dims;
  const [cnx, cny, cnz] = coarse;
  const [ci0, cj0, ck0, ci1, cj1, ck1] = clip;
  const field = new Float32Array(cnx * cny * cnz);
  const invBlock = 1 / (stride * stride * stride);
  const planeStride = nx * ny;
  for (let cz = 0; cz < cnz; cz++) {
    const zBase = lo[2] + cz * stride;
    for (let cy = 0; cy < cny; cy++) {
      const yBase = lo[1] + cy * stride;
      const out = (cz * cny + cy) * cnx;
      for (let cx = 0; cx < cnx; cx++) {
        const xBase = lo[0] + cx * stride;
        let sum = 0;
        for (let dz = 0; dz < stride; dz++) {
          const z = zBase + dz;
          if (z < ck0 || z > ck1) continue;
          for (let dy = 0; dy < stride; dy++) {
            const y = yBase + dy;
            if (y < cj0 || y > cj1) continue;
            const rowBase = z * planeStride + y * nx;
            const xStart = Math.max(0, ci0 - xBase);
            const xEnd = Math.min(stride, ci1 - xBase + 1);
            for (let dx = xStart; dx < xEnd; dx++) {
              if (mask[rowBase + xBase + dx] !== 0) sum++;
            }
          }
        }
        field[out + cx] = sum * invBlock;
      }
    }
  }
  return field;
}

describe('cost of the passes that scale with the volume', () => {
  const n = 96;
  const dims: [number, number, number] = [n, n, n];

  it('discards a cell without touching its corner values', () => {
    const field = new Float32Array(n * n * n);
    const reps = 4;
    const shipped = (): void => {
      for (let r = 0; r < reps; r++) {
        surfaceNets({ field, dims, spacing: [1, 1, 1], isoValue: 0.5, origin: [0, 0, 0] });
      }
    };
    const withScratchFill = (): void => {
      for (let r = 0; r < reps; r++) scanFillingScratch(field, dims, 0.5);
    };
    for (let warm = 0; warm < 4; warm++) {
      shipped();
      withScratchFill();
    }
    const [ours, theirs] = raceBestOf(9, shipped, withScratchFill);
    console.log(`  cell scan ${ours.toFixed(1)} ms vs ${theirs.toFixed(1)} ms filling scratch`);
    // Measures 0.37-0.43 as written and 0.89 with the scratch fill restored.
    expect(ours / theirs).toBeLessThan(0.65);
  });

  it('clamps a crop column once per column, not once per sample', () => {
    // A box far larger than the structure inside it, and no blur, so almost
    // all the work is reading voxels.
    const mask = boxMask(n, [46, 46, 46, 49, 49, 49]);
    const bounds: [number, number, number, number, number, number] = [3, 3, 3, n - 4, n - 4, n - 4];
    const pad = 2;
    const lo: [number, number, number] = [bounds[0] - pad, bounds[1] - pad, bounds[2] - pad];
    const coarse: [number, number, number] = [
      bounds[3] + pad - lo[0] + 1, bounds[4] + pad - lo[1] + 1, bounds[5] + pad - lo[2] + 1,
    ];
    const reps = 3;
    const shipped = (): void => {
      for (let r = 0; r < reps; r++) {
        extractOrganMesh({ mask, dims, spacing: [1, 1, 1], bounds, blurPasses: 0, smoothIterations: 0 });
      }
    };
    const nested = (): void => {
      for (let r = 0; r < reps; r++) {
        const field = cropWithNestedBlockLoops(mask, dims, bounds, lo, coarse, 1);
        surfaceNets({
          field: boxBlur3(field, coarse, 0),
          dims: coarse,
          spacing: [1, 1, 1],
          isoValue: 0.5,
          origin: lo,
        });
      }
    };
    for (let warm = 0; warm < 4; warm++) {
      shipped();
      nested();
    }
    const [ours, theirs] = raceBestOf(9, shipped, nested);
    console.log(`  crop pipeline ${ours.toFixed(1)} ms vs ${theirs.toFixed(1)} ms with nested block loops`);
    // Measures 0.57-0.60; the two pipelines share everything but the crop.
    expect(ours / theirs).toBeLessThan(0.8);
  });
});

describe('extractOrganMesh input contract', () => {
  const dims: [number, number, number] = [16, 16, 16];

  it('rejects a mask shorter than its dims instead of meshing past the end', () => {
    // Reads past the end of a typed array give undefined, and undefined !== 0
    // is true, so a truncated mask used to come back as solid tissue filling
    // the missing slices rather than as an error.
    expect(() =>
      extractOrganMesh({
        mask: new Uint8Array(16 * 16 * 8),
        dims,
        spacing: [1, 1, 1],
        bounds: [0, 0, 0, 15, 15, 15],
      }),
    ).toThrow(/2048.*4096/);
  });

  it('never reads a voxel outside the declared bounds', () => {
    // `bounds` is documented as a superset of the structure, so anything
    // outside it is background whatever the buffer happens to hold. Callers
    // share one scratch mask between structures and only clear the box they
    // are about to fill, which leaves exactly this kind of residue behind.
    const n = 40;
    const bounds: [number, number, number, number, number, number] = [14, 14, 14, 24, 24, 24];
    const clean = boxMask(n, bounds);
    const withResidue = boxMask(n, bounds);
    withResidue[(26 * n + 19) * n + 19] = 1;

    const build = (mask: Uint8Array): OrganMesh =>
      extractOrganMesh({ mask, dims: [n, n, n], spacing: [1, 1, 1], bounds, blurPasses: 3 });
    const expected = build(clean);
    const actual = build(withResidue);

    expect(actual.triangleCount).toBe(expected.triangleCount);
    expect(actual.bounds).toEqual(expected.bounds);
    expect(Array.from(actual.positions)).toEqual(Array.from(expected.positions));
  });

  it('closes the mesh where bounds cut through solid mask', () => {
    // The mirror image of the halo case: bounds tighter than the structure.
    // Reading outside the box would leave the crop straddling solid material
    // at its edge, and the surface would run off it (108 boundary edges, and
    // an enclosed volume a quarter of the truth) with no error raised.
    const n = 48;
    const radius = 14;
    const centre = 24;
    const mask = sphereMask(n, radius, [centre, centre, centre]);
    const out = extractOrganMesh({
      mask,
      dims: [n, n, n],
      spacing: [1, 1, 1],
      bounds: [centre - radius, centre - radius, centre - radius, centre + radius, centre + radius, centre],
      blurPasses: 3,
    });
    const report = checkManifold(out.indices, out.positions.length / 3);
    expect(report.firstBad).toBeNull();
    expect(report.unbalancedEdges).toBe(0);
    // Half a sphere, since the far half was declared out of bounds.
    const expected = ((4 / 3) * Math.PI * radius ** 3) / 2;
    const actual = enclosedVolume(out.positions, out.indices);
    expect(Math.abs(actual - expected) / expected).toBeLessThan(0.05);
  });

  it('keeps the mesh closed at blur settings far past the default', () => {
    // The crop pad has to grow with the blur radius. Fixed at two voxels, the
    // blurred field is still above the isovalue at the array border once the
    // pass count reaches the low teens (border clamping reflects the solid
    // region back in), and the surface runs off the edge of the crop: 16
    // passes gave 288 boundary edges and lost 37% of the volume.
    const n = 40;
    const bounds: [number, number, number, number, number, number] = [10, 10, 10, 29, 29, 29];
    const mask = boxMask(n, bounds);
    for (const blurPasses of [2, 8, 16, 24]) {
      const out = extractOrganMesh({
        mask,
        dims: [n, n, n],
        spacing: [1, 1, 1],
        bounds,
        blurPasses,
        smoothIterations: 0,
      });
      const label = `blurPasses ${blurPasses}`;
      const report = checkManifold(out.indices, out.positions.length / 3);
      expect(report.firstBad, label).toBeNull();
      expect(report.badEdges, label).toBe(0);
      // A 20^3 box. Heavy blurring rounds its corners off, which is the point
      // of blurring, but it must not eat the box.
      const volume = enclosedVolume(out.positions, out.indices);
      expect(volume, label).toBeGreaterThan(0.6 * 8000);
      expect(volume, label).toBeLessThan(8000);
    }
  });
});

describe('structures thinner than the blur', () => {
  const n = 24;
  const dims: [number, number, number] = [n, n, n];

  /**
   * The blur runs before the 0.5 threshold, so a thin structure's peak
   * occupancy can land below the isovalue and the mesh comes back empty: byte
   * for byte what an empty mask returns, while the structure still reports a
   * non-zero volume in the UI. The last column is the pass counts at which
   * that happened, and three is what the app runs. None of the nine reference
   * organs is this thin, but the anatomy catalogue lists adrenal glands, ribs
   * and small vessels.
   */
  const cases: ReadonlyArray<
    readonly [string, [number, number, number, number, number, number], number, number[]]
  > = [
    ['3-voxel cube', [11, 11, 11, 13, 13, 13], 27, [2, 3]],
    ['2-voxel slab', [4, 4, 11, 19, 19, 12], 512, [3]],
    ['3-voxel tube', [11, 11, 4, 13, 13, 19], 144, [3]],
  ];

  for (const [name, box, voxels, erasedAt] of cases) {
    it(`meshes a ${name} that the blur would erase`, () => {
      const mask = boxMask(n, box);
      for (const blurPasses of erasedAt) {
        const label = `${name} at ${blurPasses} passes`;
        const out = extractOrganMesh({
          mask,
          dims,
          spacing: [1, 1, 1],
          bounds: box,
          blurPasses,
          smoothIterations: 12,
        });
        expect(out.triangleCount, label).toBeGreaterThan(0);
        const report = checkManifold(out.indices, out.positions.length / 3);
        expect(report.firstBad, label).toBeNull();
        // Dropping the blur rather than the structure keeps the mesh the size
        // of the mask; keeping some blur would leave whichever sliver survived
        // it, which for the cube is 11% of its volume.
        const volume = enclosedVolume(out.positions, out.indices);
        expect(volume, label).toBeGreaterThan(0.4 * voxels);
        expect(volume, label).toBeLessThan(voxels);
      }
    });
  }

  it('still reports nothing for a mask with nothing in it', () => {
    const out = extractOrganMesh({
      mask: new Uint8Array(n * n * n),
      dims,
      spacing: [1, 1, 1],
      bounds: [8, 8, 8, 12, 12, 12],
      blurPasses: 3,
    });
    expect(out.triangleCount).toBe(0);
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

  it('meshes all nine organs at the settings the app actually ships', async () => {
    // The module defaults are 2 blur passes and 12 smoothing steps; the mesh
    // worker asks for 3 and 24, because the masks are segmented slice by slice
    // and terrace along z at 2.5 mm spacing. Everything above tests the
    // defaults, so nothing tested the configuration that runs in the browser.
    const rows: string[] = [];
    for (const [organ, truthVoxels] of ORGAN_VOXELS) {
      const { dims, data } = await readNiftiInt8(
        new URL(`${SEGMENTATIONS}/${organ}.nii.gz`, import.meta.url),
      );
      const { voxels, bounds } = maskExtent(data, dims);
      expect(voxels, organ).toBe(truthVoxels);

      const out = extractOrganMesh({
        mask: data,
        dims,
        spacing: LIVER_SPACING,
        bounds,
        blurPasses: 3,
        smoothIterations: 24,
      });
      const report = checkManifold(out.indices, out.positions.length / 3);
      const voxelMl = (voxels * LIVER_SPACING[0] * LIVER_SPACING[1] * LIVER_SPACING[2]) / 1000;
      const meshMl = enclosedVolume(out.positions, out.indices) / 1000;
      const error = (meshMl - voxelMl) / voxelMl;

      rows.push(
        `${organ.padEnd(13)} T=${String(out.triangleCount).padStart(6)}` +
          ` bad ${String(report.badEdges).padStart(2)} of ${String(report.edgeCount).padStart(6)} edges` +
          ` ${meshMl.toFixed(1).padStart(7)} mL (${(error * 100).toFixed(2).padStart(6)} %)`,
      );

      // Closed and consistently oriented is the property normals and the
      // divergence-theorem volume depend on, and it holds everywhere.
      expect(report.unbalancedEdges, organ).toBe(0);
      // Strict manifoldness does not, quite: the heavier blur puts more detail
      // at the sample scale, and the liver hits the pinched-cell case
      // documented in surfaceNets.ts at two of its 340k edges. Pinned so that
      // a change which makes it common fails here.
      expect(report.badEdges, organ).toBeLessThanOrEqual(2);
      // Heavier blurring pulls curved boundaries further in, so the small
      // organs read lower than they do at the default two passes (-6.6% for
      // the pancreas against -4.3%).
      expect(error, organ).toBeLessThan(0);
      expect(Math.abs(error), organ).toBeLessThan(0.07);
    }
    console.log(`\n  ${rows.join('\n  ')}`);
  }, 120_000);

  it('ignores the neighbours a shared scratch mask leaves in the halo', async () => {
    // What the mesh worker really passes: one reused buffer, cleared and
    // filled only inside the box of the structure being built. The pancreas
    // box has 11518 stomach and 10097 liver voxels within two voxels of it,
    // which is what the crop pad reaches into, and before the bounds clamp
    // 26.5% of the pancreas mesh was made of them.
    const load = async (organ: string): Promise<Uint8Array> =>
      (await readNiftiInt8(new URL(`${SEGMENTATIONS}/${organ}.nii.gz`, import.meta.url))).data;
    const { dims } = await readNiftiInt8(
      new URL(`${SEGMENTATIONS}/pancreas.nii.gz`, import.meta.url),
    );
    const pancreas = await load('pancreas');
    const { bounds } = maskExtent(pancreas, dims);
    const [i0, j0, k0, i1, j1, k1] = bounds;

    const shared = new Uint8Array(pancreas.length);
    for (const organ of ['liver', 'stomach']) {
      const other = await load(organ);
      for (let v = 0; v < other.length; v++) if (other[v] !== 0) shared[v] = 1;
    }
    let halo = 0;
    for (let k = k0; k <= k1; k++) {
      for (let j = j0; j <= j1; j++) {
        const row = (k * dims[1] + j) * dims[0];
        for (let i = i0; i <= i1; i++) shared[row + i] = pancreas[row + i] !== 0 ? 1 : 0;
      }
    }
    for (let k = Math.max(0, k0 - 2); k <= Math.min(dims[2] - 1, k1 + 2); k++) {
      for (let j = Math.max(0, j0 - 2); j <= Math.min(dims[1] - 1, j1 + 2); j++) {
        const row = (k * dims[1] + j) * dims[0];
        const inRows = j >= j0 && j <= j1 && k >= k0 && k <= k1;
        for (let i = Math.max(0, i0 - 2); i <= Math.min(dims[0] - 1, i1 + 2); i++) {
          if (inRows && i >= i0 && i <= i1) continue;
          if (shared[row + i] !== 0) halo++;
        }
      }
    }
    expect(halo).toBeGreaterThan(10_000);

    const settings = { dims, spacing: LIVER_SPACING, bounds, blurPasses: 3, smoothIterations: 24 };
    const alone = extractOrganMesh({ mask: pancreas, ...settings });
    const reused = extractOrganMesh({ mask: shared, ...settings });

    expect(reused.triangleCount).toBe(alone.triangleCount);
    expect(reused.bounds).toEqual(alone.bounds);
    expect(Array.from(reused.positions)).toEqual(Array.from(alone.positions));
  }, 120_000);

  it('refuses a truncated mask instead of meshing the missing slices as solid', async () => {
    const { dims, data } = await readNiftiInt8(LIVER_NII);
    const { bounds } = maskExtent(data, dims);
    // Eight slices short, as a partial fetch would leave it. The out-of-range
    // reads counted as inside, and the liver came back at 3899.8 mL, 2.5x its
    // true volume, filling the missing slices with tissue.
    const short = data.subarray(0, dims[0] * dims[1] * (dims[2] - 8));
    expect(() =>
      extractOrganMesh({ mask: short, dims, spacing: LIVER_SPACING, bounds }),
    ).toThrow(/mask has \d+ voxels/);
  }, 120_000);

  it('keeps a decimated preview the size of the organ it previews', async () => {
    // The synthetic sphere above is a gentle case. Real organs are thin
    // somewhere, and the blur that a decimated grid inherits erodes them from
    // that direction: at a fixed three passes the gall bladder read 25% small
    // at stride 2, 51% at stride 3 and 80% at stride 4, which is not a preview
    // of anything. Scaling the blur down with the stride is what bounds this.
    const rows: string[] = [];
    for (const organ of ['gall_bladder', 'aorta', 'pancreas', 'liver']) {
      const { dims, data } = await readNiftiInt8(
        new URL(`${SEGMENTATIONS}/${organ}.nii.gz`, import.meta.url),
      );
      const { voxels, bounds } = maskExtent(data, dims);
      const voxelMl = (voxels * LIVER_SPACING[0] * LIVER_SPACING[1] * LIVER_SPACING[2]) / 1000;

      const full = extractOrganMesh({
        mask: data, dims, spacing: LIVER_SPACING, bounds, blurPasses: 3, smoothIterations: 24,
      });
      const errors: string[] = [];
      for (const stride of [2, 3, 4]) {
        const preview = extractOrganMesh({
          mask: data, dims, spacing: LIVER_SPACING, bounds,
          decimateStride: stride, blurPasses: 3, smoothIterations: 24,
        });
        const label = `${organ} stride ${stride}`;
        const meshMl = enclosedVolume(preview.positions, preview.indices) / 1000;
        const error = (meshMl - voxelMl) / voxelMl;
        errors.push(`s${stride} ${(error * 100).toFixed(1)}%`);

        expect(preview.triangleCount, label).toBeGreaterThan(0);
        expect(preview.triangleCount, label).toBeLessThan(full.triangleCount / (stride + 1));
        // A preview only promises closed and consistently oriented; see the
        // note on pinch cells in surfaceNets.ts.
        const report = checkManifold(preview.indices, preview.positions.length / 3);
        expect(report.unbalancedEdges, label).toBe(0);
        // Worst of these is the gall bladder at stride 4, reading 17% small.
        expect(Math.abs(error), label).toBeLessThan(0.2);
        // Same place, not just the same size.
        for (let a = 0; a < 6; a++) {
          expect(Math.abs(preview.bounds[a] - full.bounds[a]), `${label} bound ${a}`)
            .toBeLessThan(4 * stride * LIVER_SPACING[2]);
        }
      }
      rows.push(`${organ.padEnd(13)} ${errors.join('  ')}`);
    }
    console.log(`\n  ${rows.join('\n  ')}`);
  }, 120_000);
});
