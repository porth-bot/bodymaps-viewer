/**
 * Naive surface nets (dual contouring on the cell grid) for binary organ masks.
 *
 * Not marching cubes: no 256-entry case table to mistranscribe, every
 * sign-changing grid edge emits exactly one quad so the mesh is closed and
 * consistently oriented by construction, and one centred vertex per cell avoids
 * the slivers marching cubes leaves on a blurred mask. The price is that sharp
 * features get rounded off, which for anatomy is the right trade.
 *
 * Coordinates are volume local millimetres: voxelIndex * spacing, origin at
 * voxel (0,0,0). The world affine is deliberately not applied, the renderer
 * stays in this local space.
 */

import { boxBlur3 } from './blur';
import { taubinSmooth } from './smooth';

export interface SurfaceNetsInput {
  /** Scalar field sampled on the grid described by `dims`. */
  field: Float32Array;
  dims: [number, number, number];
  spacing: [number, number, number];
  /** Samples strictly greater than this are inside the solid. */
  isoValue: number;
  /** Voxel-index offset of field[0] inside the full volume. May be fractional. */
  origin: [number, number, number];
}

export interface RawMesh {
  positions: Float32Array;
  indices: Uint32Array;
}

/**
 * Cube corner numbering: bit 0 is the +i offset, bit 1 is +j, bit 2 is +k.
 * The three axis groups below list the corner pairs of the 12 cube edges, so
 * edges 0-3 run along x, 4-7 along y and 8-11 along z.
 */
const EDGE_A = new Int32Array([
  0, 2, 4, 6, // x edges
  0, 1, 4, 5, // y edges
  0, 1, 2, 3, // z edges
]);
const EDGE_B = new Int32Array([
  1, 3, 5, 7,
  2, 3, 6, 7,
  4, 5, 6, 7,
]);

/**
 * The six cube faces, each a cycle of four corners plus the matching cycle of
 * four edges, where FACE_EDGES[f*4+n] joins FACE_CORNERS[f*4+n] to the corner
 * after it. Order: k=0, k=1, j=0, j=1, i=0, i=1.
 */
const FACE_CORNERS = new Int32Array([
  0, 1, 3, 2,
  4, 5, 7, 6,
  0, 1, 5, 4,
  2, 3, 7, 6,
  0, 2, 6, 4,
  1, 3, 7, 5,
]);
const FACE_EDGES = new Int32Array([
  0, 5, 1, 4,
  2, 7, 3, 6,
  0, 9, 2, 8,
  1, 11, 3, 10,
  4, 10, 6, 8,
  5, 11, 7, 9,
]);

/**
 * Local cube-edge id of a grid edge as seen from each of the four cells sharing
 * it, in the same order as the quad those cells form. Falls out of the corner
 * numbering: the +x grid edge at (i,j,k) runs between corners 0 and 1 of cell
 * (i,j,k), corners 2 and 3 of cell (i,j-1,k), and so on.
 */
const QUAD_LOCAL_EDGE_X = [3, 2, 0, 1] as const;
const QUAD_LOCAL_EDGE_Y = [7, 5, 4, 6] as const;
const QUAD_LOCAL_EDGE_Z = [11, 10, 8, 9] as const;

class FloatBuffer {
  data: Float32Array;
  length = 0;
  constructor(capacity: number) {
    this.data = new Float32Array(Math.max(capacity, 3));
  }
  push3(x: number, y: number, z: number): void {
    if (this.length + 3 > this.data.length) {
      const grown = new Float32Array(this.data.length * 2);
      grown.set(this.data);
      this.data = grown;
    }
    this.data[this.length++] = x;
    this.data[this.length++] = y;
    this.data[this.length++] = z;
  }
  trimmed(): Float32Array {
    return this.data.subarray(0, this.length).slice();
  }
}

class IndexBuffer {
  data: Uint32Array;
  length = 0;
  constructor(capacity: number) {
    this.data = new Uint32Array(Math.max(capacity, 6));
  }
  pushQuad(a: number, b: number, c: number, d: number): void {
    if (this.length + 6 > this.data.length) {
      const grown = new Uint32Array(this.data.length * 2);
      grown.set(this.data);
      this.data = grown;
    }
    const i = this.length;
    this.data[i] = a;
    this.data[i + 1] = b;
    this.data[i + 2] = c;
    this.data[i + 3] = a;
    this.data[i + 4] = c;
    this.data[i + 5] = d;
    this.length = i + 6;
  }
  trimmed(): Uint32Array {
    return this.data.subarray(0, this.length).slice();
  }
}

/**
 * Split a cell's sign-changing cube edges into surface components.
 *
 * A cell would emit one vertex, except the surface can pass through it as two
 * disjoint sheets, and collapsing both onto one vertex leaves an edge shared by
 * four triangles. The reference liver mask hits that seven times, so cells emit
 * one vertex per component instead.
 *
 * The surface inside a cell is a set of closed loops, and two crossing edges are
 * consecutive on a loop when an arc across a shared face joins them. Every face
 * carries an even number of crossings, so pairing them on all six faces and
 * taking connected components recovers the loops with no case table. Four
 * crossings on a face is the classic ambiguous case, settled by the asymptotic
 * decider, the saddle value of the bilinear interpolant on the face: it reads
 * only the four face corner values, so both cells sharing that face always
 * decide the same way, and that agreement is what stitches the two dual meshes
 * into a manifold. `parent` comes back as a union-find over the 12 edge slots,
 * at most four components (the shortest loop uses three edges).
 *
 * One pathological case survives: a component that leaves and re-enters through
 * the same ambiguous face. Cutting it anywhere else needs the neighbour across
 * that face to agree, and it will not, so the mesh gains a boundary instead of a
 * fix. Subdividing the cell is the real answer, and that is manifold dual
 * contouring, deliberately not this. It takes field detail near the sample
 * spacing, so at full resolution only the liver hits it, two edges out of 337k;
 * decimated previews meet it more often (pancreas, strides 2 to 4, one or two
 * edges). The mesh is still closed and consistently oriented, so normals and the
 * divergence-theorem volume hold (liver within 0.3% of voxel count); only
 * "exactly two triangles per edge" is lost, at those edges.
 */
function pairFaceCrossings(g: Float64Array, crossMask: number, parent: Int32Array): void {
  for (let e = 0; e < 12; e++) parent[e] = e;

  for (let f = 0; f < 6; f++) {
    const fe = f * 4;
    const e0 = FACE_EDGES[fe];
    const e1 = FACE_EDGES[fe + 1];
    const e2 = FACE_EDGES[fe + 2];
    const e3 = FACE_EDGES[fe + 3];
    const crossings =
      ((crossMask >> e0) & 1) +
      ((crossMask >> e1) & 1) +
      ((crossMask >> e2) & 1) +
      ((crossMask >> e3) & 1);
    if (crossings === 0) continue;

    if (crossings === 2) {
      let first = -1;
      for (let n = 0; n < 4; n++) {
        const e = FACE_EDGES[fe + n];
        if (((crossMask >> e) & 1) === 0) continue;
        if (first < 0) first = e;
        else union(parent, first, e);
      }
      continue;
    }

    // Ambiguous face. Corners are g0..g3 in cycle order and edge e0 joins g0 to
    // g1, so pairing (e0,e1) and (e2,e3) cuts corners 1 and 3 off individually,
    // while pairing (e1,e2) and (e3,e0) cuts off 0 and 2.
    const g0 = g[FACE_CORNERS[fe]];
    const g1 = g[FACE_CORNERS[fe + 1]];
    const g2 = g[FACE_CORNERS[fe + 2]];
    const g3 = g[FACE_CORNERS[fe + 3]];
    const denom = g0 + g2 - g1 - g3;
    // The diagonal signs of an ambiguous face force denom away from zero; the
    // guard only covers a NaN slipping in from a malformed field.
    const saddleInside = denom !== 0 ? (g0 * g2 - g1 * g3) / denom > 0 : false;
    if ((g0 > 0) === saddleInside) {
      union(parent, e0, e1);
      union(parent, e2, e3);
    } else {
      union(parent, e1, e2);
      union(parent, e3, e0);
    }
  }
}

function findRoot(parent: Int32Array, x: number): number {
  let root = x;
  while (parent[root] !== root) root = parent[root];
  while (parent[x] !== root) {
    const next = parent[x];
    parent[x] = root;
    x = next;
  }
  return root;
}

function union(parent: Int32Array, a: number, b: number): void {
  const ra = findRoot(parent, a);
  const rb = findRoot(parent, b);
  if (ra !== rb) parent[rb] = ra;
}

/**
 * Extract the `isoValue` isosurface of `field`.
 *
 * A cell is the 2x2x2 sample block at (i..i+1, j..j+1, k..k+1), and one whose
 * corners straddle the isovalue emits a vertex per surface component at the mean
 * of that component's interpolated crossings. Each sign-changing grid edge is
 * then shared by four cells, all of which contain it and so have a vertex, and
 * those four are one quad.
 *
 * Winding is counter-clockwise seen from outside, where inside means
 * field > isoValue, so the outward normal points down-gradient.
 *
 * Memory scales with a slice, not the volume: the cell -> vertex lookup is a
 * rolling pair of (nx-1)*(ny-1) buffers holding two words per cell, the index of
 * its first vertex and a 2-bit-per-edge map from cube edge to component.
 */
export function surfaceNets(input: SurfaceNetsInput): RawMesh {
  const { field, dims, spacing, isoValue, origin } = input;
  const [nx, ny, nz] = dims;
  const cnx = nx - 1;
  const cny = ny - 1;
  const cnz = nz - 1;
  if (cnx <= 0 || cny <= 0 || cnz <= 0) {
    return { positions: new Float32Array(0), indices: new Uint32Array(0) };
  }
  if (field.length < nx * ny * nz) {
    throw new Error(`surfaceNets: field has ${field.length} samples, dims need ${nx * ny * nz}`);
  }

  const strideY = nx;
  const strideZ = nx * ny;
  const cornerOffset = new Int32Array([
    0,
    1,
    strideY,
    strideY + 1,
    strideZ,
    strideZ + 1,
    strideZ + strideY,
    strideZ + strideY + 1,
  ]);

  const [sx, sy, sz] = spacing;
  const [ox, oy, oz] = origin;

  // Roughly one vertex per surface cell, so the largest face area seeds it and
  // typical organs never grow the buffers.
  const seedVertices = Math.min(1 << 20, Math.max(1024, cnx * cny));
  const positions = new FloatBuffer(seedVertices * 3);
  const indices = new IndexBuffer(seedVertices * 6);

  const sliceCells = cnx * cny;
  let belowBase = new Int32Array(sliceCells);
  let belowMap = new Int32Array(sliceCells);
  let curBase = new Int32Array(sliceCells);
  let curMap = new Int32Array(sliceCells);

  // `g` is corner values relative to the isovalue, so the sign tests and the
  // face decider agree by construction.
  const g = new Float64Array(8);
  const parent = new Int32Array(12);
  const componentOfRoot = new Int32Array(12);
  const crossX = new Float64Array(12);
  const crossY = new Float64Array(12);
  const crossZ = new Float64Array(12);
  const sumX = new Float64Array(4);
  const sumY = new Float64Array(4);
  const sumZ = new Float64Array(4);
  const sumN = new Int32Array(4);

  for (let k = 0; k < cnz; k++) {
    for (let j = 0; j < cny; j++) {
      const rowBase = k * strideZ + j * strideY;
      const cellRow = j * cnx;
      for (let i = 0; i < cnx; i++) {
        const slot = cellRow + i;
        const base = rowBase + i;
        // Sign test only: 97% of the reference liver crop dies on the next line
        // and filling `g` for those cells was a quarter of this pass. Safe
        // because fl(a - b) > 0 is exactly a > b for doubles.
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          if (field[base + cornerOffset[c]] > isoValue) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) {
          curBase[slot] = -1;
          curMap[slot] = 0;
          continue;
        }
        for (let c = 0; c < 8; c++) g[c] = field[base + cornerOffset[c]] - isoValue;

        let crossMask = 0;
        for (let e = 0; e < 12; e++) {
          const a = EDGE_A[e];
          const b = EDGE_B[e];
          if (((mask >> a) & 1) === ((mask >> b) & 1)) continue;
          crossMask |= 1 << e;
          const va = g[a];
          const denom = g[b] - va;
          // denom cannot be 0 while the endpoints straddle the isovalue, but a
          // NaN in the field would slip through the sign test, so clamp.
          let t = denom !== 0 ? -va / denom : 0.5;
          if (!(t >= 0 && t <= 1)) t = 0.5;
          const ax = a & 1;
          const ay = (a >> 1) & 1;
          const az = (a >> 2) & 1;
          crossX[e] = ax + t * ((b & 1) - ax);
          crossY[e] = ay + t * (((b >> 1) & 1) - ay);
          crossZ[e] = az + t * (((b >> 2) & 1) - az);
        }

        pairFaceCrossings(g, crossMask, parent);

        let components = 0;
        let edgeToComponent = 0;
        for (let e = 0; e < 12; e++) componentOfRoot[e] = -1;
        for (let e = 0; e < 12; e++) {
          if (((crossMask >> e) & 1) === 0) continue;
          const root = findRoot(parent, e);
          let component = componentOfRoot[root];
          if (component < 0) {
            component = components++;
            componentOfRoot[root] = component;
            sumX[component] = 0;
            sumY[component] = 0;
            sumZ[component] = 0;
            sumN[component] = 0;
          }
          edgeToComponent |= component << (e * 2);
          sumX[component] += crossX[e];
          sumY[component] += crossY[e];
          sumZ[component] += crossZ[e];
          sumN[component]++;
        }

        const vertexBase = positions.length / 3;
        for (let c = 0; c < components; c++) {
          const inv = 1 / sumN[c];
          positions.push3(
            (i + sumX[c] * inv + ox) * sx,
            (j + sumY[c] * inv + oy) * sy,
            (k + sumZ[c] * inv + oz) * sz,
          );
        }
        curBase[slot] = vertexBase;
        curMap[slot] = edgeToComponent;

        // Quads for the three grid edges leaving corner 0. Which way round the
        // four neighbouring cells are wound depends on corner 0 being inside:
        // the orders below have their normal along +x, +y and +z, outward
        // exactly when corner 0 is the inside end of the crossing.
        const inside0 = (mask & 1) !== 0;

        if (j > 0 && k > 0 && inside0 !== ((mask & 2) !== 0)) {
          const a = vertexAt(belowBase, belowMap, slot - cnx, QUAD_LOCAL_EDGE_X[0]);
          const b = vertexAt(belowBase, belowMap, slot, QUAD_LOCAL_EDGE_X[1]);
          const c = vertexAt(curBase, curMap, slot, QUAD_LOCAL_EDGE_X[2]);
          const d = vertexAt(curBase, curMap, slot - cnx, QUAD_LOCAL_EDGE_X[3]);
          if (inside0) indices.pushQuad(a, b, c, d);
          else indices.pushQuad(a, d, c, b);
        }
        if (i > 0 && k > 0 && inside0 !== ((mask & 4) !== 0)) {
          const a = vertexAt(belowBase, belowMap, slot - 1, QUAD_LOCAL_EDGE_Y[0]);
          const b = vertexAt(curBase, curMap, slot - 1, QUAD_LOCAL_EDGE_Y[1]);
          const c = vertexAt(curBase, curMap, slot, QUAD_LOCAL_EDGE_Y[2]);
          const d = vertexAt(belowBase, belowMap, slot, QUAD_LOCAL_EDGE_Y[3]);
          if (inside0) indices.pushQuad(a, b, c, d);
          else indices.pushQuad(a, d, c, b);
        }
        if (i > 0 && j > 0 && inside0 !== ((mask & 16) !== 0)) {
          const a = vertexAt(curBase, curMap, slot - cnx - 1, QUAD_LOCAL_EDGE_Z[0]);
          const b = vertexAt(curBase, curMap, slot - cnx, QUAD_LOCAL_EDGE_Z[1]);
          const c = vertexAt(curBase, curMap, slot, QUAD_LOCAL_EDGE_Z[2]);
          const d = vertexAt(curBase, curMap, slot - 1, QUAD_LOCAL_EDGE_Z[3]);
          if (inside0) indices.pushQuad(a, b, c, d);
          else indices.pushQuad(a, d, c, b);
        }
      }
    }
    const swapBase = belowBase;
    const swapMap = belowMap;
    belowBase = curBase;
    belowMap = curMap;
    curBase = swapBase;
    curMap = swapMap;
  }

  return { positions: positions.trimmed(), indices: indices.trimmed() };
}

/** Vertex a cell contributes to one of its cube edges, via the 2-bit map. */
function vertexAt(
  bases: Int32Array,
  maps: Int32Array,
  slot: number,
  localEdge: number,
): number {
  return bases[slot] + ((maps[slot] >>> (localEdge * 2)) & 3);
}

/**
 * Area-weighted vertex normals. The un-normalised cross product has length
 * 2*area, so accumulating it raw is the weighting, and it has to be weighted:
 * surface nets emits dense clusters of tiny triangles wherever the surface
 * bends, and with unit face normals those clusters outvote the large flat faces
 * they border and visibly tilt the shading across smooth regions.
 */
export function computeNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length);
  for (let t = 0; t < indices.length; t += 3) {
    const i0 = indices[t] * 3;
    const i1 = indices[t + 1] * 3;
    const i2 = indices[t + 2] * 3;

    const ax = positions[i1] - positions[i0];
    const ay = positions[i1 + 1] - positions[i0 + 1];
    const az = positions[i1 + 2] - positions[i0 + 2];
    const bx = positions[i2] - positions[i0];
    const by = positions[i2 + 1] - positions[i0 + 1];
    const bz = positions[i2 + 2] - positions[i0 + 2];

    const nx = ay * bz - az * by;
    const ny = az * bx - ax * bz;
    const nz = ax * by - ay * bx;

    normals[i0] += nx; normals[i0 + 1] += ny; normals[i0 + 2] += nz;
    normals[i1] += nx; normals[i1 + 1] += ny; normals[i1 + 2] += nz;
    normals[i2] += nx; normals[i2 + 1] += ny; normals[i2 + 2] += nz;
  }
  for (let n = 0; n < normals.length; n += 3) {
    const len = Math.hypot(normals[n], normals[n + 1], normals[n + 2]);
    if (len > 0) {
      const inv = 1 / len;
      normals[n] *= inv;
      normals[n + 1] *= inv;
      normals[n + 2] *= inv;
    }
  }
  return normals;
}

/** Axis-aligned bounds [minX,minY,minZ,maxX,maxY,maxZ]. Empty input gives zeros. */
export function meshBounds(
  positions: Float32Array,
): [number, number, number, number, number, number] {
  if (positions.length === 0) return [0, 0, 0, 0, 0, 0];
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let p = 0; p < positions.length; p += 3) {
    const x = positions[p];
    const y = positions[p + 1];
    const z = positions[p + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  return [minX, minY, minZ, maxX, maxY, maxZ];
}

export interface ExtractOrganMeshOptions {
  /** Full-volume mask; any non-zero value inside `bounds` counts as inside. */
  mask: Uint8Array;
  dims: [number, number, number];
  spacing: [number, number, number];
  /** Inclusive voxel bbox [i0,j0,k0,i1,j1,k1]; may be looser than the true one. */
  bounds: [number, number, number, number, number, number];
  blurPasses?: number;
  smoothIterations?: number;
  /** > 1 builds a coarse preview mesh from a downsampled field. */
  decimateStride?: number;
}

/** Inclusive voxel box [i0,j0,k0,i1,j1,k1]. */
type Box = [number, number, number, number, number, number];

/** Occupancy above this is inside. Fixed by the 0/1 nature of a mask. */
const ISO_VALUE = 0.5;

export interface OrganMesh {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  /** Mesh extent in volume local mm. */
  bounds: [number, number, number, number, number, number];
  triangleCount: number;
}

function emptyMesh(): OrganMesh {
  return {
    positions: new Float32Array(0),
    normals: new Float32Array(0),
    indices: new Uint32Array(0),
    bounds: [0, 0, 0, 0, 0, 0],
    triangleCount: 0,
  };
}

/**
 * Mask -> smooth closed triangle mesh, in volume local mm.
 *
 * Crop, occupancy field, blur, surface nets at 0.5, Taubin smooth, normals. No
 * welding step: surface nets already emits one shared vertex per cell.
 *
 * The padding is what keeps the mesh closed, so it has to be at least as wide as
 * the blur reaches (one voxel per pass) or the blurred field is still above the
 * isovalue at the array border and the surface runs off it. Samples outside the
 * volume read as background rather than clamped, because organ masks routinely
 * touch the scan boundary (the reference liver starts at k = 0) and a clamped
 * crop would leave the surface open right there.
 *
 * Nothing outside `bounds` is read. That is a contract, not an optimisation:
 * callers share one mask buffer between structures, and a halo of whatever the
 * previous structure left behind would be welded straight onto this mesh (the
 * pancreas box has eleven thousand stomach voxels within two voxels of it).
 */
export function extractOrganMesh(opts: ExtractOrganMeshOptions): OrganMesh {
  const { mask, dims, spacing, bounds } = opts;
  const requestedPasses = Math.max(0, Math.round(opts.blurPasses ?? 2));
  const smoothIterations = opts.smoothIterations ?? 12;
  const stride = Math.max(1, Math.floor(opts.decimateStride ?? 1));

  const [nx, ny, nz] = dims;
  const voxelCount = nx * ny * nz;
  // Out-of-range reads on a typed array give undefined, and `undefined !== 0` is
  // true, so a short mask would mesh as solid tissue instead of failing. Catch
  // it here rather than shipping fabricated anatomy.
  if (mask.length < voxelCount) {
    throw new Error(`extractOrganMesh: mask has ${mask.length} voxels, dims need ${voxelCount}`);
  }

  const [i0, j0, k0, i1, j1, k1] = bounds;
  if (i1 < i0 || j1 < j0 || k1 < k0) return emptyMesh();
  const clip: Box = [
    Math.max(0, i0), Math.max(0, j0), Math.max(0, k0),
    Math.min(nx - 1, i1), Math.min(ny - 1, j1), Math.min(nz - 1, k1),
  ];

  const blurPasses = blurPassesForStride(requestedPasses, stride);
  const pad = Math.max(2, blurPasses) * stride;
  const lo: [number, number, number] = [i0 - pad, j0 - pad, k0 - pad];
  // Round up to whole stride blocks; the slack lands in the zero padding.
  const coarse: [number, number, number] = [
    Math.ceil((i1 + pad - lo[0] + 1) / stride),
    Math.ceil((j1 + pad - lo[1] + 1) / stride),
    Math.ceil((k1 + pad - lo[2] + 1) / stride),
  ];
  if (coarse[0] < 2 || coarse[1] < 2 || coarse[2] < 2) return emptyMesh();

  const field = sampleOccupancy(mask, dims, clip, lo, coarse, stride);

  // A coarse sample stands for the centre of its stride block, so the field
  // origin shifts by half a block. With stride 1 this is exactly `lo`.
  const half = (stride - 1) / 2;
  const meshAtBlur = (passes: number): RawMesh =>
    surfaceNets({
      field: boxBlur3(field, coarse, passes),
      dims: coarse,
      spacing: [spacing[0] * stride, spacing[1] * stride, spacing[2] * stride],
      isoValue: ISO_VALUE,
      origin: [
        (lo[0] + half) / stride,
        (lo[1] + half) / stride,
        (lo[2] + half) / stride,
      ],
    });

  let raw = meshAtBlur(blurPasses);
  // Blur before threshold means anything thinner than about 2 * blurPasses
  // voxels never reaches the isovalue and comes back empty, byte for byte what
  // an empty mask returns. Drop the blur, not the structure. Backing off one
  // pass at a time keeps whichever sliver survives, and that sliver is not the
  // structure: a 3-voxel cube meshes at 11% of its volume at one pass, 72% at none.
  if (raw.indices.length === 0 && blurPasses > 0 && maxSample(field) > ISO_VALUE) {
    raw = meshAtBlur(0);
  }
  if (raw.indices.length === 0) return emptyMesh();

  const positions = taubinSmooth(raw.positions, raw.indices, smoothIterations);
  return {
    positions,
    normals: computeNormals(positions, raw.indices),
    indices: raw.indices,
    bounds: meshBounds(positions),
    triangleCount: raw.indices.length / 3,
  };
}

/**
 * Blur passes on a stride-`s` grid matching what `passes` would do at full
 * resolution. A radius-1 box pass has variance 2/3 in grid units, so (2/3)s^2
 * whole voxels, and the s^3 block average contributes (s^2 - 1)/12 on its own.
 *
 * Matching total variance is what keeps a preview the size of the mesh it stands
 * in for: the 0.5 crossing moves inward on a convex boundary in proportion to
 * the variance, so a fixed pass count cost the gall bladder 80% of its volume at
 * stride 4, where decimation alone costs 17%.
 */
function blurPassesForStride(passes: number, stride: number): number {
  const perPass = (2 / 3) * stride * stride;
  const fromBlocks = (stride * stride - 1) / 12;
  return Math.max(0, Math.round(((2 / 3) * passes - fromBlocks) / perPass));
}

function maxSample(field: Float32Array): number {
  let max = -Infinity;
  for (let i = 0; i < field.length; i++) if (field[i] > max) max = field[i];
  return max;
}

/**
 * Crop the mask into a Float32 occupancy field, averaging each stride^3 block.
 * Averaging rather than point sampling because a binary mask point-sampled at
 * stride 2 or more loses thin structures outright (the aorta is barely three
 * voxels across in plane) and aliases the rest into blocky noise.
 *
 * Voxels outside the volume and voxels outside `clip` both read as background:
 * the first is the zero border the isosurface needs to close, the second keeps a
 * shared mask buffer's leftovers out of the mesh.
 */
function sampleOccupancy(
  mask: Uint8Array,
  dims: [number, number, number],
  clip: Box,
  lo: [number, number, number],
  coarse: [number, number, number],
  stride: number,
): Float32Array {
  return stride === 1
    ? cropOccupancy(mask, dims, clip, lo, coarse)
    : blockOccupancy(mask, dims, clip, lo, coarse, stride);
}

/** Walks the source box, not the destination grid, so padding never enters the loop. */
function cropOccupancy(
  mask: Uint8Array,
  dims: [number, number, number],
  clip: Box,
  lo: [number, number, number],
  coarse: [number, number, number],
): Float32Array {
  const [nx, ny] = dims;
  const [cnx, cny, cnz] = coarse;
  const field = new Float32Array(cnx * cny * cnz);
  const [ci0, cj0, ck0, ci1, cj1, ck1] = clip;
  if (ci1 < ci0 || cj1 < cj0 || ck1 < ck0) return field;

  const planeStride = nx * ny;
  const x0 = Math.max(ci0, lo[0]);
  const x1 = Math.min(ci1, lo[0] + cnx - 1);
  const y0 = Math.max(cj0, lo[1]);
  const y1 = Math.min(cj1, lo[1] + cny - 1);
  const z0 = Math.max(ck0, lo[2]);
  const z1 = Math.min(ck1, lo[2] + cnz - 1);

  for (let z = z0; z <= z1; z++) {
    const outZ = (z - lo[2]) * cny;
    for (let y = y0; y <= y1; y++) {
      const src = z * planeStride + y * nx;
      const out = (outZ + y - lo[1]) * cnx - lo[0];
      for (let x = x0; x <= x1; x++) {
        if (mask[src + x] !== 0) field[out + x] = 1;
      }
    }
  }
  return field;
}

function blockOccupancy(
  mask: Uint8Array,
  dims: [number, number, number],
  clip: Box,
  lo: [number, number, number],
  coarse: [number, number, number],
  stride: number,
): Float32Array {
  const [nx, ny] = dims;
  const [cnx, cny, cnz] = coarse;
  const field = new Float32Array(cnx * cny * cnz);
  const [ci0, cj0, ck0, ci1, cj1, ck1] = clip;
  if (ci1 < ci0 || cj1 < cj0 || ck1 < ck0) return field;

  const planeStride = nx * ny;
  const invBlock = 1 / (stride * stride * stride);

  // A column's source span depends only on cx, so clamp once per column.
  const colStart = new Int32Array(cnx);
  const colCount = new Int32Array(cnx);
  for (let cx = 0; cx < cnx; cx++) {
    const first = Math.max(lo[0] + cx * stride, ci0);
    const last = Math.min(lo[0] + cx * stride + stride - 1, ci1);
    colStart[cx] = first;
    colCount[cx] = Math.max(0, last - first + 1);
  }

  const rowSums = new Int32Array(cnx);
  for (let cz = 0; cz < cnz; cz++) {
    const z0 = Math.max(lo[2] + cz * stride, ck0);
    const z1 = Math.min(lo[2] + cz * stride + stride - 1, ck1);
    if (z1 < z0) continue;
    for (let cy = 0; cy < cny; cy++) {
      const y0 = Math.max(lo[1] + cy * stride, cj0);
      const y1 = Math.min(lo[1] + cy * stride + stride - 1, cj1);
      if (y1 < y0) continue;
      rowSums.fill(0);
      for (let z = z0; z <= z1; z++) {
        for (let y = y0; y <= y1; y++) {
          const rowBase = z * planeStride + y * nx;
          for (let cx = 0; cx < cnx; cx++) {
            const start = rowBase + colStart[cx];
            const count = colCount[cx];
            let sum = 0;
            for (let d = 0; d < count; d++) {
              if (mask[start + d] !== 0) sum++;
            }
            rowSums[cx] += sum;
          }
        }
      }
      const out = (cz * cny + cy) * cnx;
      for (let cx = 0; cx < cnx; cx++) field[out + cx] = rowSums[cx] * invBlock;
    }
  }
  return field;
}
