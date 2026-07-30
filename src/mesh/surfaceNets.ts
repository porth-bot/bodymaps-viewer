/**
 * Naive surface nets (dual contouring on the cell grid) for binary organ masks.
 *
 * Chosen over marching cubes on purpose:
 *
 *   - No 256-entry case table. Marching cubes needs a hand-transcribed lookup
 *     table whose errors show up as single missing triangles, which are close
 *     to impossible to spot by eye and painful to debug. The cube topology here
 *     is derived from two small tables of cube edges and faces instead.
 *   - Manifold. Every sign-changing grid edge emits exactly one quad joining
 *     the four cells around it, so the result is closed and consistently
 *     oriented with no boundary and no ambiguous-face special cases.
 *   - Better triangle quality on label fields. Marching cubes on a blurred
 *     binary mask produces long slivers wherever a crossing sits near a corner;
 *     the dual formulation puts one well-centred vertex per cell instead.
 *
 * The trade is that surface nets does not reproduce sharp features, which is
 * exactly right for anatomy.
 *
 * One refinement over the textbook version is needed to make "manifold" true
 * rather than nearly true. A cell normally emits a single vertex, but the
 * surface can pass through one cell as two disjoint sheets, and collapsing both
 * onto one vertex produces an edge shared by four triangles. That is not
 * hypothetical: the reference liver mask hits it seven times. So each cell is
 * partitioned into its surface components first (see `pairFaceCrossings`) and
 * emits one vertex per component. In the overwhelmingly common case there is
 * exactly one component and this costs nothing but the bookkeeping. With it,
 * all nine reference organ masks come out strictly manifold.
 *
 * Known limit: one pathological case survives, where a single surface component
 * of a cell leaves and re-enters through the same ambiguous face. Splitting that
 * cell would mean cutting the component at some other face, which the
 * neighbouring cell across that face would not agree to, and the mesh would gain
 * a boundary instead. Properly fixing it means subdividing the cell, which is
 * what manifold dual contouring does and what this deliberately does not. It
 * needs field noise comparable to the voxel size to occur at all (never at full
 * resolution on the reference data, once on a stride-3 preview of a kidney), and
 * when it does the mesh is still closed and consistently oriented, so normals
 * and the divergence-theorem volume stay correct. Only the "exactly two
 * triangles per edge" property is lost, at that one edge.
 *
 * COORDINATE CONTRACT: everything here works in "volume local millimetres",
 * defined as voxelIndex * spacing with the origin at voxel (0,0,0). The world
 * affine is deliberately not applied; the renderer stays in this local space.
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
 * The six cube faces, each as a cycle of four corners and the matching cycle of
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
 * Local cube-edge id of a grid edge as seen from each of the four cells that
 * share it, listed in the same order as the quad those cells form. Derived
 * from the corner numbering: the grid edge at sample (i,j,k) along +x is the
 * cell-local edge between corners 0 and 1 of cell (i,j,k), between corners 2
 * and 3 of cell (i,j-1,k), and so on.
 */
const QUAD_LOCAL_EDGE_X = [3, 2, 0, 1] as const;
const QUAD_LOCAL_EDGE_Y = [7, 5, 4, 6] as const;
const QUAD_LOCAL_EDGE_Z = [11, 10, 8, 9] as const;

/** Growable Float32 backing store for interleaved xyz positions. */
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

/** Growable Uint32 backing store for triangle indices. */
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
 * The surface inside one cell is a set of closed loops, and each loop is a
 * cycle of crossing cube edges: two crossing edges are consecutive on a loop
 * when they are joined by an arc across a shared cube face. Every face carries
 * an even number of crossings, so pairing them up on all six faces and taking
 * connected components recovers the loops without ever consulting a case table.
 *
 * A face with two crossings has only one possible pairing. A face with four is
 * the classic ambiguous case: its two inside corners sit diagonally, and the
 * surface can either pinch them apart or join them through the middle. That is
 * decided with the asymptotic decider, the saddle value of the bilinear
 * interpolant on the face. The decision depends only on the four face corner
 * values, so the two cells sharing the face always agree, which is exactly what
 * keeps the two dual meshes stitching together into a manifold.
 *
 * Fills `parent` as a union-find over the 12 edge slots. A cell has at most
 * four components, since the shortest possible loop cuts off a single corner
 * and so uses three edges.
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

    // Two crossings admit only one pairing, so simply chain them together.
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

    // Ambiguous face. Corners are g0..g3 in cycle order, edge e0 joins g0 to
    // g1 and so on, so pairing (e0,e1) and (e2,e3) cuts corners 1 and 3 off
    // individually, while pairing (e1,e2) and (e3,e0) cuts off 0 and 2.
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
 * Two phases share one cell loop:
 *
 *   1. Cell vertices. A cell is the 2x2x2 block of samples at (i..i+1,
 *      j..j+1, k..k+1). If its 8 corners are not all on the same side of the
 *      isovalue the cell straddles the surface: every sign-changing cube edge
 *      contributes its linearly interpolated crossing point, and the cell emits
 *      one vertex per surface component at the average of that component's
 *      crossings.
 *   2. Faces. Each grid edge that changes sign is shared by exactly four cells,
 *      all of which are guaranteed to have emitted a vertex (they all contain
 *      the crossing edge). Those four vertices form one quad of the surface.
 *
 * Winding is counter-clockwise seen from outside, where inside means
 * field > isoValue, so the outward normal points down-gradient.
 *
 * Memory is proportional to a slice, not to the volume: the cell -> vertex
 * lookup is a rolling pair of (nx-1)*(ny-1) buffers rather than a full-volume
 * array. Each cell needs two words there, the index of its first vertex and a
 * 2-bit-per-edge map from cube edge to component, which is enough to answer
 * "which of this cell's vertices does that grid edge belong to".
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

  // A closed surface through a box of cells has roughly one vertex per surface
  // cell; seed with the area of the largest face so typical organs never grow.
  const seedVertices = Math.min(1 << 20, Math.max(1024, cnx * cny));
  const positions = new FloatBuffer(seedVertices * 3);
  const indices = new IndexBuffer(seedVertices * 6);

  const sliceCells = cnx * cny;
  let belowBase = new Int32Array(sliceCells);
  let belowMap = new Int32Array(sliceCells);
  let curBase = new Int32Array(sliceCells);
  let curMap = new Int32Array(sliceCells);

  // Per-cell scratch, allocated once. `g` holds corner values relative to the
  // isovalue so the sign tests and the face decider agree by construction.
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
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const v = field[base + cornerOffset[c]] - isoValue;
          g[c] = v;
          if (v > 0) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) {
          curBase[slot] = -1;
          curMap[slot] = 0;
          continue;
        }

        // Interpolate the crossings. Only surface cells reach this point, so
        // everything below scales with surface area, not with volume.
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

        // Quads for the three grid edges leaving corner 0 of this cell. Corner
        // 0 is inside when bit 0 of the mask is set, and that decides which way
        // round the four neighbouring cells are wound: the orders below have
        // their normal along +x, +y and +z, which is the outward direction
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
 * Area-weighted vertex normals.
 *
 * The un-normalised triangle cross product has length 2*area, so accumulating
 * it directly weights each face by its area. That matters here because surface
 * nets emits dense clusters of tiny triangles wherever the surface bends: with
 * per-face unit normals those clusters would outvote the large flat faces they
 * border and visibly tilt the shading across smooth regions.
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
  /** Full-volume mask; any non-zero value counts as inside. */
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
 * The mask is cropped to its bounding box padded by two voxels (times the
 * decimation stride), converted to a Float32 occupancy field, blurred, run
 * through surface nets at 0.5, Taubin smoothed, and normalled. No welding step
 * is needed: surface nets already emits exactly one shared vertex per cell.
 *
 * The padding is what keeps the mesh closed. It is applied in field space and
 * samples outside the volume read as background rather than being clamped away,
 * because organ masks routinely touch the scan boundary (the liver in the
 * reference case starts at k = 0) and a clamped crop would leave the surface
 * open right there, breaking both manifoldness and any volume measurement.
 */
export function extractOrganMesh(opts: ExtractOrganMeshOptions): OrganMesh {
  const { mask, dims, spacing, bounds } = opts;
  const blurPasses = opts.blurPasses ?? 2;
  const smoothIterations = opts.smoothIterations ?? 12;
  const stride = Math.max(1, Math.floor(opts.decimateStride ?? 1));

  const [nx, ny, nz] = dims;
  const [i0, j0, k0, i1, j1, k1] = bounds;
  if (i1 < i0 || j1 < j0 || k1 < k0) return emptyMesh();

  const pad = 2 * stride;
  const lo: [number, number, number] = [i0 - pad, j0 - pad, k0 - pad];
  // Round the crop up to a whole number of stride blocks so the coarse grid is
  // exact; the slack lands inside the zero padding and costs nothing.
  const coarse: [number, number, number] = [
    Math.ceil((i1 + pad - lo[0] + 1) / stride),
    Math.ceil((j1 + pad - lo[1] + 1) / stride),
    Math.ceil((k1 + pad - lo[2] + 1) / stride),
  ];
  if (coarse[0] < 2 || coarse[1] < 2 || coarse[2] < 2) return emptyMesh();

  const field = sampleOccupancy(mask, nx, ny, nz, lo, coarse, stride);
  const blurred = boxBlur3(field, coarse, blurPasses);

  // A coarse sample stands for the centre of its stride block, so the field
  // origin shifts by half a block. With stride 1 this is exactly `lo`.
  const half = (stride - 1) / 2;
  const raw = surfaceNets({
    field: blurred,
    dims: coarse,
    spacing: [spacing[0] * stride, spacing[1] * stride, spacing[2] * stride],
    isoValue: 0.5,
    origin: [
      (lo[0] + half) / stride,
      (lo[1] + half) / stride,
      (lo[2] + half) / stride,
    ],
  });
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
 * Crop the mask into a Float32 occupancy field, averaging each stride^3 block.
 *
 * Block averaging rather than point sampling for the decimated case: a binary
 * mask point-sampled at stride 2 or more loses thin structures outright (the
 * aorta is barely three voxels across in plane) and aliases the rest into
 * blocky noise. The average costs one pass over the crop, which is the cheap
 * part next to blurring and meshing.
 *
 * Samples outside the volume are background, which is what guarantees the
 * zero border the isosurface needs to close.
 */
function sampleOccupancy(
  mask: Uint8Array,
  nx: number,
  ny: number,
  nz: number,
  lo: [number, number, number],
  coarse: [number, number, number],
  stride: number,
): Float32Array {
  const [cnx, cny, cnz] = coarse;
  const field = new Float32Array(cnx * cny * cnz);
  const invBlock = 1 / (stride * stride * stride);
  const strideZ = nx * ny;

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
          if (z < 0 || z >= nz) continue;
          for (let dy = 0; dy < stride; dy++) {
            const y = yBase + dy;
            if (y < 0 || y >= ny) continue;
            const rowBase = z * strideZ + y * nx;
            const xStart = Math.max(0, -xBase);
            const xEnd = Math.min(stride, nx - xBase);
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
