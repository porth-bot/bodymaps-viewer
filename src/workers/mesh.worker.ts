/// <reference lib="webworker" />
/**
 * Builds one organ surface at a time from the packed label volume. Initialised
 * once with a copy of the volume rather than getting it per request: it is
 * 12 MB, and three workers re-cloning it for nine organs move 300 MB.
 */

import { extractOrganMesh } from '../mesh/surfaceNets';

export interface MeshInit {
  type: 'init';
  dims: [number, number, number];
  spacing: [number, number, number];
  values: Uint8Array;
}

export interface MeshBuildRequest {
  type: 'build';
  index: number;
  /** Inclusive voxel bounding box from the structure statistics. */
  bounds: [number, number, number, number, number, number];
  /** 1 = full resolution. 2 halves the sampling rate on each axis for a fast preview. */
  stride?: number;
  smoothIterations?: number;
  blurPasses?: number;
  token: number;
}

export type MeshMessage =
  | {
      type: 'mesh';
      token: number;
      index: number;
      positions: Float32Array;
      normals: Float32Array;
      indices: Uint32Array;
      bounds: [number, number, number, number, number, number];
      triangleCount: number;
      ms: number;
    }
  | { type: 'error'; token: number; index: number; message: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

let dims: [number, number, number] | null = null;
let spacing: [number, number, number] | null = null;
let labels: Uint8Array | null = null;

/** Scratch mask, reused so nine organs are not nine 12 MB allocations. */
let scratch: Uint8Array | null = null;

ctx.onmessage = (event: MessageEvent<MeshInit | MeshBuildRequest>) => {
  const msg = event.data;

  if (msg.type === 'init') {
    dims = msg.dims;
    spacing = msg.spacing;
    labels = msg.values;
    scratch = new Uint8Array(msg.values.length);
    return;
  }

  if (msg.type !== 'build') return;

  try {
    if (!dims || !spacing || !labels || !scratch) {
      throw new Error('Mesh worker received a build request before its label volume');
    }

    const t0 = performance.now();

    // Only the bounding box gets cleared and filled. Touching all 12 million
    // voxels for a 29 mL aorta costs more than the extraction itself.
    const [i0, j0, k0, i1, j1, k1] = msg.bounds;
    if (i1 < i0 || j1 < j0 || k1 < k0) {
      ctx.postMessage({
        type: 'mesh', token: msg.token, index: msg.index,
        positions: new Float32Array(0), normals: new Float32Array(0), indices: new Uint32Array(0),
        bounds: [0, 0, 0, 0, 0, 0], triangleCount: 0, ms: 0,
      } satisfies MeshMessage);
      return;
    }

    const [nx, ny] = dims;
    const target = msg.index;
    for (let k = k0; k <= k1; k++) {
      const kOff = k * nx * ny;
      for (let j = j0; j <= j1; j++) {
        const row = kOff + j * nx;
        for (let i = i0; i <= i1; i++) {
          scratch[row + i] = labels[row + i] === target ? 1 : 0;
        }
      }
    }

    // These masks come out of a model that segments axially, so adjacent slices
    // disagree slightly and the raw isosurface terraces along z, which 2.5 mm
    // spacing makes very visible. Three blur passes plus a longer Taubin run
    // flattens the steps, and Taubin is volume preserving, so the extra
    // iterations cost shape fidelity far more slowly than Laplacian would.
    const mesh = extractOrganMesh({
      mask: scratch,
      dims,
      spacing,
      bounds: msg.bounds,
      decimateStride: msg.stride ?? 1,
      blurPasses: msg.blurPasses ?? 3,
      smoothIterations: msg.smoothIterations ?? 24,
    });

    const message: MeshMessage = {
      type: 'mesh',
      token: msg.token,
      index: msg.index,
      positions: mesh.positions,
      normals: mesh.normals,
      indices: mesh.indices,
      bounds: mesh.bounds,
      triangleCount: mesh.triangleCount,
      ms: performance.now() - t0,
    };
    ctx.postMessage(message, [mesh.positions.buffer, mesh.normals.buffer, mesh.indices.buffer]);

    // Zero only what was written, ready for the next structure.
    for (let k = k0; k <= k1; k++) {
      const kOff = k * nx * ny;
      for (let j = j0; j <= j1; j++) {
        scratch.fill(0, kOff + j * nx + i0, kOff + j * nx + i1 + 1);
      }
    }
  } catch (err) {
    ctx.postMessage({
      type: 'error',
      token: msg.token,
      index: msg.index,
      message: err instanceof Error ? err.message : String(err),
    } satisfies MeshMessage);
  }
};
