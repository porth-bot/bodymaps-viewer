/**
 * Taubin lambda/mu mesh smoothing.
 *
 * Plain Laplacian smoothing attenuates every non-zero frequency, so a closed
 * surface shrinks a little on every iteration and a smoothed organ ends up
 * visibly smaller than its mask. Taubin alternates the shrinking step with a
 * negative one (mu < 0, |mu| > lambda), giving a gain above 1 at low frequencies
 * and below 1 at high ones: noise still dies, the overall shape holds its size.
 * The crossover sits at 1/lambda + 1/mu, which is where the defaults come from:
 * they put it at 0.11, so everything but the lowest frequencies is attenuated.
 */

/**
 * Smooth `positions`. The input is never modified.
 *
 * `iterations` counts individual steps, not lambda/mu pairs, and they alternate
 * starting with lambda. Pass an even number so the sequence ends on a mu step,
 * otherwise the last shrink is left uncompensated.
 *
 * Neighbours come from the directed triangle edges. On a closed, consistently
 * oriented mesh each triangle around a vertex contributes exactly one outgoing
 * edge, so every neighbour appears exactly once and the uniform weights are
 * genuinely uniform with no deduplication pass.
 */
export function taubinSmooth(
  positions: Float32Array,
  indices: Uint32Array,
  iterations: number,
  lambda = 0.5,
  mu = -0.53,
): Float32Array {
  const vertexCount = positions.length / 3;
  if (iterations <= 0 || vertexCount === 0 || indices.length === 0) {
    return positions.slice();
  }

  const { offsets, adjacency } = buildAdjacency(indices, vertexCount);

  let cur = positions.slice();
  let next = new Float32Array(positions.length);

  for (let it = 0; it < iterations; it++) {
    const factor = it % 2 === 0 ? lambda : mu;
    for (let v = 0; v < vertexCount; v++) {
      const start = offsets[v];
      const end = offsets[v + 1];
      const p = v * 3;
      const degree = end - start;
      if (degree === 0) {
        // Isolated vertex, only reachable if the caller passed a mesh with
        // unreferenced positions. Leave it where it is.
        next[p] = cur[p];
        next[p + 1] = cur[p + 1];
        next[p + 2] = cur[p + 2];
        continue;
      }
      let sx = 0;
      let sy = 0;
      let sz = 0;
      for (let a = start; a < end; a++) {
        const q = adjacency[a] * 3;
        sx += cur[q];
        sy += cur[q + 1];
        sz += cur[q + 2];
      }
      const inv = 1 / degree;
      next[p] = cur[p] + factor * (sx * inv - cur[p]);
      next[p + 1] = cur[p + 1] + factor * (sy * inv - cur[p + 1]);
      next[p + 2] = cur[p + 2] + factor * (sz * inv - cur[p + 2]);
    }
    const swap = cur;
    cur = next;
    next = swap;
  }
  return cur;
}

interface Adjacency {
  /** offsets[v] .. offsets[v+1] indexes the neighbours of v in `adjacency`. */
  offsets: Int32Array;
  adjacency: Int32Array;
}

/** CSR adjacency, built once: rebuilding it per iteration dominated everything else. */
function buildAdjacency(indices: Uint32Array, vertexCount: number): Adjacency {
  const offsets = new Int32Array(vertexCount + 1);

  for (let t = 0; t < indices.length; t += 3) {
    offsets[indices[t]]++;
    offsets[indices[t + 1]]++;
    offsets[indices[t + 2]]++;
  }

  let total = 0;
  for (let v = 0; v <= vertexCount; v++) {
    const c = offsets[v];
    offsets[v] = total;
    total += c;
  }

  const adjacency = new Int32Array(total);
  const cursor = offsets.slice(0, vertexCount);
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    adjacency[cursor[a]++] = b;
    adjacency[cursor[b]++] = c;
    adjacency[cursor[c]++] = a;
  }
  return { offsets, adjacency };
}
