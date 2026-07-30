/**
 * Separable box blur for scalar fields on a regular voxel grid.
 *
 * Why an organ mask gets blurred before isosurfacing: a 0/1 field has every
 * sign change landing at exactly 0.5 between two voxels, so the extracted
 * surface can only ever sit on the voxel mid-planes. The result is a staircase
 * that no amount of post-hoc smoothing fully removes, because the smoothing has
 * no idea which way the true boundary ran. Blurring first replaces the hard
 * mask with a smooth occupancy field, so the 0.5 crossing interpolates to a
 * sub-voxel position and the surface follows the real organ boundary. Three
 * successive box passes converge on a Gaussian (central limit theorem), which
 * is why the blur is built from cheap radius-1 boxes rather than an explicit
 * Gaussian kernel.
 */

/**
 * Blur `field` without modifying it.
 *
 * One "pass" is a full x, then y, then z sweep with a radius-1 box (the
 * [1,1,1]/3 kernel), border-clamped. So `passes` passes have an effective
 * support radius of `passes` voxels, and a caller who needs the blurred field to
 * stay below the 0.5 isovalue at the array border (which is what makes the
 * extracted isosurface closed) has to pad its crop by at least that much. Two
 * voxels of padding, the old fixed figure, silently stops being enough somewhere
 * past ten passes: border clamping reflects a solid region back into the border,
 * so the value there climbs toward 0.5 rather than decaying to zero.
 *
 * Allocation is exactly two scratch volumes regardless of `passes`: the sweeps
 * ping-pong between them and no per-row temporary is ever created.
 */
export function boxBlur3(
  field: Float32Array,
  dims: [number, number, number],
  passes = 2,
): Float32Array {
  const [nx, ny, nz] = dims;
  const n = nx * ny * nz;
  if (field.length < n) {
    throw new Error(`boxBlur3: field has ${field.length} samples, dims need ${n}`);
  }
  if (passes <= 0) return field.slice(0, n);

  const bufA = new Float32Array(n);
  const bufB = new Float32Array(n);

  // The first sweep reads the caller's array; every later sweep alternates
  // between the two scratch buffers.
  let src: Float32Array = field;
  let dst: Float32Array = bufA;
  const advance = (): void => {
    src = dst;
    dst = src === bufA ? bufB : bufA;
  };

  for (let p = 0; p < passes; p++) {
    blurX(src, dst, nx, ny * nz);
    advance();
    blurStrided(src, dst, nx, ny, nz);
    advance();
    blurStrided(src, dst, nx * ny, nz, 1);
    advance();
  }
  // `advance` moved the finished z sweep into `src`.
  return src;
}

/** Contiguous axis: each row of `width` samples is a straight 3-tap scan. */
function blurX(src: Float32Array, dst: Float32Array, width: number, rows: number): void {
  const third = 1 / 3;
  if (width === 1) {
    dst.set(src.subarray(0, rows));
    return;
  }
  const last = width - 1;
  for (let r = 0; r < rows; r++) {
    const base = r * width;
    // Ends replicate the edge sample, so a solid region touching the array
    // border is not eroded by the blur.
    dst[base] = (src[base] * 2 + src[base + 1]) * third;
    for (let i = 1; i < last; i++) {
      dst[base + i] = (src[base + i - 1] + src[base + i] + src[base + i + 1]) * third;
    }
    dst[base + last] = (src[base + last - 1] + src[base + last] * 2) * third;
  }
}

/**
 * Non-contiguous axis, blurred a whole plane at a time.
 *
 * `stride` is the sample distance along the blurred axis and doubles as the
 * width of the contiguous run to process per step: for y that is nx (one row),
 * for z it is nx*ny (one slice). `count` is the number of steps along the axis
 * and `outer` the number of independent blocks above it (nz for y, 1 for z).
 * Sweeping whole contiguous runs keeps this bandwidth bound instead of walking
 * one column at a time and missing the cache on every sample.
 */
function blurStrided(
  src: Float32Array,
  dst: Float32Array,
  stride: number,
  count: number,
  outer: number,
): void {
  const third = 1 / 3;
  const blockSize = stride * count;
  if (count === 1) {
    dst.set(src.subarray(0, blockSize * outer));
    return;
  }
  for (let o = 0; o < outer; o++) {
    const block = o * blockSize;
    for (let c = 0; c < count; c++) {
      const cur = block + c * stride;
      const lo = c === 0 ? cur : cur - stride;
      const hi = c === count - 1 ? cur : cur + stride;
      for (let w = 0; w < stride; w++) {
        dst[cur + w] = (src[lo + w] + src[cur + w] + src[hi + w]) * third;
      }
    }
  }
}
