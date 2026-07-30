/**
 * Separable box blur for scalar fields on a regular voxel grid.
 *
 * The mask is blurred before it is thresholded because a 0/1 field puts every
 * sign change exactly halfway between two voxels, so the isosurface can only
 * land on the voxel mid-planes. No later smoothing undoes that staircase, since
 * by then nothing knows which way the true boundary ran; blurred first, the 0.5
 * crossing interpolates to a sub-voxel position.
 */

/**
 * Blur `field` without modifying it. A pass is an x, then y, then z sweep with a
 * radius-1 box, so `passes` passes reach `passes` voxels. Repeated boxes
 * converge on a Gaussian, which is why there is no explicit Gaussian kernel.
 *
 * Border-clamped, so a caller that needs the blurred field below 0.5 at the
 * array border (which is what makes the extracted isosurface closed) has to pad
 * its crop by at least that much. Two voxels, the old fixed figure, silently
 * stops being enough past ten passes: clamping reflects a solid region back into
 * the border, so the value there climbs toward 0.5 rather than decaying to zero.
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
 * Non-contiguous axis, blurred a whole plane at a time. `stride` is the sample
 * distance along the blurred axis and doubles as the width of the contiguous run
 * to process per step: nx (a row) for y, nx*ny (a slice) for z. `count` is the
 * number of steps along the axis, `outer` the independent blocks above it (nz
 * for y, 1 for z). Sweeping whole runs rather than one column at a time is the
 * difference between bandwidth bound and a cache miss per sample.
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
