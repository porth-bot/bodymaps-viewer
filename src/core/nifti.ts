/**
 * NIfTI-1 and NIfTI-2 reader.
 *
 * Scope note: this reads the *first* 3D volume of a file. dim[4..7] (time,
 * vectors) are parsed into the header but not materialised, because the whole
 * app downstream is built on a single scalar grid (see `Volume` in types.ts).
 *
 * The two header layouts share almost no field offsets, so they are read by
 * separate functions into one intermediate shape and then finished in common
 * code. Offsets below come from nifti1.h / nifti2.h and were cross-checked
 * against files written by nibabel.
 */

import { NiftiDataType } from './types';
import type { FlateError } from 'fflate';
import type { Mat4, NiftiHeader, NiftiImage, TypedNumberArray } from './types';

const NIFTI1_HEADER_SIZE = 348;
const NIFTI2_HEADER_SIZE = 540;

/** Where the voxels start when vox_offset is left at 0 in a single-file image. */
const NIFTI1_DEFAULT_DATA_OFFSET = 352;
const NIFTI2_DEFAULT_DATA_OFFSET = 544;

/**
 * Detected rather than assumed: every consumer platform today is
 * little-endian, but the byte-swap decision below is only correct if this is
 * a fact about the machine instead of a hardcoded guess.
 */
const PLATFORM_LITTLE_ENDIAN = new Uint8Array(Uint16Array.of(1).buffer)[0] === 1;

interface DataTypeInfo {
  name: string;
  /** Bytes per stored element, which is also the alignment a view needs. */
  bytes: number;
  view(buffer: ArrayBuffer, byteOffset: number, count: number): TypedNumberArray;
}

/**
 * 64-bit integers are widened to float64 because the rest of the app (and
 * WebGL) has no bigint path. Magnitudes above 2^53 lose their low bits; that
 * only occurs in synthetic label data, never in imaging.
 *
 * The loop is spelled out instead of `Float64Array.from(view, Number)` because
 * that form goes through the generic iterate-and-convert path and allocates a
 * second heap bigint per element: about 8x slower, or a full second of stalled
 * worker on a 12.4M voxel volume.
 */
function widenBigInts(source: BigInt64Array | BigUint64Array): Float64Array {
  const out = new Float64Array(source.length);
  for (let i = 0; i < source.length; i++) out[i] = Number(source[i]);
  return out;
}

const DATA_TYPES: Partial<Record<number, DataTypeInfo>> = {
  [NiftiDataType.UINT8]: { name: 'uint8', bytes: 1, view: (b, o, n) => new Uint8Array(b, o, n) },
  [NiftiDataType.INT16]: { name: 'int16', bytes: 2, view: (b, o, n) => new Int16Array(b, o, n) },
  [NiftiDataType.INT32]: { name: 'int32', bytes: 4, view: (b, o, n) => new Int32Array(b, o, n) },
  [NiftiDataType.FLOAT32]: { name: 'float32', bytes: 4, view: (b, o, n) => new Float32Array(b, o, n) },
  [NiftiDataType.FLOAT64]: { name: 'float64', bytes: 8, view: (b, o, n) => new Float64Array(b, o, n) },
  [NiftiDataType.INT8]: { name: 'int8', bytes: 1, view: (b, o, n) => new Int8Array(b, o, n) },
  [NiftiDataType.UINT16]: { name: 'uint16', bytes: 2, view: (b, o, n) => new Uint16Array(b, o, n) },
  [NiftiDataType.UINT32]: { name: 'uint32', bytes: 4, view: (b, o, n) => new Uint32Array(b, o, n) },
  [NiftiDataType.INT64]: {
    name: 'int64',
    bytes: 8,
    view: (b, o, n) => widenBigInts(new BigInt64Array(b, o, n)),
  },
  [NiftiDataType.UINT64]: {
    name: 'uint64',
    bytes: 8,
    view: (b, o, n) => widenBigInts(new BigUint64Array(b, o, n)),
  },
};

/** Fields both header versions carry, normalised to JS numbers. */
interface RawFields {
  dim: number[];
  pixdim: number[];
  datatype: number;
  bitpix: number;
  voxOffset: number;
  sclSlope: number;
  sclInter: number;
  calMin: number;
  calMax: number;
  qformCode: number;
  sformCode: number;
  xyztUnits: number;
  quatern: [number, number, number];
  qoffset: [number, number, number];
  /** srow_x, srow_y, srow_z concatenated (12 values, row-major). */
  srow: number[];
  description: string;
  intentName: string;
}

class FieldReader {
  private readonly view: DataView;
  private readonly bytes: Uint8Array;
  private readonly le: boolean;

  constructor(buffer: ArrayBuffer, littleEndian: boolean) {
    this.view = new DataView(buffer);
    this.bytes = new Uint8Array(buffer);
    this.le = littleEndian;
  }

  u8(offset: number): number {
    return this.bytes[offset];
  }

  i16(offset: number): number {
    return this.view.getInt16(offset, this.le);
  }

  i32(offset: number): number {
    return this.view.getInt32(offset, this.le);
  }

  i64(offset: number): number {
    return Number(this.view.getBigInt64(offset, this.le));
  }

  f32(offset: number): number {
    return this.view.getFloat32(offset, this.le);
  }

  f64(offset: number): number {
    return this.view.getFloat64(offset, this.le);
  }

  array(offset: number, count: number, stride: number, read: (o: number) => number): number[] {
    const out = new Array<number>(count);
    for (let i = 0; i < count; i++) out[i] = read(offset + i * stride);
    return out;
  }

  /** NUL-terminated ASCII, padded with junk in some writers past the NUL. */
  ascii(offset: number, maxLength: number): string {
    const limit = Math.min(offset + maxLength, this.bytes.length);
    let end = offset;
    while (end < limit && this.bytes[end] !== 0) end++;
    let out = '';
    for (let i = offset; i < end; i++) out += String.fromCharCode(this.bytes[i]);
    return out;
  }
}

function detectVersion(buffer: ArrayBuffer): { version: 1 | 2; littleEndian: boolean } {
  if (buffer.byteLength < 4) {
    throw new Error(`Not a NIfTI file: buffer is ${buffer.byteLength} bytes, need at least 348`);
  }
  const view = new DataView(buffer);
  for (const littleEndian of [true, false]) {
    const sizeofHdr = view.getInt32(0, littleEndian);
    if (sizeofHdr === NIFTI1_HEADER_SIZE) return { version: 1, littleEndian };
    if (sizeofHdr === NIFTI2_HEADER_SIZE) return { version: 2, littleEndian };
  }
  throw new Error(
    `Not a NIfTI file: sizeof_hdr reads ${view.getInt32(0, true)} little-endian / ` +
      `${view.getInt32(0, false)} big-endian, expected 348 (NIfTI-1) or 540 (NIfTI-2)`,
  );
}

function readNifti1(r: FieldReader): RawFields {
  const magic = r.ascii(344, 4);
  if (magic !== 'n+1' && magic !== 'ni1') {
    throw new Error(`Bad NIfTI-1 magic "${magic}", expected "n+1" or "ni1"`);
  }
  return {
    dim: r.array(40, 8, 2, (o) => r.i16(o)),
    datatype: r.i16(70),
    bitpix: r.i16(72),
    pixdim: r.array(76, 8, 4, (o) => r.f32(o)),
    voxOffset: r.f32(108),
    sclSlope: r.f32(112),
    sclInter: r.f32(116),
    xyztUnits: r.u8(123),
    calMax: r.f32(124),
    calMin: r.f32(128),
    description: r.ascii(148, 80),
    qformCode: r.i16(252),
    sformCode: r.i16(254),
    quatern: [r.f32(256), r.f32(260), r.f32(264)],
    qoffset: [r.f32(268), r.f32(272), r.f32(276)],
    srow: r.array(280, 12, 4, (o) => r.f32(o)),
    intentName: r.ascii(328, 16),
  };
}

function readNifti2(r: FieldReader): RawFields {
  const magic = r.ascii(4, 4);
  if (magic !== 'n+2' && magic !== 'ni2') {
    throw new Error(`Bad NIfTI-2 magic "${magic}", expected "n+2" or "ni2"`);
  }
  return {
    datatype: r.i16(12),
    bitpix: r.i16(14),
    dim: r.array(16, 8, 8, (o) => r.i64(o)),
    pixdim: r.array(104, 8, 8, (o) => r.f64(o)),
    voxOffset: r.i64(168),
    sclSlope: r.f64(176),
    sclInter: r.f64(184),
    calMax: r.f64(192),
    calMin: r.f64(200),
    description: r.ascii(240, 80),
    qformCode: r.i32(344),
    sformCode: r.i32(348),
    quatern: [r.f64(352), r.f64(360), r.f64(368)],
    qoffset: [r.f64(376), r.f64(384), r.f64(392)],
    srow: r.array(400, 12, 8, (o) => r.f64(o)),
    xyztUnits: r.i32(500),
    intentName: r.ascii(508, 16),
  };
}

function affineFromSrow(srow: number[]): Mat4 {
  const m = new Float64Array(16);
  for (let i = 0; i < 12; i++) m[i] = srow[i];
  m[15] = 1;
  return m;
}

/**
 * nifti1.h "method 2": rebuild the rotation from the unit quaternion whose
 * real part was dropped on write, then scale its columns by the voxel sizes.
 */
function affineFromQuaternion(
  quatern: [number, number, number],
  qoffset: [number, number, number],
  pixdim: number[],
): Mat4 {
  let [b, c, d] = quatern;
  let a: number;
  const norm = b * b + c * c + d * d;
  if (1 - norm < 1e-7) {
    // Storage rounding can push the sum just past 1, which would make `a`
    // imaginary. The spec's remedy is a 180 degree rotation about the
    // renormalised (b,c,d) axis.
    const scale = 1 / Math.sqrt(norm);
    b *= scale;
    c *= scale;
    d *= scale;
    a = 0;
  } else {
    a = Math.sqrt(1 - norm);
  }

  const r = [
    a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c),
    2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b),
    2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c,
  ];

  // pixdim[0] carries the handedness flag, not a size. Only -1 means anything
  // and only to the third column; writers that leave it 0 mean 1.
  const qfac = pixdim[0] < 0 ? -1 : 1;
  const scale = [pixdim[1], pixdim[2], pixdim[3] * qfac];

  const m = new Float64Array(16);
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) m[row * 4 + col] = r[row * 3 + col] * scale[col];
    m[row * 4 + 3] = qoffset[row];
  }
  m[15] = 1;
  return m;
}

function affineFromPixdim(pixdim: number[]): Mat4 {
  const m = new Float64Array(16);
  for (let i = 0; i < 3; i++) {
    // A zero or garbage pixdim would make the affine singular and every
    // world-to-voxel inverse downstream throw, so fall back to unit spacing.
    const step = pixdim[i + 1];
    m[i * 4 + i] = Number.isFinite(step) && step !== 0 ? step : 1;
  }
  m[15] = 1;
  return m;
}

/**
 * Relative floor on the affine's conditioning, see `isUsableAffine`. Well below
 * anything a real scan produces (columns near orthogonal give ~1) and well
 * above the ~2e-7 a genuinely rank-deficient matrix reaches once its entries
 * have been rounded to the float32 that NIfTI-1 stores srow_* in.
 */
const MIN_RELATIVE_DETERMINANT = 1e-6;

/**
 * Whether an affine is safe to hand downstream.
 *
 * Two malformed-header shapes occur in the wild. Some converters set
 * sform_code without filling in srow_*, leaving a degenerate matrix; others
 * write a partially initialised header whose rotation block is fine but whose
 * origin comes out NaN or Infinity. Both have to be caught so the qform can
 * take over, because neither is stopped further downstream: a NaN translation
 * defeats the pivot check in `mat4Invert` and reaches the renderer as world
 * coordinates that are silently NaN, i.e. a black screen with no diagnostic.
 *
 * Conditioning is measured as the determinant relative to the product of the
 * column norms rather than against an absolute floor. A raw determinant scales
 * as the cube of the voxel size, so a fixed threshold would reject a perfectly
 * conditioned fine-voxel grid, and it would make the verdict depend on
 * xyzt_units, which is only applied after this runs.
 */
function isUsableAffine(m: Mat4): boolean {
  // Row 3 is synthesised, never read from the file, so only the 12 meaningful
  // entries (rotation/scale plus the origin column) are worth checking.
  for (let i = 0; i < 12; i++) {
    if (!Number.isFinite(m[i])) return false;
  }
  const columnScale =
    Math.hypot(m[0], m[4], m[8]) * Math.hypot(m[1], m[5], m[9]) * Math.hypot(m[2], m[6], m[10]);
  if (!(columnScale > 0)) return false;
  const det =
    m[0] * (m[5] * m[10] - m[6] * m[9]) -
    m[1] * (m[4] * m[10] - m[6] * m[8]) +
    m[2] * (m[4] * m[9] - m[5] * m[8]);
  return Math.abs(det) / columnScale > MIN_RELATIVE_DETERMINANT;
}

/** Millimetres per file unit, from the low 3 bits of xyzt_units. */
function spatialScale(units: number): number {
  switch (units) {
    case 1:
      return 1000; // metres
    case 3:
      return 0.001; // micrometres
    default:
      return 1; // millimetres, or unspecified (0), which everyone means as mm
  }
}

export function parseNiftiHeader(buffer: ArrayBuffer): NiftiHeader {
  const { version, littleEndian } = detectVersion(buffer);
  const headerSize = version === 1 ? NIFTI1_HEADER_SIZE : NIFTI2_HEADER_SIZE;
  if (buffer.byteLength < headerSize) {
    throw new Error(
      `Truncated NIfTI-${version} header: ${buffer.byteLength} bytes available, ${headerSize} required`,
    );
  }

  const reader = new FieldReader(buffer, littleEndian);
  const raw = version === 1 ? readNifti1(reader) : readNifti2(reader);

  const info = DATA_TYPES[raw.datatype];
  if (!info) {
    throw new Error(
      `Unsupported NIfTI datatype code ${raw.datatype}. Supported: ` +
        `${Object.keys(DATA_TYPES).join(', ')} (uint8, int16, int32, float32, float64, ` +
        `int8, uint16, uint32, int64, uint64)`,
    );
  }

  // Spec: slope 0 disables scaling. Unset fields are also commonly written as
  // NaN (nibabel does this), which would otherwise poison every voxel.
  let sclSlope = raw.sclSlope;
  let sclInter = raw.sclInter;
  if (!Number.isFinite(sclSlope) || sclSlope === 0) {
    sclSlope = 1;
    sclInter = 0;
  }
  if (!Number.isFinite(sclInter)) sclInter = 0;

  const sform = raw.sformCode > 0 ? affineFromSrow(raw.srow) : null;
  const qform = raw.qformCode > 0 ? affineFromQuaternion(raw.quatern, raw.qoffset, raw.pixdim) : null;

  let affine: Mat4;
  let affineSource: NiftiHeader['affineSource'];
  if (sform && isUsableAffine(sform)) {
    affine = sform;
    affineSource = 'sform';
  } else if (qform && isUsableAffine(qform)) {
    affine = qform;
    affineSource = 'qform';
  } else {
    affine = affineFromPixdim(raw.pixdim);
    affineSource = 'pixdim';
  }

  // Convert the world frame to millimetres. Scaling whole rows covers both the
  // direction/spacing block (which sform carries in file units) and the
  // origin (which qform carries separately), so the affine stays consistent.
  const spatialUnits = raw.xyztUnits & 0x07;
  const scale = spatialScale(spatialUnits);
  const pixdim = raw.pixdim.slice();
  if (scale !== 1) {
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 4; col++) affine[row * 4 + col] *= scale;
    }
    // pixdim[0] is qfac and pixdim[4..7] are non-spatial, so only 1..3 move.
    for (let i = 1; i <= 3; i++) pixdim[i] *= scale;
  }

  return {
    version,
    littleEndian,
    dim: raw.dim,
    datatype: raw.datatype as NiftiDataType,
    bitpix: raw.bitpix,
    pixdim,
    voxOffset: raw.voxOffset,
    sclSlope,
    sclInter,
    calMin: Number.isFinite(raw.calMin) ? raw.calMin : 0,
    calMax: Number.isFinite(raw.calMax) ? raw.calMax : 0,
    qformCode: raw.qformCode,
    sformCode: raw.sformCode,
    spatialUnits,
    description: raw.description,
    intentName: raw.intentName,
    affine,
    affineSource,
  };
}

function swapBytesInPlace(bytes: Uint8Array, width: number): void {
  if (width < 2) return;
  const half = width >> 1;
  for (let base = 0; base + width <= bytes.length; base += width) {
    for (let i = 0; i < half; i++) {
      const lo = base + i;
      const hi = base + width - 1 - i;
      const tmp = bytes[lo];
      bytes[lo] = bytes[hi];
      bytes[hi] = tmp;
    }
  }
}

const AXIS_NAMES = ['i', 'j', 'k'] as const;

/**
 * The three spatial sizes, validated.
 *
 * dim[0] is the dimensionality, and a 2D image leaves dim[3] unused: writers
 * spell "unused" as 0 about as often as 1, so an axis past dim[0] may default.
 * Within dim[0] a size of 0 is a corrupt header and has to be reported.
 * Defaulting it to 1 (which is what a bare `dim[n] || 1` does) turns the file
 * into a plausible one-voxel-thick volume that renders without complaint.
 */
function spatialDims(header: NiftiHeader): [number, number, number] {
  const declaredRank = header.dim[0];
  // A rank outside the spec's 1..7 is itself corrupt; treating it as 3 means
  // all three sizes are then required rather than quietly defaulted.
  const rankIsSane = Number.isInteger(declaredRank) && declaredRank >= 1 && declaredRank <= 7;
  const rank = rankIsSane ? declaredRank : 3;

  const sizes: [number, number, number] = [1, 1, 1];
  for (let axis = 0; axis < 3; axis++) {
    const size = header.dim[axis + 1];
    if (axis + 1 > rank && size === 0) continue;
    if (!Number.isInteger(size) || size < 1) {
      throw new Error(`Invalid NIfTI dimension along ${AXIS_NAMES[axis]}: ${size}`);
    }
    sizes[axis] = size;
  }
  return sizes;
}

export function parseNifti(buffer: ArrayBuffer): NiftiImage {
  const header = parseNiftiHeader(buffer);
  const info = DATA_TYPES[header.datatype];
  if (!info) throw new Error(`Unsupported NIfTI datatype code ${header.datatype}`);

  const [nx, ny, nz] = spatialDims(header);
  const count = nx * ny * nz;

  const defaultOffset = header.version === 1 ? NIFTI1_DEFAULT_DATA_OFFSET : NIFTI2_DEFAULT_DATA_OFFSET;
  const dataOffset = header.voxOffset > 0 ? header.voxOffset : defaultOffset;
  if (!Number.isInteger(dataOffset)) {
    // vox_offset is a float32 in NIfTI-1, so a corrupt file can name a
    // fractional start. Slicing there would silently truncate.
    throw new Error(`Invalid NIfTI vox_offset ${header.voxOffset}: must be a whole byte count`);
  }
  if (dataOffset < defaultOffset) {
    // Writers that confuse sizeof_hdr (348) with the first data byte (352) are
    // common enough to name. The past-the-end check below cannot catch this:
    // starting early always leaves more than enough bytes, so the volume just
    // comes out shifted by a couple of samples and still renders.
    throw new Error(
      `Invalid NIfTI vox_offset ${header.voxOffset}: voxels cannot start before byte ` +
        `${defaultOffset} in a single-file NIfTI-${header.version}, which is where the header ` +
        `block ends. Lower values name bytes inside the header itself.`,
    );
  }
  const byteLength = count * info.bytes;
  if (dataOffset + byteLength > buffer.byteLength) {
    throw new Error(
      `NIfTI voxel data runs past the end of the buffer: need ${byteLength} bytes at offset ` +
        `${dataOffset} but only ${buffer.byteLength} are present. A detached .hdr/.img pair ` +
        `must be loaded from the .img file.`,
    );
  }

  const needsSwap = header.littleEndian !== PLATFORM_LITTLE_ENDIAN;
  const aligned = dataOffset % info.bytes === 0;
  if (!needsSwap && aligned) {
    // Zero-copy: a 25 MB CT does not need a second 25 MB just to be read. The
    // returned array aliases `buffer`, which is safe because every caller
    // hands us a buffer freshly produced by fetch or gunzip.
    return { header, data: info.view(buffer, dataOffset, count) };
  }
  const copy = buffer.slice(dataOffset, dataOffset + byteLength);
  if (needsSwap) swapBytesInPlace(new Uint8Array(copy), info.bytes);
  return { header, data: info.view(copy, 0, count) };
}

async function gunzipNative(buffer: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).arrayBuffer();
}

/**
 * Decompress with fflate, for browsers without DecompressionStream (Safari
 * before 16.4, Firefox before 113) and for the streams the native decoder
 * rejects.
 *
 * fflate's one-shot `gunzipSync` is not usable here: it sizes its output from
 * the last four bytes of the whole buffer, so a member padded with NULs comes
 * back as zero bytes and a concatenated stream comes back as its first member
 * only, both without an error. The streaming decoder walks member by member
 * instead, and stopping at InvalidHeader keeps every complete member that
 * preceded the trailing bytes.
 */
async function gunzipFflate(buf: ArrayBuffer): Promise<ArrayBuffer> {
  // Imported lazily so browsers with a native gzip decoder never pay for the
  // fallback in the initial bundle.
  const { FlateErrorCode, Gunzip } = await import('fflate');

  const chunks: Uint8Array[] = [];
  let total = 0;
  const stream = new Gunzip((chunk) => {
    if (chunk.length === 0) return;
    chunks.push(chunk);
    total += chunk.length;
  });

  try {
    stream.push(new Uint8Array(buf), true);
  } catch (error) {
    // InvalidHeader once at least one whole member has been inflated means the
    // bytes after it are padding rather than another member. Every other
    // failure (above all UnexpectedEOF) means what we have is incomplete, and
    // returning a clean prefix would be reported downstream as a missing .img.
    if (total === 0 || (error as FlateError).code !== FlateErrorCode.InvalidHeader) throw error;
  }
  if (total === 0) {
    throw new Error('Gzip stream decompressed to zero bytes: the file is empty or truncated');
  }

  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out.buffer as ArrayBuffer;
}

/**
 * Transparently decompress a .nii.gz. Returns `buf` untouched when it is not
 * gzipped, so callers can stay ignorant of which form they fetched.
 */
export async function gunzipIfNeeded(buf: ArrayBuffer): Promise<ArrayBuffer> {
  if (buf.byteLength < 2) return buf;
  const magic = new Uint8Array(buf, 0, 2);
  if (magic[0] !== 0x1f || magic[1] !== 0x8b) return buf;

  if (typeof DecompressionStream !== 'undefined') {
    try {
      return await gunzipNative(buf);
    } catch {
      // Fall through: DecompressionStream refuses a stream with anything after
      // the last member, which some writers pad with NULs to a block boundary.
    }
  }
  return await gunzipFflate(buf);
}
