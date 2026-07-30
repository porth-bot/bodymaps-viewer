/**
 * NIfTI-1 and NIfTI-2 reader. Only the first 3D volume is read: dim[4..7]
 * (time, vectors) reach the header but are never materialised. Field offsets
 * are from nifti1.h / nifti2.h, cross-checked against nibabel output.
 */

import { NiftiDataType } from './types';
import type { FlateError } from 'fflate';
import type { Mat4, NiftiHeader, NiftiImage, TypedNumberArray } from './types';

const NIFTI1_HEADER_SIZE = 348;
const NIFTI2_HEADER_SIZE = 540;

/** Where the voxels start when vox_offset is left at 0 in a single-file image. */
const NIFTI1_DEFAULT_DATA_OFFSET = 352;
const NIFTI2_DEFAULT_DATA_OFFSET = 544;

/** Detected, not assumed: the byte-swap decision below depends on it. */
const PLATFORM_LITTLE_ENDIAN = new Uint8Array(Uint16Array.of(1).buffer)[0] === 1;

interface DataTypeInfo {
  name: string;
  /** Bytes per stored element, which is also the alignment a view needs. */
  bytes: number;
  view(buffer: ArrayBuffer, byteOffset: number, count: number): TypedNumberArray;
}

/**
 * Widened to float64 because nothing downstream, WebGL included, takes bigint.
 * Above 2^53 the low bits go, which only happens in synthetic label data.
 *
 * Not `Float64Array.from(view, Number)`: that allocates a heap bigint per
 * element, 8x slower, a full second of stalled worker on 12.4M voxels.
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

  /** NUL-terminated ASCII. Some writers leave junk past the NUL. */
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

/** nifti1.h "method 2". The quaternion's real part is not stored, so it has to be recovered. */
function affineFromQuaternion(
  quatern: [number, number, number],
  qoffset: [number, number, number],
  pixdim: number[],
): Mat4 {
  let [b, c, d] = quatern;
  let a: number;
  const norm = b * b + c * c + d * d;
  if (1 - norm < 1e-7) {
    // Rounding can push the sum past 1, which would make `a` imaginary. The
    // spec's remedy is 180 degrees about the renormalised (b,c,d) axis.
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

  // pixdim[0] is the handedness flag, not a size. Only -1 means anything, and
  // only to the third column. Writers that leave it 0 mean 1.
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
    // A zero or garbage pixdim makes the affine singular, and every
    // world-to-voxel inverse downstream then throws.
    const step = pixdim[i + 1];
    m[i * 4 + i] = Number.isFinite(step) && step !== 0 ? step : 1;
  }
  m[15] = 1;
  return m;
}

/** Conditioning floor for `isUsableAffine`. Real scans give ~1; a rank-deficient
 *  matrix reaches ~2e-7 once rounded to the float32 srow_* is stored in. */
const MIN_RELATIVE_DETERMINANT = 1e-6;

/**
 * Two malformed headers turn up in the wild: sform_code set with srow_* left
 * empty, and a sane rotation block with a NaN or Infinity origin. Both have to
 * be caught so the qform can take over. A NaN translation slips past the pivot
 * check in `mat4Invert` and reaches the renderer as a black screen.
 *
 * Determinant relative to the column norms, not an absolute floor: a raw one
 * scales as the cube of the voxel size and would depend on xyzt_units.
 */
function isUsableAffine(m: Mat4): boolean {
  // Row 3 is synthesised, never read from the file.
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

  // Spec: slope 0 disables scaling. nibabel writes unset fields as NaN, which
  // would otherwise poison every voxel.
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

  // Whole rows, so both the direction/spacing block (file units, in sform) and
  // the origin (carried separately by qform) reach millimetres together.
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
    // Filled in by parseNifti once the rank has been validated.
    volumeCount: 1,
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
 * Writers spell "unused" as 0 about as often as 1, so an axis past dim[0] may
 * default. Within dim[0] a 0 is corrupt and has to be reported: `dim[n] || 1`
 * turns the file into a plausible one-voxel-thick volume that renders fine.
 */
function spatialDims(header: NiftiHeader): [number, number, number] {
  const declaredRank = header.dim[0];
  // A rank outside 1..7 is corrupt too, and calling it 3 requires all three.
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

/** Only the first volume is read, but 4D files (time course, diffusion series)
 *  report the count so the UI can say the rest were dropped. */
function storedVolumeCount(header: NiftiHeader): number {
  const rank = header.dim[0];
  if (!Number.isInteger(rank) || rank < 4 || rank > 7) return 1;
  let n = 1;
  for (let axis = 4; axis <= rank; axis++) {
    const size = header.dim[axis];
    if (Number.isInteger(size) && size > 1) n *= size;
  }
  return n;
}

export function parseNifti(buffer: ArrayBuffer): NiftiImage {
  const header = parseNiftiHeader(buffer);
  const info = DATA_TYPES[header.datatype];
  if (!info) throw new Error(`Unsupported NIfTI datatype code ${header.datatype}`);

  const [nx, ny, nz] = spatialDims(header);
  const count = nx * ny * nz;
  header.volumeCount = storedVolumeCount(header);

  const defaultOffset = header.version === 1 ? NIFTI1_DEFAULT_DATA_OFFSET : NIFTI2_DEFAULT_DATA_OFFSET;
  const dataOffset = header.voxOffset > 0 ? header.voxOffset : defaultOffset;
  if (!Number.isInteger(dataOffset)) {
    // vox_offset is a float32 in NIfTI-1, so a corrupt file can name a fractional start.
    throw new Error(`Invalid NIfTI vox_offset ${header.voxOffset}: must be a whole byte count`);
  }
  if (dataOffset < defaultOffset) {
    // Confusing sizeof_hdr (348) with the first data byte (352) is a common
    // writer bug, and the past-the-end check below cannot catch it: starting
    // early leaves plenty of bytes, so the volume just renders a few samples off.
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
    // Zero-copy: a 25 MB CT does not need a second 25 MB to be read. The result
    // aliases `buffer`, safe because callers always pass a fresh fetch or gunzip.
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
 * Fallback for browsers without DecompressionStream (Safari before 16.4,
 * Firefox before 113) and for streams the native decoder rejects. Not fflate's
 * one-shot `gunzipSync`: it sizes output from the last four bytes of the whole
 * buffer, so a NUL-padded member decodes to zero bytes and a concatenated
 * stream to its first member only, neither with an error.
 */
async function gunzipFflate(buf: ArrayBuffer): Promise<ArrayBuffer> {
  // Lazy, so browsers with a native decoder never pay for this in the bundle.
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
    // InvalidHeader after at least one whole member means trailing padding, not
    // another member. Anything else, UnexpectedEOF above all, means what we have
    // is incomplete, and a clean prefix surfaces downstream as a missing .img.
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

/** Decompresses a .nii.gz, or returns `buf` untouched when it is not gzipped. */
export async function gunzipIfNeeded(buf: ArrayBuffer): Promise<ArrayBuffer> {
  if (buf.byteLength < 2) return buf;
  const magic = new Uint8Array(buf, 0, 2);
  if (magic[0] !== 0x1f || magic[1] !== 0x8b) return buf;

  if (typeof DecompressionStream !== 'undefined') {
    try {
      return await gunzipNative(buf);
    } catch {
      // DecompressionStream refuses anything after the last member, which some
      // writers pad with NULs to a block boundary.
    }
  }
  return await gunzipFflate(buf);
}
