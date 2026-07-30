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
  // 64-bit integers are widened to float64 because the rest of the app (and
  // WebGL) has no bigint path. Magnitudes above 2^53 lose their low bits;
  // that only occurs in synthetic label data, never in imaging.
  [NiftiDataType.INT64]: {
    name: 'int64',
    bytes: 8,
    view: (b, o, n) => Float64Array.from(new BigInt64Array(b, o, n), Number),
  },
  [NiftiDataType.UINT64]: {
    name: 'uint64',
    bytes: 8,
    view: (b, o, n) => Float64Array.from(new BigUint64Array(b, o, n), Number),
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
 * Whether an affine can actually be inverted.
 *
 * Some converters set sform_code without filling in srow_*, leaving a
 * degenerate matrix. Detecting it here lets the qform take over, instead of
 * the failure surfacing much later as a singular-matrix throw or a black
 * screen in the renderer.
 */
function isInvertible(m: Mat4): boolean {
  const det =
    m[0] * (m[5] * m[10] - m[6] * m[9]) -
    m[1] * (m[4] * m[10] - m[6] * m[8]) +
    m[2] * (m[4] * m[9] - m[5] * m[8]);
  return Number.isFinite(det) && Math.abs(det) > 1e-12;
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
  if (sform && isInvertible(sform)) {
    affine = sform;
    affineSource = 'sform';
  } else if (qform && isInvertible(qform)) {
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

export function parseNifti(buffer: ArrayBuffer): NiftiImage {
  const header = parseNiftiHeader(buffer);
  const info = DATA_TYPES[header.datatype];
  if (!info) throw new Error(`Unsupported NIfTI datatype code ${header.datatype}`);

  const nx = header.dim[1] || 1;
  const ny = header.dim[2] || 1;
  const nz = header.dim[3] || 1;
  for (const [axis, size] of [['i', nx], ['j', ny], ['k', nz]] as const) {
    if (!Number.isInteger(size) || size < 1) {
      throw new Error(`Invalid NIfTI dimension along ${axis}: ${size}`);
    }
  }
  const count = nx * ny * nz;

  const defaultOffset = header.version === 1 ? NIFTI1_DEFAULT_DATA_OFFSET : NIFTI2_DEFAULT_DATA_OFFSET;
  const dataOffset = header.voxOffset > 0 ? header.voxOffset : defaultOffset;
  if (!Number.isInteger(dataOffset)) {
    // vox_offset is a float32 in NIfTI-1, so a corrupt file can name a
    // fractional start. Slicing there would silently truncate.
    throw new Error(`Invalid NIfTI vox_offset ${header.voxOffset}: must be a whole byte count`);
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
      // Fall through: DecompressionStream rejects some streams that fflate
      // tolerates (trailing bytes, concatenated members).
    }
  }
  // Imported lazily so browsers with a native gzip decoder never pay for the
  // fallback in the initial bundle.
  const { gunzipSync } = await import('fflate');
  const out = gunzipSync(new Uint8Array(buf));
  if (out.byteOffset === 0 && out.byteLength === out.buffer.byteLength) {
    return out.buffer as ArrayBuffer;
  }
  return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength) as ArrayBuffer;
}
