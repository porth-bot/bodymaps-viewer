import { describe, expect, it } from 'vitest';
import { gzipSync } from 'fflate';

import { gunzipIfNeeded, parseNifti, parseNiftiHeader } from '../src/core/nifti';
import { NiftiDataType } from '../src/core/types';

// `node:fs` has no type declarations in this project (no @types/node), so the
// specifier is held in a variable to keep it out of the type graph. The tests
// still need real bytes off disk for the integration cases.
interface NodeFs {
  readFileSync(path: URL): Uint8Array;
}
const FS_MODULE = 'node:fs';
const fs = (await import(FS_MODULE)) as NodeFs;

const CASE_DIR = new URL('../public/data/BDMAP_00000338/', import.meta.url);

function readCaseFile(relative: string): ArrayBuffer {
  const bytes = fs.readFileSync(new URL(relative, CASE_DIR));
  return toArrayBuffer(bytes);
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

// --- NIfTI-1 synthesiser ---------------------------------------------------

interface ElementWriter {
  bytes: number;
  write(view: DataView, offset: number, value: number, littleEndian: boolean): void;
}

const ELEMENT_WRITERS: Partial<Record<number, ElementWriter>> = {
  [NiftiDataType.UINT8]: { bytes: 1, write: (v, o, x) => v.setUint8(o, x) },
  [NiftiDataType.INT8]: { bytes: 1, write: (v, o, x) => v.setInt8(o, x) },
  [NiftiDataType.INT16]: { bytes: 2, write: (v, o, x, le) => v.setInt16(o, x, le) },
  [NiftiDataType.UINT16]: { bytes: 2, write: (v, o, x, le) => v.setUint16(o, x, le) },
  [NiftiDataType.INT32]: { bytes: 4, write: (v, o, x, le) => v.setInt32(o, x, le) },
  [NiftiDataType.UINT32]: { bytes: 4, write: (v, o, x, le) => v.setUint32(o, x, le) },
  [NiftiDataType.FLOAT32]: { bytes: 4, write: (v, o, x, le) => v.setFloat32(o, x, le) },
  [NiftiDataType.FLOAT64]: { bytes: 8, write: (v, o, x, le) => v.setFloat64(o, x, le) },
  [NiftiDataType.INT64]: { bytes: 8, write: (v, o, x, le) => v.setBigInt64(o, BigInt(x), le) },
  [NiftiDataType.UINT64]: { bytes: 8, write: (v, o, x, le) => v.setBigUint64(o, BigInt(x), le) },
};

interface Nifti1Spec {
  dims: [number, number, number];
  datatype: NiftiDataType;
  values: readonly number[];
  littleEndian?: boolean;
  pixdim?: [number, number, number];
  /** pixdim[0]; negative means the qform's third column is flipped. */
  qfac?: number;
  sclSlope?: number;
  sclInter?: number;
  sformCode?: number;
  /** srow_x, srow_y, srow_z concatenated. */
  srow?: readonly number[];
  qformCode?: number;
  quatern?: readonly [number, number, number];
  qoffset?: readonly [number, number, number];
  xyztUnits?: number;
  descrip?: string;
  intentName?: string;
  /** Extra bytes between the header and the voxels, to exercise vox_offset. */
  padding?: number;
}

function writeAscii(bytes: Uint8Array, offset: number, text: string, maxLength: number): void {
  for (let i = 0; i < Math.min(text.length, maxLength - 1); i++) {
    bytes[offset + i] = text.charCodeAt(i);
  }
}

function synthNifti1(spec: Nifti1Spec): ArrayBuffer {
  const le = spec.littleEndian ?? true;
  const writer = ELEMENT_WRITERS[spec.datatype];
  if (!writer) throw new Error(`test synthesiser has no writer for datatype ${spec.datatype}`);

  const dataOffset = 352 + (spec.padding ?? 0);
  const buffer = new ArrayBuffer(dataOffset + spec.values.length * writer.bytes);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  view.setInt32(0, 348, le);
  bytes[38] = 'r'.charCodeAt(0); // `regular`, cosmetic but present in real files
  view.setInt16(40, 3, le);
  for (let i = 0; i < 3; i++) view.setInt16(42 + i * 2, spec.dims[i], le);
  for (let i = 4; i < 8; i++) view.setInt16(40 + i * 2, 1, le);

  view.setInt16(70, spec.datatype, le);
  view.setInt16(72, writer.bytes * 8, le);

  const pixdim = spec.pixdim ?? [1, 1, 1];
  view.setFloat32(76, spec.qfac ?? 1, le);
  for (let i = 0; i < 3; i++) view.setFloat32(80 + i * 4, pixdim[i], le);
  for (let i = 4; i < 8; i++) view.setFloat32(76 + i * 4, 1, le);

  view.setFloat32(108, dataOffset, le);
  view.setFloat32(112, spec.sclSlope ?? 1, le);
  view.setFloat32(116, spec.sclInter ?? 0, le);
  bytes[123] = spec.xyztUnits ?? 2;
  writeAscii(bytes, 148, spec.descrip ?? '', 80);

  view.setInt16(252, spec.qformCode ?? 0, le);
  view.setInt16(254, spec.sformCode ?? 0, le);
  const quatern = spec.quatern ?? [0, 0, 0];
  const qoffset = spec.qoffset ?? [0, 0, 0];
  for (let i = 0; i < 3; i++) view.setFloat32(256 + i * 4, quatern[i], le);
  for (let i = 0; i < 3; i++) view.setFloat32(268 + i * 4, qoffset[i], le);
  const srow = spec.srow ?? new Array<number>(12).fill(0);
  for (let i = 0; i < 12; i++) view.setFloat32(280 + i * 4, srow[i], le);

  writeAscii(bytes, 328, spec.intentName ?? '', 16);
  writeAscii(bytes, 344, 'n+1', 4);

  for (let i = 0; i < spec.values.length; i++) {
    writer.write(view, dataOffset + i * writer.bytes, spec.values[i], le);
  }
  return buffer;
}

function ramp(count: number, start = 0): number[] {
  return Array.from({ length: count }, (_, i) => i + start);
}

/**
 * NIfTI-2 files written by nibabel 5.x, base64'd, so the NIfTI-2 offsets are
 * checked against an independent writer instead of against this file's own
 * assumptions. Both hold int16 0..23 on a 2x3x4 grid, sform diag(2,3,4) with
 * origin (-10,-20,-30), scl_slope 2, scl_inter 5, xyzt_units 10 (mm + sec).
 * Regenerate with nibabel: Nifti2Image(np.arange(24, dtype=np.int16)
 * .reshape((2,3,4), order='F'), affine).to_bytes().
 */
const NIFTI2_LE_BASE64 =
  'HAIAAG4rMgANChoKBAAQAAMAAAAAAAAAAgAAAAAAAAADAAAAAAAAAAQAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADwPwAAAAAAAABAAAAAAAAACEAAAAAAAAAQQAAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/IAIAAAAAAAAAAAAAAAAAQAAAAAAAABRAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAbmlmdGkyIGZpeHR1cmUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACTAAAAAAAAANMAAAAAAAAA+wAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACTAAAAAAAAAAAAAAAAAAAAIQAAAAAAAAAAAAAAAAAAANMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEEAAAAAAAAA+wAAAAAAKAAAAAAAAAHByb2JlAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQACAAMABAAFAAYABwAIAAkACgALAAwADQAOAA8AEAARABIAEwAUABUAFgAXAA==';
const NIFTI2_BE_BASE64 =
  'AAACHG4rMgANChoKAAQAEAAAAAAAAAADAAAAAAAAAAIAAAAAAAAAAwAAAAAAAAAEAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/8AAAAAAAAEAAAAAAAAAAQAgAAAAAAABAEAAAAAAAAD/wAAAAAAAAP/AAAAAAAAA/8AAAAAAAAD/wAAAAAAAAAAAAAAAAAiBAAAAAAAAAAEAUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAbmlmdGkyIGZpeHR1cmUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAkAAAAAAAAwDQAAAAAAADAPgAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAkAAAAAAAAAAAAAAAAAABACAAAAAAAAAAAAAAAAAAAwDQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQBAAAAAAAADAPgAAAAAAAAAAAAAAAAAKAAAAAHByb2JlAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAgADAAQABQAGAAcACAAJAAoACwAMAA0ADgAPABAAEQASABMAFAAVABYAFw==';

function decodeBase64(text: string): ArrayBuffer {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

describe('parseNifti datatypes', () => {
  const cases: Array<{ datatype: NiftiDataType; ctor: unknown; values: number[] }> = [
    { datatype: NiftiDataType.UINT8, ctor: Uint8Array, values: [0, 1, 200, 255, 7, 9] },
    { datatype: NiftiDataType.INT8, ctor: Int8Array, values: [0, -1, -128, 127, 7, 9] },
    { datatype: NiftiDataType.INT16, ctor: Int16Array, values: [0, -1000, 32767, -32768, 5, 6] },
    { datatype: NiftiDataType.UINT16, ctor: Uint16Array, values: [0, 65535, 1234, 7, 8, 9] },
    { datatype: NiftiDataType.INT32, ctor: Int32Array, values: [0, -2147483648, 2147483647, 3, 4, 5] },
    { datatype: NiftiDataType.UINT32, ctor: Uint32Array, values: [0, 4294967295, 12345, 3, 4, 5] },
    { datatype: NiftiDataType.FLOAT32, ctor: Float32Array, values: [0, -0.5, 1.25, 1e10, -3.75, 2] },
    { datatype: NiftiDataType.FLOAT64, ctor: Float64Array, values: [0, -0.1, 1e300, 1 / 3, -7, 2] },
  ];

  for (const { datatype, ctor, values } of cases) {
    it(`round-trips datatype ${datatype}`, () => {
      const buffer = synthNifti1({
        dims: [3, 2, 1],
        datatype,
        values,
        pixdim: [2, 3, 4],
        sformCode: 1,
        srow: [2, 0, 0, -10, 0, 3, 0, -20, 0, 0, 4, -30],
        descrip: 'synthetic',
      });
      const image = parseNifti(buffer);

      expect(image.data).toBeInstanceOf(ctor as never);
      expect(Array.from(image.data)).toEqual(values);
      expect(image.header.datatype).toBe(datatype);
      expect(image.header.dim.slice(0, 4)).toEqual([3, 3, 2, 1]);
      expect(image.header.pixdim.slice(1, 4)).toEqual([2, 3, 4]);
      expect(image.header.version).toBe(1);
      expect(image.header.description).toBe('synthetic');
      expect(image.header.affineSource).toBe('sform');
    });
  }

  it('widens int64 and uint64 to float64', () => {
    for (const datatype of [NiftiDataType.INT64, NiftiDataType.UINT64] as const) {
      const values = [0, 1, 2, 3, 4, 9007199254740991];
      const image = parseNifti(synthNifti1({ dims: [3, 2, 1], datatype, values }));
      expect(image.data).toBeInstanceOf(Float64Array);
      expect(Array.from(image.data)).toEqual(values);
    }
  });

  it('rejects a datatype it cannot represent', () => {
    const buffer = synthNifti1({ dims: [2, 1, 1], datatype: NiftiDataType.UINT8, values: [1, 2] });
    new DataView(buffer).setInt16(70, 128, true); // DT_RGB24
    expect(() => parseNifti(buffer)).toThrow(/Unsupported NIfTI datatype code 128/);
  });

  it('rejects a buffer that is not NIfTI at all', () => {
    const junk = new ArrayBuffer(400);
    new DataView(junk).setInt32(0, 12345, true);
    expect(() => parseNifti(junk)).toThrow(/Not a NIfTI file/);
  });
});

describe('endianness', () => {
  const spec: Nifti1Spec = {
    dims: [4, 3, 2],
    datatype: NiftiDataType.INT16,
    values: ramp(24, -12),
    pixdim: [0.5, 1.5, 2.5],
    sformCode: 2,
    srow: [0.5, 0, 0, -1, 0, 1.5, 0, -2, 0, 0, 2.5, -3],
    sclSlope: 4,
    sclInter: 0.5,
    descrip: 'endian probe',
    intentName: 'mask',
  };

  it('parses a big-endian file identically to its little-endian twin', () => {
    const little = parseNifti(synthNifti1({ ...spec, littleEndian: true }));
    const big = parseNifti(synthNifti1({ ...spec, littleEndian: false }));

    expect(big.header.littleEndian).toBe(false);
    expect(little.header.littleEndian).toBe(true);
    expect(Array.from(big.data)).toEqual(Array.from(little.data));
    expect(Array.from(big.header.affine)).toEqual(Array.from(little.header.affine));
    expect({ ...big.header, littleEndian: true, affine: null }).toEqual({
      ...little.header,
      affine: null,
    });
  });

  it('swaps every supported element width', () => {
    for (const datatype of [
      NiftiDataType.INT32,
      NiftiDataType.FLOAT32,
      NiftiDataType.FLOAT64,
      NiftiDataType.INT64,
    ] as const) {
      const values = [1, -2, 3, -4];
      const big = parseNifti(synthNifti1({ dims: [4, 1, 1], datatype, values, littleEndian: false }));
      expect(Array.from(big.data)).toEqual(values);
    }
  });
});

describe('scl_slope / scl_inter', () => {
  const base: Nifti1Spec = { dims: [2, 1, 1], datatype: NiftiDataType.UINT8, values: [1, 2] };

  it('normalises slope 0 to no scaling', () => {
    const header = parseNiftiHeader(synthNifti1({ ...base, sclSlope: 0, sclInter: 17 }));
    expect(header.sclSlope).toBe(1);
    expect(header.sclInter).toBe(0);
  });

  it('normalises NaN slope and intercept, which real writers emit when unset', () => {
    const header = parseNiftiHeader(synthNifti1({ ...base, sclSlope: NaN, sclInter: NaN }));
    expect(header.sclSlope).toBe(1);
    expect(header.sclInter).toBe(0);
  });

  it('keeps a genuine slope and intercept', () => {
    const header = parseNiftiHeader(synthNifti1({ ...base, sclSlope: 0.25, sclInter: -1.5 }));
    expect(header.sclSlope).toBe(0.25);
    expect(header.sclInter).toBe(-1.5);
  });
});

describe('affine selection', () => {
  const values = [1, 2, 3, 4, 5, 6];
  const dims: [number, number, number] = [3, 2, 1];

  it('prefers sform when both codes are set', () => {
    const header = parseNiftiHeader(
      synthNifti1({
        dims,
        datatype: NiftiDataType.UINT8,
        values,
        pixdim: [2, 3, 4],
        sformCode: 1,
        srow: [0, 0, 1, 5, 1, 0, 0, 6, 0, 1, 0, 7],
        qformCode: 1,
        qoffset: [10, 20, 30],
      }),
    );
    expect(header.affineSource).toBe('sform');
    expect(Array.from(header.affine)).toEqual([0, 0, 1, 5, 1, 0, 0, 6, 0, 1, 0, 7, 0, 0, 0, 1]);
  });

  it('falls back to the quaternion, identity rotation giving diag(pixdim)', () => {
    const header = parseNiftiHeader(
      synthNifti1({
        dims,
        datatype: NiftiDataType.UINT8,
        values,
        pixdim: [2, 3, 4],
        qformCode: 1,
        quatern: [0, 0, 0],
        qoffset: [10, 20, 30],
      }),
    );
    expect(header.affineSource).toBe('qform');
    expect(Array.from(header.affine)).toEqual([
      2, 0, 0, 10,
      0, 3, 0, 20,
      0, 0, 4, 30,
      0, 0, 0, 1,
    ]);
  });

  it('builds a 90 degree rotation from the quaternion', () => {
    // b = sin(45 deg), c = d = 0 is a +90 deg rotation about x, so
    // R = [[1,0,0],[0,0,-1],[0,1,0]] before the pixdim column scaling.
    const header = parseNiftiHeader(
      synthNifti1({
        dims,
        datatype: NiftiDataType.UINT8,
        values,
        pixdim: [2, 3, 4],
        qformCode: 1,
        quatern: [Math.SQRT1_2, 0, 0],
        qoffset: [10, 20, 30],
      }),
    );
    const expected = [
      2, 0, 0, 10,
      0, 0, -4, 20,
      0, 3, 0, 30,
      0, 0, 0, 1,
    ];
    // quatern_b is stored as float32, so `a` is recovered to about 1e-7.
    for (let i = 0; i < 16; i++) expect(header.affine[i]).toBeCloseTo(expected[i], 6);
  });

  it('applies qfac = -1 to the third column only', () => {
    const spec: Nifti1Spec = {
      dims,
      datatype: NiftiDataType.UINT8,
      values,
      pixdim: [2, 3, 4],
      qformCode: 1,
      quatern: [Math.SQRT1_2, 0, 0],
      qoffset: [10, 20, 30],
    };
    const plain = parseNiftiHeader(synthNifti1(spec)).affine;
    const flipped = parseNiftiHeader(synthNifti1({ ...spec, qfac: -1 })).affine;

    for (let row = 0; row < 3; row++) {
      for (const col of [0, 1, 3]) {
        expect(flipped[row * 4 + col]).toBe(plain[row * 4 + col]);
      }
      expect(flipped[row * 4 + 2]).toBe(-plain[row * 4 + 2]);
    }
  });

  it('treats qfac 0 as +1', () => {
    const spec: Nifti1Spec = {
      dims,
      datatype: NiftiDataType.UINT8,
      values,
      pixdim: [2, 3, 4],
      qformCode: 1,
      quatern: [0, 0, 0],
    };
    const zero = parseNiftiHeader(synthNifti1({ ...spec, qfac: 0 })).affine;
    expect(Array.from(zero)).toEqual(Array.from(parseNiftiHeader(synthNifti1(spec)).affine));
  });

  it('handles a quaternion whose components already sum to one', () => {
    // b = 1 means a 180 degree rotation about x; naively taking sqrt(1 - 1)
    // is fine here, but rounding can push the sum past 1 in real files.
    const header = parseNiftiHeader(
      synthNifti1({
        dims,
        datatype: NiftiDataType.UINT8,
        values,
        pixdim: [1, 1, 1],
        qformCode: 1,
        quatern: [1, 0, 0],
      }),
    );
    expect(Array.from(header.affine)).toEqual([
      1, 0, 0, 0,
      0, -1, 0, 0,
      0, 0, -1, 0,
      0, 0, 0, 1,
    ]);
  });

  it('ignores an sform_code that is set over an empty srow', () => {
    const header = parseNiftiHeader(
      synthNifti1({
        dims,
        datatype: NiftiDataType.UINT8,
        values,
        pixdim: [2, 3, 4],
        sformCode: 1,
        srow: new Array<number>(12).fill(0),
        qformCode: 1,
        quatern: [0, 0, 0],
        qoffset: [10, 20, 30],
      }),
    );
    expect(header.affineSource).toBe('qform');
    expect(header.affine[3]).toBe(10);
  });

  it('falls back to pixdim scaling when neither form is set', () => {
    const header = parseNiftiHeader(
      synthNifti1({ dims, datatype: NiftiDataType.UINT8, values, pixdim: [2, 3, 4] }),
    );
    expect(header.affineSource).toBe('pixdim');
    expect(Array.from(header.affine)).toEqual([
      2, 0, 0, 0,
      0, 3, 0, 0,
      0, 0, 4, 0,
      0, 0, 0, 1,
    ]);
  });
});

describe('spatial units', () => {
  const spec: Nifti1Spec = {
    dims: [2, 1, 1],
    datatype: NiftiDataType.UINT8,
    values: [1, 2],
    pixdim: [0.002, 0.003, 0.004],
    sformCode: 1,
    srow: [0.002, 0, 0, -0.1, 0, 0.003, 0, -0.2, 0, 0, 0.004, -0.3],
  };

  it('scales metres to millimetres, affine and pixdim together', () => {
    const header = parseNiftiHeader(synthNifti1({ ...spec, xyztUnits: 1 | 8 }));
    expect(header.spatialUnits).toBe(1);
    expect(header.pixdim[1]).toBeCloseTo(2, 6);
    expect(header.pixdim[3]).toBeCloseTo(4, 6);
    // Tolerances track float32 header storage: -0.1 m round-trips as
    // -0.10000000149, so scaling by 1000 carries the error up with it.
    expect(header.affine[0]).toBeCloseTo(2, 6);
    expect(header.affine[3]).toBeCloseTo(-100, 4);
    expect(header.affine[11]).toBeCloseTo(-300, 4);
  });

  it('scales micrometres down', () => {
    const header = parseNiftiHeader(synthNifti1({ ...spec, xyztUnits: 3 }));
    expect(header.spatialUnits).toBe(3);
    expect(header.affine[3]).toBeCloseTo(-0.0001, 10);
    expect(header.pixdim[1]).toBeCloseTo(0.000002, 12);
  });

  it('leaves millimetres alone', () => {
    const header = parseNiftiHeader(synthNifti1({ ...spec, xyztUnits: 2 | 16 }));
    expect(header.spatialUnits).toBe(2);
    expect(header.affine[3]).toBeCloseTo(-0.1, 6);
  });
});

describe('header text and layout', () => {
  it('stops descrip and intent_name at the NUL', () => {
    const buffer = synthNifti1({
      dims: [2, 1, 1],
      datatype: NiftiDataType.UINT8,
      values: [1, 2],
      descrip: 'FSL 5.0.10',
      intentName: 'liver',
    });
    // Junk after the terminator, which some writers leave behind.
    new Uint8Array(buffer).set([65, 66, 67], 148 + 11);
    new Uint8Array(buffer).set([88], 328 + 6);

    const header = parseNiftiHeader(buffer);
    expect(header.description).toBe('FSL 5.0.10');
    expect(header.intentName).toBe('liver');
  });

  it('honours a vox_offset past the usual 352', () => {
    const image = parseNifti(
      synthNifti1({
        dims: [4, 1, 1],
        datatype: NiftiDataType.INT16,
        values: [10, 20, 30, 40],
        padding: 64,
      }),
    );
    expect(image.header.voxOffset).toBe(416);
    expect(Array.from(image.data)).toEqual([10, 20, 30, 40]);
  });

  it('copies when vox_offset misaligns the element width', () => {
    const image = parseNifti(
      synthNifti1({
        dims: [4, 1, 1],
        datatype: NiftiDataType.INT16,
        values: [10, 20, 30, 40],
        padding: 1,
      }),
    );
    expect(image.header.voxOffset).toBe(353);
    expect(Array.from(image.data)).toEqual([10, 20, 30, 40]);
  });

  it('views the source buffer without copying when it can', () => {
    const buffer = synthNifti1({
      dims: [4, 1, 1],
      datatype: NiftiDataType.INT16,
      values: [10, 20, 30, 40],
    });
    expect(parseNifti(buffer).data.buffer).toBe(buffer);
  });

  it('reads a header without needing the voxels', () => {
    const full = synthNifti1({
      dims: [4, 4, 4],
      datatype: NiftiDataType.INT16,
      values: ramp(64),
      sformCode: 1,
      srow: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    });
    const headerOnly = full.slice(0, 348);

    const probed = parseNiftiHeader(headerOnly);
    expect(probed.dim.slice(0, 4)).toEqual([3, 4, 4, 4]);
    expect(probed).toEqual(parseNiftiHeader(full));
    expect(() => parseNifti(headerOnly)).toThrow(/runs past the end of the buffer/);
  });

  it('reports a truncated header rather than reading garbage', () => {
    const tiny = synthNifti1({ dims: [1, 1, 1], datatype: NiftiDataType.UINT8, values: [1] }).slice(0, 100);
    expect(() => parseNiftiHeader(tiny)).toThrow(/Truncated NIfTI-1 header/);
  });
});

describe('NIfTI-2', () => {
  it('parses a nibabel-written little-endian file', () => {
    const image = parseNifti(decodeBase64(NIFTI2_LE_BASE64));
    expect(image.header.version).toBe(2);
    expect(image.header.littleEndian).toBe(true);
    expect(image.header.dim.slice(0, 4)).toEqual([3, 2, 3, 4]);
    expect(image.header.datatype).toBe(NiftiDataType.INT16);
    expect(image.header.bitpix).toBe(16);
    expect(image.header.voxOffset).toBe(544);
    expect(image.header.sclSlope).toBe(2);
    expect(image.header.sclInter).toBe(5);
    expect(image.header.spatialUnits).toBe(2);
    expect(image.header.description).toBe('nifti2 fixture');
    expect(image.header.intentName).toBe('probe');
    expect(image.header.affineSource).toBe('sform');
    expect(Array.from(image.header.affine)).toEqual([
      2, 0, 0, -10,
      0, 3, 0, -20,
      0, 0, 4, -30,
      0, 0, 0, 1,
    ]);
    expect(image.header.pixdim.slice(1, 4)).toEqual([2, 3, 4]);
    expect(Array.from(image.data)).toEqual(ramp(24));
  });

  it('parses the big-endian twin identically', () => {
    const little = parseNifti(decodeBase64(NIFTI2_LE_BASE64));
    const big = parseNifti(decodeBase64(NIFTI2_BE_BASE64));
    expect(big.header.littleEndian).toBe(false);
    expect(Array.from(big.data)).toEqual(Array.from(little.data));
    expect(Array.from(big.header.affine)).toEqual(Array.from(little.header.affine));
    expect({ ...big.header, littleEndian: true, affine: null }).toEqual({
      ...little.header,
      affine: null,
    });
  });
});

describe('gunzipIfNeeded', () => {
  const plain = synthNifti1({
    dims: [4, 3, 2],
    datatype: NiftiDataType.INT16,
    values: ramp(24, -5),
    sformCode: 1,
    srow: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
  });

  it('returns a non-gzipped buffer untouched', async () => {
    expect(await gunzipIfNeeded(plain)).toBe(plain);
  });

  it('decompresses a gzipped buffer', async () => {
    const gz = toArrayBuffer(gzipSync(new Uint8Array(plain)));
    const out = await gunzipIfNeeded(gz);
    expect(new Uint8Array(out)).toEqual(new Uint8Array(plain));
    expect(Array.from(parseNifti(out).data)).toEqual(ramp(24, -5));
  });

  it('falls back to fflate when DecompressionStream is missing', async () => {
    const native = globalThis.DecompressionStream;
    // @ts-expect-error deliberately removing a global to exercise the fallback
    delete globalThis.DecompressionStream;
    try {
      const gz = toArrayBuffer(gzipSync(new Uint8Array(plain)));
      const out = await gunzipIfNeeded(gz);
      expect(new Uint8Array(out)).toEqual(new Uint8Array(plain));
    } finally {
      globalThis.DecompressionStream = native;
    }
  });

  it('handles a buffer too short to hold a magic number', async () => {
    const tiny = new ArrayBuffer(1);
    expect(await gunzipIfNeeded(tiny)).toBe(tiny);
  });
});

describe('real BDMAP_00000338 CT', () => {
  it('parses the header and voxels off disk', async () => {
    const image = parseNifti(await gunzipIfNeeded(readCaseFile('ct.nii.gz')));
    const h = image.header;

    expect(h.version).toBe(1);
    expect(h.littleEndian).toBe(true);
    expect(h.dim.slice(0, 4)).toEqual([3, 502, 348, 71]);
    expect(h.datatype).toBe(NiftiDataType.INT16);
    expect(h.bitpix).toBe(16);
    expect(h.voxOffset).toBe(352);
    expect(h.spatialUnits).toBe(2);
    expect(h.qformCode).toBe(1);
    expect(h.sformCode).toBe(1);
    expect(h.affineSource).toBe('sform');
    expect(h.description).toBe('5.0.10');
    expect(h.sclSlope).toBeCloseTo(0.030518043786287308, 12);
    expect(h.sclInter).toBeCloseTo(0.015259021893143654, 12);
    expect(h.pixdim[1]).toBeCloseTo(0.81641, 5);
    expect(h.pixdim[2]).toBeCloseTo(0.81641, 5);
    expect(h.pixdim[3]).toBeCloseTo(2.5, 12);

    const expectedAffine = [
      0.81640601, 0, 0, -417.18347,
      0, 0.81640601, 0, -417.18347,
      0, 0, 2.5, 0,
      0, 0, 0, 1,
    ];
    for (let i = 0; i < 16; i++) expect(h.affine[i]).toBeCloseTo(expectedAffine[i], 4);

    expect(image.data).toBeInstanceOf(Int16Array);
    expect(image.data.length).toBe(502 * 348 * 71);
    // Slope and intercept are chosen so the full int16 range is exactly
    // -1000..1000 HU, which is the sanity check that scaling is wired up.
    expect(image.data[0] * h.sclSlope + h.sclInter).toBeCloseTo(-1000, 5);
  });

  it('parses every segmentation mask on the same grid', async () => {
    const names = [
      'aorta', 'gall_bladder', 'kidney_left', 'kidney_right', 'liver',
      'pancreas', 'postcava', 'spleen', 'stomach',
    ];
    const ct = parseNiftiHeader(await gunzipIfNeeded(readCaseFile('ct.nii.gz')));
    for (const name of names) {
      const mask = parseNifti(await gunzipIfNeeded(readCaseFile(`segmentations/${name}.nii.gz`)));
      expect(mask.header.datatype, name).toBe(NiftiDataType.INT8);
      expect(mask.header.dim.slice(1, 4), name).toEqual([502, 348, 71]);
      // Masks carry no scaling, so the parser must not invent any.
      expect(mask.header.sclSlope, name).toBe(1);
      expect(mask.header.sclInter, name).toBe(0);
      for (let i = 0; i < 16; i++) {
        expect(mask.header.affine[i], `${name}[${i}]`).toBeCloseTo(ct.affine[i], 9);
      }
    }
  });
});
