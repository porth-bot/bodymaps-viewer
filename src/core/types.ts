/**
 * Shared data contracts for the viewer.
 *
 * After loading, every volume is reordered into canonical RAS voxel order: i
 * toward the patient's Right, j toward Anterior, k toward Superior. Index math
 * elsewhere never has to think about the source file's orientation again.
 * `NiftiImage` is what comes off disk, in any orientation; `Volume` is what the
 * rest of the app sees, always RAS.
 */

/** The subset of the spec's datatype codes that occurs in practice. */
export const enum NiftiDataType {
  UINT8 = 2,
  INT16 = 4,
  INT32 = 8,
  FLOAT32 = 16,
  FLOAT64 = 64,
  INT8 = 256,
  UINT16 = 512,
  UINT32 = 768,
  INT64 = 1024,
  UINT64 = 1280,
}

export type TypedNumberArray =
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

/** Row-major 4x4, indexed m[row * 4 + col]. Maps voxel (i,j,k,1) to world mm. */
export type Mat4 = Float64Array;

export interface NiftiHeader {
  version: 1 | 2;
  littleEndian: boolean;
  /** dim[0] is the dimensionality; dim[1..7] are sizes. */
  dim: number[];
  datatype: NiftiDataType;
  bitpix: number;
  pixdim: number[];
  voxOffset: number;
  sclSlope: number;
  sclInter: number;
  calMin: number;
  calMax: number;
  qformCode: number;
  sformCode: number;
  /** Spatial unit code from xyzt_units (1=meter, 2=mm, 3=micron). */
  spatialUnits: number;
  description: string;
  intentName: string;
  /** Per the NIfTI rules: sform if set, else qform, else scaling-only. */
  affine: Mat4;
  affineSource: 'sform' | 'qform' | 'pixdim';
  /** 3D volumes in the file. Above 1 means a 4D series; only the first is read. */
  volumeCount: number;
}

/** A parsed NIfTI file, still in its own native voxel order. */
export interface NiftiImage {
  header: NiftiHeader;
  /** Raw voxel values, NOT rescaled. Length = dim[1]*dim[2]*dim[3]. */
  data: TypedNumberArray;
}

export type AxisCode = 'L' | 'R' | 'A' | 'P' | 'S' | 'I';
/** Orientation code, e.g. ['R','A','S'] or ['L','P','S']. */
export type AxCodes = [AxisCode, AxisCode, AxisCode];

/**
 * A scalar volume in canonical RAS voxel order. `values` holds raw stored
 * values; apply `slope`/`intercept` for real units (HU for CT). Not
 * pre-multiplied, so the HU probe reports exact values and a 12M-voxel volume
 * does not cost a float64 array.
 */
export interface Volume {
  dims: [number, number, number];
  /** mm per voxel along i, j, k. */
  spacing: [number, number, number];
  /** Voxel-to-world, already adjusted for the RAS reorientation. */
  affine: Mat4;
  /** Orientation of the ORIGINAL file, kept for display ("loaded as LPS"). */
  originalAxCodes: AxCodes;
  /** Above 1 is a 4D series; only the first frame loads, and the UI says so. */
  volumeCount: number;
  values: TypedNumberArray;
  slope: number;
  intercept: number;
  /** Rescaled, i.e. real-world units. Computed once at load. */
  min: number;
  max: number;
  /** Physical extent, mm per axis. */
  extent: [number, number, number];
}

export interface Structure {
  /** Label index in the packed label volume, 1-based. 0 is background. */
  index: number;
  /** Machine name as it appeared in the file, e.g. "kidney_left". */
  key: string;
  name: string;
  /** sRGB 0-255. */
  color: [number, number, number];
  visible: boolean;
  voxelCount: number;
  /** Millilitres. */
  volumeMl: number;
  /** Mean of the underlying scalar volume inside the mask, in HU. */
  meanHu: number;
  /** Inclusive voxel bounding box [i0,j0,k0,i1,j1,k1] in RAS voxel order. */
  bounds: [number, number, number, number, number, number];
  /** Centroid in voxel coordinates. */
  centroid: [number, number, number];
}

/**
 * Every structure packed into one uint8 volume, so the GPU carries one texture
 * instead of N. Overlaps go to the smaller structure: a big organ swallowing a
 * vessel reads worse than the reverse.
 */
export interface LabelVolume {
  dims: [number, number, number];
  spacing: [number, number, number];
  affine: Mat4;
  values: Uint8Array;
  structures: Structure[];
  /** Number of voxels claimed by more than one source mask. */
  overlapVoxels: number;
}

/** A triangle mesh produced by isosurface extraction, in world (mm) space. */
export interface Mesh {
  /** The structure this mesh belongs to. */
  index: number;
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  /** Axis-aligned bounds in world mm: [minX,minY,minZ,maxX,maxY,maxZ]. */
  bounds: [number, number, number, number, number, number];
  triangleCount: number;
}

export type ViewKind = 'axial' | 'coronal' | 'sagittal' | 'volume';

export interface WindowLevel {
  /** Centre, HU. */
  level: number;
  /** Width, HU. */
  window: number;
}

export interface Preset extends WindowLevel {
  id: string;
  name: string;
  hotkey?: string;
}

/** Reported from the workers back to the UI. */
export interface LoadProgress {
  stage: string;
  /** 0..1, or null when indeterminate. */
  fraction: number | null;
  detail?: string;
}
