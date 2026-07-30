/**
 * Shared data contracts for the viewer.
 *
 * The whole app agrees on one convention: after loading, every volume is
 * resampled into *canonical RAS voxel order*, meaning
 *
 *   i increases toward the patient's Right
 *   j increases toward Anterior
 *   k increases toward Superior
 *
 * so index math elsewhere never has to think about the source file's
 * orientation again. `NiftiImage` is what comes off disk (any orientation);
 * `Volume` is what the rest of the app consumes (always RAS).
 */

/** NIfTI datatype codes we understand (subset of the spec that occurs in practice). */
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
  /** 1 for NIfTI-1, 2 for NIfTI-2. */
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
  /** Affine chosen per the NIfTI rules: sform if set, else qform, else scaling-only. */
  affine: Mat4;
  /** Which of the three methods produced `affine`. Useful for the info panel. */
  affineSource: 'sform' | 'qform' | 'pixdim';
}

/** A parsed NIfTI file, still in its own native voxel order. */
export interface NiftiImage {
  header: NiftiHeader;
  /** Raw voxel values, NOT rescaled. Length = dim[1]*dim[2]*dim[3]. */
  data: TypedNumberArray;
}

/** Three-letter orientation code, e.g. ['R','A','S'] or ['L','P','S']. */
export type AxisCode = 'L' | 'R' | 'A' | 'P' | 'S' | 'I';
export type AxCodes = [AxisCode, AxisCode, AxisCode];

/**
 * A scalar volume in canonical RAS voxel order.
 *
 * `values` holds raw stored values; apply `slope`/`intercept` to get real
 * units (Hounsfield units for CT). We keep them separate rather than
 * pre-multiplying so the HU probe can report exact values and so we do not
 * pay a float64 array for a 12M-voxel volume.
 */
export interface Volume {
  dims: [number, number, number];
  /** mm per voxel along i, j, k. */
  spacing: [number, number, number];
  /** Voxel-to-world affine, already adjusted for the RAS reorientation. */
  affine: Mat4;
  /** Orientation of the ORIGINAL file, kept for display ("loaded as LPS"). */
  originalAxCodes: AxCodes;
  values: TypedNumberArray;
  slope: number;
  intercept: number;
  /** Rescaled min/max, i.e. real-world units. Computed once at load. */
  min: number;
  max: number;
  /** Physical extent in mm along each axis. */
  extent: [number, number, number];
}

/** One annotated structure. */
export interface Structure {
  /** Label index in the packed label volume, 1-based. 0 is background. */
  index: number;
  /** Machine name as it appeared in the file, e.g. "kidney_left". */
  key: string;
  /** Human label for the UI, e.g. "Kidney (left)". */
  name: string;
  /** sRGB 0-255. */
  color: [number, number, number];
  visible: boolean;
  /** Voxel count and derived volume in millilitres. */
  voxelCount: number;
  volumeMl: number;
  /** Mean value of the underlying scalar volume inside the mask, in HU. */
  meanHu: number;
  /** Inclusive voxel bounding box [i0,j0,k0,i1,j1,k1] in RAS voxel order. */
  bounds: [number, number, number, number, number, number];
  /** Centroid in voxel coordinates. */
  centroid: [number, number, number];
}

/**
 * All structures packed into one uint8 volume (so the GPU carries one texture
 * instead of N). Overlaps are resolved by area priority: smaller structures
 * win, because a big organ swallowing a small vessel reads worse than the
 * reverse.
 */
export interface LabelVolume {
  dims: [number, number, number];
  spacing: [number, number, number];
  affine: Mat4;
  /** Values are structure indices; 0 = background. */
  values: Uint8Array;
  structures: Structure[];
  /** Number of voxels claimed by more than one source mask. */
  overlapVoxels: number;
}

/** A triangle mesh produced by isosurface extraction, in world (mm) space. */
export interface Mesh {
  /** Structure index this mesh belongs to. */
  index: number;
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  /** Axis-aligned bounds in world mm: [minX,minY,minZ,maxX,maxY,maxZ]. */
  bounds: [number, number, number, number, number, number];
  triangleCount: number;
}

/** The three orthogonal planes, plus the 3D view. */
export type ViewKind = 'axial' | 'coronal' | 'sagittal' | 'volume';

export interface WindowLevel {
  /** Window centre in HU. */
  level: number;
  /** Window width in HU. */
  window: number;
}

export interface Preset extends WindowLevel {
  id: string;
  name: string;
  /** Keyboard digit that selects it, if any. */
  hotkey?: string;
}

/** Progress reporting from workers back to the UI. */
export interface LoadProgress {
  stage: string;
  /** 0..1, or null when indeterminate. */
  fraction: number | null;
  detail?: string;
}
