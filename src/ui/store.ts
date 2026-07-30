/**
 * Application state.
 *
 * One plain object, one update function, one subscriber list. A framework
 * would not earn its bytes here: the render path already takes the whole state
 * every frame, so there is nothing for a diffing layer to do.
 */

import type { LabelVolume, LoadProgress, Structure, Volume, WindowLevel } from '../core/types';
import type { LayoutMode } from '../gl/layout';
import type { VolumeRenderMode, ViewSettings } from '../gl/renderer';

export type Tool = 'navigate' | 'window' | 'zoom' | 'pan' | 'ruler';
export type MeshStatus = 'none' | 'queued' | 'building' | 'ready' | 'error';

export interface Measurement {
  id: number;
  /** Which slice view it was drawn on. */
  view: 'axial' | 'coronal' | 'sagittal';
  /** Slice index it belongs to, so it only shows on its own slice. */
  slice: number;
  /** Endpoints in continuous voxel coordinates. */
  a: [number, number, number];
  b: [number, number, number];
  lengthMm: number;
}

export interface AppState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  progress: LoadProgress | null;
  error: string | null;

  caseName: string;
  volume: Volume | null;
  labels: LabelVolume | null;

  /** Continuous voxel coordinates in canonical RAS order. */
  crosshair: [number, number, number];
  windowLevel: WindowLevel;
  presetId: string | null;

  showLabels: boolean;
  labelOpacity: number;
  labelOutline: boolean;
  outlineWidth: number;
  structures: Structure[];
  meshStatus: Record<number, MeshStatus>;

  layout: LayoutMode;
  views: Record<string, ViewSettings>;
  showCrosshair: boolean;
  showOrientationLabels: boolean;

  show3DVolume: boolean;
  volumeMode: VolumeRenderMode;
  volumeDensity: number;
  volumeShade: boolean;
  volumeQuality: number;
  volumeLabelBoost: number;

  showMeshes: boolean;
  meshOpacity: number;
  showSlicesIn3D: boolean;
  showBoundingBox: boolean;

  tool: Tool;
  measurements: Measurement[];
  /** In-progress ruler drag, drawn but not yet committed. */
  pendingMeasurement: Measurement | null;

  /** Voxel under the pointer, for the data probe. Null when off-image. */
  probe: [number, number, number] | null;
  hoverView: string | null;

  timings: { downloadMs: number; labelsMs: number; meshMs: number };
}

function defaultViews(): Record<string, ViewSettings> {
  return {
    axial: { zoom: 1, pan: [0, 0] },
    coronal: { zoom: 1, pan: [0, 0] },
    sagittal: { zoom: 1, pan: [0, 0] },
  };
}

export function initialState(): AppState {
  return {
    status: 'idle',
    progress: null,
    error: null,

    caseName: '',
    volume: null,
    labels: null,

    crosshair: [0, 0, 0],
    // Soft tissue window until the scan arrives and a real preset is chosen.
    windowLevel: { level: 50, window: 400 },
    presetId: 'soft-tissue',

    showLabels: true,
    labelOpacity: 0.45,
    labelOutline: false,
    outlineWidth: 1,
    structures: [],
    meshStatus: {},

    layout: 'fourUp',
    views: defaultViews(),
    showCrosshair: true,
    showOrientationLabels: true,

    show3DVolume: true,
    volumeMode: 'composite',
    volumeDensity: 0.9,
    volumeShade: true,
    volumeQuality: 1,
    volumeLabelBoost: 0.35,

    showMeshes: true,
    meshOpacity: 1,
    // Off by default, as in 3D Slicer. At the opening camera angle the axial
    // plane is nearly edge-on, so it contributes almost nothing visually while
    // occluding a wide band of the volume behind it. Press P or use the 3D
    // panel to bring the planes back.
    showSlicesIn3D: false,
    showBoundingBox: false,

    tool: 'navigate',
    measurements: [],
    pendingMeasurement: null,

    probe: null,
    hoverView: null,

    timings: { downloadMs: 0, labelsMs: 0, meshMs: 0 },
  };
}

type Listener = (state: AppState, changed: ReadonlySet<keyof AppState>) => void;

export class Store {
  private state: AppState = initialState();
  private listeners = new Set<Listener>();

  get(): Readonly<AppState> {
    return this.state;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  /**
   * Merge a patch and notify. The changed-key set lets expensive subscribers
   * (rebuilding the structure list, re-uploading the LUT) skip work when an
   * unrelated field moved, which matters because dragging window/level fires
   * this on every pointer move.
   */
  set(patch: Partial<AppState> | ((s: Readonly<AppState>) => Partial<AppState>)): void {
    const delta = typeof patch === 'function' ? patch(this.state) : patch;
    const changed = new Set<keyof AppState>();
    for (const key of Object.keys(delta) as (keyof AppState)[]) {
      if (!Object.is(this.state[key], delta[key])) changed.add(key);
    }
    if (changed.size === 0) return;
    this.state = { ...this.state, ...delta };
    for (const fn of this.listeners) fn(this.state, changed);
  }

  /** Replace one structure by index, preserving list order. */
  updateStructure(index: number, patch: Partial<Structure>): void {
    this.set((s) => ({
      structures: s.structures.map((st) => (st.index === index ? { ...st, ...patch } : st)),
    }));
  }

  setViewSettings(view: string, patch: Partial<ViewSettings>): void {
    this.set((s) => ({
      views: { ...s.views, [view]: { ...(s.views[view] ?? { zoom: 1, pan: [0, 0] }), ...patch } },
    }));
  }

  setMeshStatus(index: number, status: MeshStatus): void {
    this.set((s) => ({ meshStatus: { ...s.meshStatus, [index]: status } }));
  }
}
