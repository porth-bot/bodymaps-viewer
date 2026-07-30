/**
 * Pointer, wheel and keyboard handling. Bindings copy 3D Slicer and the PACS
 * workstations: left drag navigates, right drag is window and level, middle
 * drag pans, wheel scrolls slices. Not worth inventing better ones, a
 * radiologist already knows these.
 */

import type { OrbitCamera } from '../camera/orbit';
import type { Renderer } from '../gl/renderer';
import { rectAt, type ViewportRect } from '../gl/layout';
import { PLANES, planeUvToVoxel, screenToPlaneUv } from '../gl/planes';
import { clampVoxel } from '../core/volume';
import type { Store, Tool, Measurement } from './store';
import type { Vec3 } from '../core/mat4';

type DragMode = 'none' | 'navigate' | 'window' | 'pan' | 'zoom' | 'orbit' | 'orbitPan' | 'ruler';

export interface InteractionOptions {
  store: Store;
  renderer: Renderer;
  camera: OrbitCamera;
  requestRender(): void;
  onPreset(id: string): void;
  onResetView(): void;
}

let measurementId = 1;

export class Interactions {
  private drag: DragMode = 'none';
  /**
   * Which pane the drag started in, by identity rather than by geometry.
   * Caching the rect meant a layout change mid-drag (double click, keyboard
   * shortcut) fed stale coordinates to the pointer maths and the crosshair
   * jumped to a position computed against a pane that had moved.
   */
  private dragViewKind: string | null = null;
  private lastX = 0;
  private lastY = 0;
  private startWindow = { level: 0, window: 1 };
  private rulerStart: Vec3 | null = null;

  constructor(private el: HTMLElement, private opts: InteractionOptions) {
    el.addEventListener('pointerdown', this.onPointerDown);
    el.addEventListener('pointermove', this.onPointerMove);
    el.addEventListener('pointerup', this.onPointerUp);
    el.addEventListener('pointercancel', this.onPointerUp);
    el.addEventListener('pointerleave', this.onPointerLeave);
    el.addEventListener('wheel', this.onWheel, { passive: false });
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('dblclick', this.onDoubleClick);
    window.addEventListener('keydown', this.onKeyDown);
  }

  dispose(): void {
    window.removeEventListener('keydown', this.onKeyDown);
  }

  private local(e: PointerEvent | WheelEvent | MouseEvent): { x: number; y: number } {
    const r = this.el.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  private viewAt(x: number, y: number): ViewportRect | null {
    return rectAt(this.opts.renderer.viewportRects, x, y);
  }

  /** Continuous voxel coordinate under a pointer position inside a slice pane. */
  private voxelAt(rect: ViewportRect, x: number, y: number): Vec3 | null {
    const state = this.opts.store.get();
    if (!state.volume || rect.view === 'volume') return null;
    const spec = PLANES[rect.view as 'axial' | 'coronal' | 'sagittal'];
    const t = this.opts.renderer.viewTransformFor(rect, state.views);
    if (!t) return null;
    const [u, v] = screenToPlaneUv(t, x - rect.x, y - rect.y);
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    const sliceIndex = state.crosshair[spec.sliceAxis];
    return planeUvToVoxel(spec, state.volume.dims, sliceIndex, u, v);
  }

  private modeFor(e: PointerEvent, rect: ViewportRect, tool: Tool): DragMode {
    const is3D = rect.view === 'volume';
    if (e.button === 2) return is3D ? 'orbitPan' : 'window';
    if (e.button === 1) return is3D ? 'orbitPan' : 'pan';
    // Left button: modifiers first, then the active tool.
    if (e.shiftKey) return is3D ? 'orbit' : 'zoom';
    if (e.altKey || e.metaKey) return is3D ? 'orbitPan' : 'pan';
    if (is3D) return 'orbit';
    switch (tool) {
      case 'window': return 'window';
      case 'pan': return 'pan';
      case 'zoom': return 'zoom';
      case 'ruler': return 'ruler';
      default: return 'navigate';
    }
  }

  private onPointerDown = (e: PointerEvent): void => {
    const { x, y } = this.local(e);
    const rect = this.viewAt(x, y);
    if (!rect) return;
    const state = this.opts.store.get();
    if (!state.volume) return;

    this.el.setPointerCapture(e.pointerId);
    this.dragViewKind = rect.view;
    this.lastX = x;
    this.lastY = y;
    this.drag = this.modeFor(e, rect, state.tool);
    this.startWindow = { ...state.windowLevel };
    this.opts.store.set({ hoverView: rect.view });

    if (this.drag === 'navigate') {
      const voxel = this.voxelAt(rect, x, y);
      if (voxel) this.opts.store.set({ crosshair: clampVoxel(state.volume, voxel) });
    } else if (this.drag === 'ruler') {
      const voxel = this.voxelAt(rect, x, y);
      if (voxel) this.rulerStart = clampVoxel(state.volume, voxel);
    }
    this.opts.requestRender();
    e.preventDefault();
  };

  private onPointerMove = (e: PointerEvent): void => {
    const { x, y } = this.local(e);
    const state = this.opts.store.get();
    if (!state.volume) return;

    if (this.drag === 'none') {
      const rect = this.viewAt(x, y);
      const probe = rect && rect.view !== 'volume' ? this.voxelAt(rect, x, y) : null;
      const nextHover = rect ? rect.view : null;
      if (nextHover !== state.hoverView || !sameVoxel(probe, state.probe)) {
        this.opts.store.set({ hoverView: nextHover, probe });
        this.opts.requestRender();
      }
      return;
    }

    // Re-resolve the pane each move; see dragViewKind.
    const rect = this.dragViewKind
      ? this.opts.renderer.viewportRects.find((r) => r.view === this.dragViewKind) ?? null
      : null;
    if (!rect) return;
    const dx = x - this.lastX;
    const dy = y - this.lastY;

    switch (this.drag) {
      case 'navigate': {
        const voxel = this.voxelAt(rect, x, y);
        if (voxel) this.opts.store.set({ crosshair: clampVoxel(state.volume, voxel), probe: voxel });
        break;
      }

      case 'window': {
        // Horizontal widens, vertical raises the level. Scaling the step by the
        // current window keeps the same drag equally fast in a 150 HU liver
        // window and a 2000 HU bone window.
        const scale = Math.max(this.startWindow.window, 50) / 250;
        this.opts.store.set((s) => ({
          windowLevel: {
            window: Math.max(1, s.windowLevel.window + dx * scale * 2),
            level: s.windowLevel.level - dy * scale * 2,
          },
          presetId: null,
        }));
        break;
      }

      case 'pan': {
        const t = this.opts.renderer.viewTransformFor(rect, state.views);
        if (t && t.pixelsPerMm > 0) {
          const cur = state.views[rect.view] ?? { zoom: 1, pan: [0, 0] as [number, number] };
          this.opts.store.setViewSettings(rect.view, {
            pan: [cur.pan[0] + dx / t.pixelsPerMm, cur.pan[1] - dy / t.pixelsPerMm],
          });
        }
        break;
      }

      case 'zoom': {
        const cur = state.views[rect.view] ?? { zoom: 1, pan: [0, 0] as [number, number] };
        const factor = Math.exp(-dy * 0.008);
        this.opts.store.setViewSettings(rect.view, {
          zoom: Math.max(0.15, Math.min(24, cur.zoom * factor)),
        });
        break;
      }

      case 'ruler': {
        const voxel = this.voxelAt(rect, x, y);
        if (voxel && this.rulerStart && rect.view !== 'volume') {
          this.opts.store.set({
            pendingMeasurement: makeMeasurement(
              rect.view as 'axial' | 'coronal' | 'sagittal',
              this.rulerStart, clampVoxel(state.volume, voxel), state.volume.spacing,
              state.crosshair,
            ),
          });
        }
        break;
      }

      case 'orbit':
        this.opts.camera.orbit(-dx * 0.008, dy * 0.008);
        break;

      case 'orbitPan':
        this.opts.camera.pan(dx, dy, rect.height);
        break;
    }

    this.lastX = x;
    this.lastY = y;
    this.opts.requestRender();
    e.preventDefault();
  };

  private onPointerUp = (e: PointerEvent): void => {
    if (this.drag === 'ruler') {
      const pending = this.opts.store.get().pendingMeasurement;
      // Drop zero-length rulers so a stray click does not litter the image.
      if (pending && pending.lengthMm > 0.5) {
        this.opts.store.set((s) => ({
          measurements: [...s.measurements, pending],
          pendingMeasurement: null,
        }));
      } else {
        this.opts.store.set({ pendingMeasurement: null });
      }
      this.rulerStart = null;
    }
    this.drag = 'none';
    this.dragViewKind = null;
    if (this.el.hasPointerCapture(e.pointerId)) this.el.releasePointerCapture(e.pointerId);
    this.opts.requestRender();
  };

  private onPointerLeave = (): void => {
    if (this.drag === 'none') {
      this.opts.store.set({ hoverView: null, probe: null });
      this.opts.requestRender();
    }
  };

  private onWheel = (e: WheelEvent): void => {
    const { x, y } = this.local(e);
    const rect = this.viewAt(x, y);
    if (!rect) return;
    const state = this.opts.store.get();
    if (!state.volume) return;
    e.preventDefault();

    if (rect.view === 'volume') {
      this.opts.camera.dolly(Math.exp(e.deltaY * 0.0012));
      this.opts.requestRender();
      return;
    }

    const spec = PLANES[rect.view as 'axial' | 'coronal' | 'sagittal'];

    if (e.ctrlKey || e.metaKey) {
      // Zoom about the pointer: pan by however far the point under the cursor
      // would otherwise drift.
      const t = this.opts.renderer.viewTransformFor(rect, state.views);
      const cur = state.views[rect.view] ?? { zoom: 1, pan: [0, 0] as [number, number] };
      const factor = Math.exp(-e.deltaY * 0.0015);
      const next = Math.max(0.15, Math.min(24, cur.zoom * factor));
      if (t && t.pixelsPerMm > 0) {
        const px = x - rect.x - rect.width / 2;
        const py = rect.height / 2 - (y - rect.y);
        const before = t.pixelsPerMm;
        const after = (before / cur.zoom) * next;
        this.opts.store.setViewSettings(rect.view, {
          zoom: next,
          pan: [
            cur.pan[0] + px * (1 / after - 1 / before),
            cur.pan[1] + py * (1 / after - 1 / before),
          ],
        });
      } else {
        this.opts.store.setViewSettings(rect.view, { zoom: next });
      }
      this.opts.requestRender();
      return;
    }

    const step = e.deltaY > 0 ? 1 : -1;
    this.stepSlice(spec.sliceAxis, step);
  };

  private onDoubleClick = (e: MouseEvent): void => {
    const { x, y } = this.local(e);
    const rect = this.viewAt(x, y);
    if (!rect) return;
    const state = this.opts.store.get();
    this.opts.store.set({ layout: state.layout === 'fourUp' ? (rect.view as never) : 'fourUp' });
    this.opts.requestRender();
  };

  private stepSlice(axis: 0 | 1 | 2, step: number): void {
    const state = this.opts.store.get();
    if (!state.volume) return;
    const next: Vec3 = [...state.crosshair] as Vec3;
    next[axis] = Math.max(0, Math.min(state.volume.dims[axis] - 1, Math.round(next[axis]) + step));
    this.opts.store.set({ crosshair: next });
    this.opts.requestRender();
  }

  private onKeyDown = (e: KeyboardEvent): void => {
    // Leave browser and OS chords alone. Every binding here is an unmodified
    // key or a Shift chord, so without this Cmd+R reset the views and
    // swallowed the reload, and Ctrl+L, Cmd+P and friends did the same.
    // Shift is deliberately not in this test: Shift+Arrow is a real binding.
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    const target = e.target as HTMLElement | null;
    if (target) {
      const tag = target.tagName;
      const type = tag === 'INPUT' ? (target as HTMLInputElement).type : '';
      // Only text entry should swallow every key. Treating all INPUTs that way
      // killed the shortcuts for the rest of the session the moment anyone
      // clicked a checkbox or a slider, because focus stays on that control.
      const consumesText =
        tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable ||
        (tag === 'INPUT' && !/^(checkbox|radio|button|submit|reset|file|range)$/.test(type));
      if (consumesText) return;
      // A focused widget still owns the keys it natively handles.
      if (e.key === ' ' || e.key === 'Enter') return;
      if (type === 'range' && /^(Arrow|Home|End|Page)/.test(e.key)) return;
    }

    const store = this.opts.store;
    const state = store.get();
    if (!state.volume) return;

    const hover = state.hoverView && state.hoverView !== 'volume'
      ? PLANES[state.hoverView as 'axial' | 'coronal' | 'sagittal'].sliceAxis
      : 2;

    let handled = true;
    switch (e.key) {
      // All four arrows take the Shift jump. Only the horizontal pair used to,
      // which made the documented "Shift for 10" false half the time.
      case 'ArrowUp': case 'PageUp': this.stepSlice(hover, e.shiftKey ? 10 : 1); break;
      case 'ArrowDown': case 'PageDown': this.stepSlice(hover, e.shiftKey ? -10 : -1); break;
      case 'ArrowRight': this.stepSlice(hover, e.shiftKey ? 10 : 1); break;
      case 'ArrowLeft': this.stepSlice(hover, e.shiftKey ? -10 : -1); break;

      case 'a': store.set({ layout: 'axial' }); break;
      case 'c': store.set({ layout: 'coronal' }); break;
      case 's': store.set({ layout: 'sagittal' }); break;
      case 'v': store.set({ layout: 'volume' }); break;
      case 'f': store.set({ layout: 'fourUp' }); break;
      case 'g': store.set({ layout: 'row' }); break;

      case 'l': store.set({ showLabels: !state.showLabels }); break;
      case 'o': store.set({ labelOutline: !state.labelOutline }); break;
      case 'm': store.set({ showMeshes: !state.showMeshes }); break;
      case 'x': store.set({ show3DVolume: !state.show3DVolume }); break;
      case 'p': store.set({ showSlicesIn3D: !state.showSlicesIn3D }); break;
      case 'h': store.set({ showCrosshair: !state.showCrosshair }); break;
      case 'r': this.opts.onResetView(); break;
      case 'Escape': store.set({ pendingMeasurement: null, measurements: [] }); break;

      default:
        if (/^[1-9]$/.test(e.key)) this.opts.onPreset(e.key);
        else handled = false;
    }

    if (handled) {
      e.preventDefault();
      this.opts.requestRender();
    }
  };
}

function sameVoxel(a: Vec3 | null, b: readonly number[] | null): boolean {
  if (a === null || b === null) return a === b;
  return Math.round(a[0]) === Math.round(b[0])
    && Math.round(a[1]) === Math.round(b[1])
    && Math.round(a[2]) === Math.round(b[2]);
}

function makeMeasurement(
  view: 'axial' | 'coronal' | 'sagittal',
  a: Vec3, b: Vec3,
  spacing: [number, number, number],
  crosshair: readonly number[],
): Measurement {
  // Length in millimetres, not voxels. With 0.82 x 0.82 x 2.5 mm voxels a
  // measurement in voxel units would be meaningless across planes.
  const dx = (a[0] - b[0]) * spacing[0];
  const dy = (a[1] - b[1]) * spacing[1];
  const dz = (a[2] - b[2]) * spacing[2];
  return {
    id: measurementId++,
    view,
    slice: Math.round(crosshair[PLANES[view].sliceAxis]),
    a: [...a] as Vec3,
    b: [...b] as Vec3,
    lengthMm: Math.hypot(dx, dy, dz),
  };
}
