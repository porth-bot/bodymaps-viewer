/**
 * Entry point. Owns the render loop and connects the loader, the store, the
 * renderer and the DOM.
 */

import './style.css';

import { OrbitCamera } from './camera/orbit';
import { CaseLoader, sampleCase, sourcesFromFiles, type CaseSources } from './core/caseLoader';
import { buildLut } from './core/labelmap';
import { WINDOW_PRESETS, autoWindow } from './core/presets';
import type { LabelVolume, Mesh, Volume } from './core/types';
import { GLError } from './gl/gl';
import { Renderer, type RenderState } from './gl/renderer';
import { Interactions } from './ui/interactions';
import { describeProbe, Overlay } from './ui/overlay';
import { buildSidebar, buildToolbar, type PanelDeps } from './ui/panels';
import { Store, type AppState } from './ui/store';

const $ = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing element #${id}`);
  return node as T;
};

function boot(): void {
  const glCanvas = $<HTMLCanvasElement>('gl');
  const overlayCanvas = $<HTMLCanvasElement>('overlay');
  const store = new Store();
  const camera = new OrbitCamera();

  let renderer: Renderer;
  try {
    renderer = new Renderer(glCanvas);
  } catch (err) {
    showFatal(err instanceof GLError ? err.message : String(err));
    return;
  }

  const overlay = new Overlay(overlayCanvas);

  // --- render loop ------------------------------------------------------
  // Redraws only when something changed. A viewer that spins the GPU at 60 fps
  // on a still image is a laptop-battery problem, and the raycaster is the
  // most expensive thing on the page.
  let dirty = true;
  const requestRender = () => { dirty = true; };

  // Snapshot capture has to happen in the same task as the draw. The context
  // is created without preserveDrawingBuffer (keeping it costs a full-frame
  // copy every frame), so the colour buffer is only readable between the draw
  // call and the browser compositing it away.
  let pendingCapture: ((url: string) => void) | null = null;

  function frame(): void {
    if (dirty || pendingCapture) {
      dirty = false;
      const state = store.get();
      renderer.render(toRenderState(state, camera));
      overlay.draw(state, renderer, camera);
      updateStatusBar(state, renderer);

      if (pendingCapture) {
        const resolve = pendingCapture;
        pendingCapture = null;
        resolve(compositeCanvases(glCanvas, overlayCanvas));
      }
    }
    requestAnimationFrame(frame);
  }

  function captureFrame(): Promise<string> {
    return new Promise((resolve) => {
      pendingCapture = resolve;
      requestRender();
    });
  }

  async function saveSnapshot(): Promise<void> {
    const url = await captureFrame();
    const link = document.createElement('a');
    const name = store.get().caseName || 'case';
    link.download = `${name}-${describeLayout(store.get())}.png`;
    link.href = url;
    link.click();
  }

  // --- loading ----------------------------------------------------------

  const loader = new CaseLoader({
    onProgress(stage, fraction, detail) {
      store.set({ status: 'loading', progress: { stage, fraction, detail } });
      updateProgressUi(stage, fraction, detail);
    },

    onVolume(volume: Volume, normalized: Float32Array, ms: number) {
      // The upload is the one step that can fail on the GPU rather than in the
      // parser, typically a volume larger than the driver's 3D texture limit.
      // It has to surface as the error it is; letting it escape the worker
      // callback left the app stuck on the progress overlay with the real
      // reason only in the console.
      try {
        renderer.setVolume(volume, normalized);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        store.set({ status: 'error', error: message, progress: null });
        showError(message);
        hideProgress();
        return;
      }
      camera.frame(volume.extent);
      camera.reset();

      const centre: [number, number, number] = [
        Math.floor(volume.dims[0] / 2),
        Math.floor(volume.dims[1] / 2),
        Math.floor(volume.dims[2] / 2),
      ];

      // Prefer a standard abdominal window, but fall back to something derived
      // from the data if the volume is not a CT at all (an MR series, or a
      // mask opened on its own, would be invisible in a HU window).
      const looksLikeCt = volume.min < -500 && volume.max > 200;
      const preset = WINDOW_PRESETS.find((p) => p.id === 'soft-tissue');
      const wl = looksLikeCt && preset
        ? { level: preset.level, window: preset.window }
        : autoWindow(volume.min, volume.max);

      store.set({
        status: 'ready',
        volume,
        crosshair: centre,
        windowLevel: wl,
        presetId: looksLikeCt && preset ? preset.id : null,
        progress: null,
        error: null,
        timings: { ...store.get().timings, downloadMs: ms },
      });
      hideSplash();
      requestRender();
    },

    onLabels(labels: LabelVolume, ms: number) {
      try {
        renderer.setLabels(labels);
      } catch (err) {
        // The scan is already on screen and usable, so a failed label upload
        // is a warning rather than a dead end.
        showError(`Structures could not be displayed: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }
      const structures = labels.structures;
      renderer.setLut(buildLut(structures));
      const meshStatus = Object.fromEntries(structures.map((s) => [s.index, 'queued' as const]));
      store.set({
        labels,
        structures,
        meshStatus,
        progress: null,
        // The mask progress messages flipped status back to 'loading' after
        // the scan arrived, and nothing set it back, so the app read as
        // permanently loading once a case finished.
        status: 'ready',
        timings: { ...store.get().timings, labelsMs: ms },
      });
      for (const s of structures) {
        renderer.setMeshAppearance(s.index, s.color, s.visible);
      }
      loader.buildMeshes(labels, structures);
      requestRender();
    },

    onMesh(mesh: Mesh) {
      const s = store.get().structures.find((x) => x.index === mesh.index);
      renderer.setMesh(mesh.index, mesh, s?.color ?? [200, 200, 200], s?.visible ?? true);
      store.setMeshStatus(mesh.index, mesh.triangleCount > 0 ? 'ready' : 'none');
      requestRender();
    },

    onMeshError(index, message) {
      console.warn(`Surface extraction failed for structure ${index}: ${message}`);
      if (index >= 0) store.setMeshStatus(index, 'error');
    },

    onAllMeshes(totalMs) {
      store.set({ timings: { ...store.get().timings, meshMs: totalMs } });
      requestRender();
    },

    onError(message) {
      store.set({ status: 'error', error: message, progress: null });
      showError(message);
      hideProgress();
    },
  });

  function loadCase(sources: CaseSources): void {
    renderer.clearMeshes();
    renderer.setLabels(null);
    store.set({
      status: 'loading',
      caseName: sources.name,
      volume: null,
      labels: null,
      structures: [],
      meshStatus: {},
      measurements: [],
      pendingMeasurement: null,
      error: null,
      timings: { downloadMs: 0, labelsMs: 0, meshMs: 0 },
      views: {
        axial: { zoom: 1, pan: [0, 0] },
        coronal: { zoom: 1, pan: [0, 0] },
        sagittal: { zoom: 1, pan: [0, 0] },
      },
    });
    hideError();
    showProgress();
    loader.load(sources);
    requestRender();
  }

  // --- actions ----------------------------------------------------------

  function applyPreset(idOrHotkey: string): void {
    const preset = WINDOW_PRESETS.find((p) => p.id === idOrHotkey || p.hotkey === idOrHotkey);
    if (!preset) return;
    store.set({ windowLevel: { level: preset.level, window: preset.window }, presetId: preset.id });
    requestRender();
  }

  function resetView(): void {
    const volume = store.get().volume;
    if (volume) {
      camera.frame(volume.extent);
      camera.reset();
    }
    store.set({
      views: {
        axial: { zoom: 1, pan: [0, 0] },
        coronal: { zoom: 1, pan: [0, 0] },
        sagittal: { zoom: 1, pan: [0, 0] },
      },
    });
    requestRender();
  }

  function focusStructure(index: number): void {
    const { structures, volume } = store.get();
    const s = structures.find((x) => x.index === index);
    if (!s || !volume || s.voxelCount === 0) return;
    store.set({
      crosshair: [
        Math.round(s.centroid[0]),
        Math.round(s.centroid[1]),
        Math.round(s.centroid[2]),
      ],
    });
    requestRender();
  }

  const deps: PanelDeps = {
    store,
    camera,
    requestRender,
    loadSample: () => loadCase(sampleCase(import.meta.env.BASE_URL)),
    loadFiles: (files) => {
      const sources = sourcesFromFiles(files);
      if (!sources) {
        showError('No NIfTI files found. Drop .nii or .nii.gz files, or a folder containing them.');
        return;
      }
      loadCase(sources);
    },
    applyPreset,
    resetView,
    focusStructure,
    saveSnapshot,
  };

  // Debugging handle for the dev server only. Vite drops the whole branch from
  // the production bundle, so nothing is exposed on the deployed page.
  if (import.meta.env.DEV) {
    (window as unknown as Record<string, unknown>).__viewer = {
      store, camera, renderer, captureFrame, requestRender,
    };
  }

  // --- UI ---------------------------------------------------------------

  const toolbarUpdaters: Array<(s: AppState) => void> = [];
  buildToolbar($('toolbar'), deps, toolbarUpdaters);
  const panels = buildSidebar($('sidebar'), deps);

  new Interactions(overlayCanvas, {
    store, renderer, camera, requestRender,
    onPreset: applyPreset,
    onResetView: resetView,
  });

  $('splash-load').addEventListener('click', () => deps.loadSample());

  // Keep the GPU-side lookup table and the per-surface colours in step with
  // the structure list. Doing it from a subscription rather than at every call
  // site means a visibility toggle can never update the list but not the view.
  store.subscribe((state, changed) => {
    if (changed.has('structures')) {
      renderer.setLut(buildLut(state.structures));
      for (const s of state.structures) renderer.setMeshAppearance(s.index, s.color, s.visible);
      requestRender();
    }
    panels.update(state);
    for (const u of toolbarUpdaters) u(state);
    if (changed.has('progress')) {
      const p = state.progress;
      if (p) updateProgressUi(p.stage, p.fraction, p.detail);
      else hideProgress();
    }
  });

  panels.update(store.get());
  for (const u of toolbarUpdaters) u(store.get());

  const resizeObserver = new ResizeObserver(() => requestRender());
  resizeObserver.observe($('viewport'));
  window.addEventListener('resize', requestRender);

  setupDragAndDrop(deps);
  requestAnimationFrame(frame);

  // Auto-load the sample so a reviewer sees anatomy without clicking anything.
  // A demo that opens on an empty grey box asks the visitor to trust that the
  // rest works.
  deps.loadSample();
}

// ---------------------------------------------------------------------------

function toRenderState(state: AppState, camera: OrbitCamera): RenderState {
  return {
    layout: state.layout,
    crosshair: state.crosshair,
    windowLevel: state.windowLevel,
    showLabels: state.showLabels,
    labelOpacity: state.labelOpacity,
    labelOutline: state.labelOutline,
    outlineWidth: state.outlineWidth,
    views: state.views,
    show3DVolume: state.show3DVolume,
    volumeMode: state.volumeMode,
    volumeDensity: state.volumeDensity,
    volumeShade: state.volumeShade,
    volumeQuality: state.volumeQuality,
    volumeLabelBoost: state.volumeLabelBoost,
    showMeshes: state.showMeshes,
    meshOpacity: state.meshOpacity,
    showSlicesIn3D: state.showSlicesIn3D,
    showBoundingBox: state.showBoundingBox,
    camera,
    activeView: state.hoverView,
  };
}

function updateStatusBar(state: AppState, renderer: Renderer): void {
  const probe = document.getElementById('probe');
  const right = document.getElementById('status-right');
  if (probe) probe.textContent = describeProbe(state);
  if (right) {
    const bits: string[] = [];
    if (state.volume) {
      bits.push(`${state.volume.dims.join(' x ')}`);
      bits.push(`${state.volume.spacing.map((s) => s.toFixed(2)).join(' x ')} mm`);
    }
    if (renderer.meshCount > 0) bits.push(`${renderer.meshCount} surfaces`);
    const ms = renderer.frameTimeMs;
    if (ms > 0) bits.push(`${ms.toFixed(1)} ms/frame`);
    right.textContent = bits.join('   |   ');
  }
}

function updateProgressUi(stage: string, fraction: number | null, detail?: string): void {
  const root = document.getElementById('progress');
  const stageEl = document.getElementById('progress-stage');
  const bar = document.getElementById('progress-bar');
  const detailEl = document.getElementById('progress-detail');
  if (!root || !stageEl || !bar || !detailEl) return;
  root.hidden = false;
  stageEl.textContent = stage;
  bar.style.width = fraction === null ? '35%' : `${Math.round(fraction * 100)}%`;
  detailEl.textContent = detail ?? '';
}

function showProgress(): void {
  updateProgressUi('Starting', 0);
}

function hideProgress(): void {
  const root = document.getElementById('progress');
  if (root) root.hidden = true;
}

function hideSplash(): void {
  const splash = document.getElementById('splash');
  if (splash) splash.hidden = true;
  hideProgress();
}

function showError(message: string): void {
  const el = document.getElementById('error');
  if (!el) return;
  el.textContent = message;
  el.hidden = false;
}

function hideError(): void {
  const el = document.getElementById('error');
  if (el) el.hidden = true;
}

function showFatal(message: string): void {
  const splash = document.getElementById('splash');
  if (splash) {
    splash.innerHTML = '';
    const card = document.createElement('div');
    card.className = 'splash-card';
    const h = document.createElement('h2');
    h.textContent = 'This browser cannot run the viewer';
    const p = document.createElement('p');
    p.textContent = message;
    card.append(h, p);
    splash.append(card);
    splash.hidden = false;
  }
}

/**
 * Flatten the GL canvas and the annotation overlay into one image, at full
 * device resolution so the export is not a blurry copy of the screen.
 */
function compositeCanvases(gl: HTMLCanvasElement, overlay: HTMLCanvasElement): string {
  const out = document.createElement('canvas');
  out.width = gl.width;
  out.height = gl.height;
  const ctx = out.getContext('2d');
  if (!ctx) return '';
  ctx.drawImage(gl, 0, 0);
  ctx.drawImage(overlay, 0, 0, out.width, out.height);
  return out.toDataURL('image/png');
}

function describeLayout(state: AppState): string {
  return state.layout === 'fourUp' ? '4up' : state.layout;
}

/**
 * Drag and drop over the whole window, not just the viewport. Dropping onto a
 * 4-pane grid is fiddly, and a folder dropped anywhere clearly means "open
 * this".
 */
function setupDragAndDrop(deps: PanelDeps): void {
  const overlay = document.getElementById('drop-overlay');
  let depth = 0;

  window.addEventListener('dragenter', (e) => {
    e.preventDefault();
    depth++;
    if (overlay) overlay.hidden = false;
  });

  window.addEventListener('dragover', (e) => {
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
  });

  window.addEventListener('dragleave', (e) => {
    e.preventDefault();
    depth = Math.max(0, depth - 1);
    if (depth === 0 && overlay) overlay.hidden = true;
  });

  window.addEventListener('drop', async (e) => {
    e.preventDefault();
    depth = 0;
    if (overlay) overlay.hidden = true;
    if (!e.dataTransfer) return;
    const files = await collectFiles(e.dataTransfer);
    if (files.length) deps.loadFiles(files);
  });
}

/**
 * Walk dropped directory entries so a whole case folder works, not just a
 * multi-select of files. These datasets ship as a folder with the scan at the
 * top and the masks in a subdirectory, which is exactly what people will drag.
 */
async function collectFiles(dt: DataTransfer): Promise<File[]> {
  const entries: FileSystemEntry[] = [];
  for (const item of dt.items) {
    const entry = item.webkitGetAsEntry?.();
    if (entry) entries.push(entry);
  }
  if (entries.length === 0) return [...dt.files];

  const files: File[] = [];
  const walk = async (entry: FileSystemEntry, path: string): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File | null>((resolve) =>
        (entry as FileSystemFileEntry).file((f) => resolve(f), () => resolve(null)),
      );
      if (file) {
        // Preserve the relative path so the loader can tell a scan from a mask
        // by which folder it came out of.
        Object.defineProperty(file, 'webkitRelativePath', { value: `${path}${file.name}` });
        files.push(file);
      }
      return;
    }
    if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      // readEntries returns at most 100 at a time and must be called until it
      // returns an empty batch, which is easy to miss and silently truncates
      // any folder with more structures than that.
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((resolve) =>
          reader.readEntries((r) => resolve(r), () => resolve([])),
        );
        if (batch.length === 0) break;
        for (const child of batch) await walk(child, `${path}${entry.name}/`);
      }
    }
  };

  for (const entry of entries) await walk(entry, '');
  return files;
}

boot();
