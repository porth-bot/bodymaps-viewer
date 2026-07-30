/**
 * The sidebar. Builds the DOM once, then reconciles it from state.
 *
 * The structure list is the only part that rebuilds its children, and only
 * when the structure set itself changes. Everything else updates in place, so
 * dragging a slider never touches layout.
 */

import type { OrbitCamera } from '../camera/orbit';
import { WINDOW_PRESETS } from '../core/presets';
import { describeAxCodes } from '../core/volume';
import { LAYOUT_LABELS, type LayoutMode } from '../gl/layout';
import type { VolumeRenderMode } from '../gl/renderer';
import {
  button, el, formatCount, formatVolume, rgbCss, section, segmented, select, slider, toggle,
} from './controls';
import type { AppState, Store, Tool } from './store';

export interface PanelDeps {
  store: Store;
  camera: OrbitCamera;
  requestRender(): void;
  loadSample(): void;
  loadFiles(files: File[]): void;
  applyPreset(id: string): void;
  resetView(): void;
  focusStructure(index: number): void;
  saveSnapshot(): void;
}

export interface Panels {
  update(state: AppState): void;
}

export function buildSidebar(root: HTMLElement, deps: PanelDeps): Panels {
  const updaters: Array<(s: AppState) => void> = [];

  root.append(buildCasePanel(deps, updaters));
  root.append(buildDisplayPanel(deps, updaters));
  root.append(buildStructuresPanel(deps, updaters));
  root.append(build3DPanel(deps, updaters));
  root.append(buildShortcutsPanel());

  return {
    update(state) {
      for (const u of updaters) u(state);
    },
  };
}

// ---------------------------------------------------------------------------

function buildCasePanel(deps: PanelDeps, updaters: Array<(s: AppState) => void>): HTMLElement {
  const { root, body } = section('Case');

  const actions = el('div', 'row');
  actions.append(button('Load sample case', () => deps.loadSample(), 'primary'));

  const fileInput = el('input');
  fileInput.type = 'file';
  fileInput.multiple = true;
  fileInput.accept = '.nii,.gz';
  fileInput.className = 'hidden-input';
  fileInput.addEventListener('change', () => {
    if (fileInput.files) deps.loadFiles([...fileInput.files]);
  });
  actions.append(button('Open files', () => fileInput.click()));
  body.append(actions, fileInput);

  const hint = el('p', 'hint', 'Or drop a case folder anywhere on the window. Reads .nii and .nii.gz, any orientation.');
  body.append(hint);

  const info = el('dl', 'info');
  body.append(info);

  updaters.push((s) => {
    info.textContent = '';
    if (!s.volume) return;
    const v = s.volume;
    const rows: Array<[string, string]> = [
      ['Case', s.caseName || 'unnamed'],
      ['Dimensions', `${v.dims[0]} x ${v.dims[1]} x ${v.dims[2]}`],
      ['Voxel size', `${v.spacing.map((x) => x.toFixed(3)).join(' x ')} mm`],
      ['Field of view', `${v.extent.map((x) => Math.round(x)).join(' x ')} mm`],
      ['Value range', `${Math.round(v.min)} to ${Math.round(v.max)} HU`],
      // "Reoriented", not "resampled": the transform is a permutation and a
      // set of flips, so voxel values are moved but never interpolated.
      ['Stored as', describeAxCodes(v.originalAxCodes) === 'RAS'
        ? 'RAS (native)'
        : `${describeAxCodes(v.originalAxCodes)}, reoriented to RAS`],
      ['Voxels', formatCount(v.dims[0] * v.dims[1] * v.dims[2])],
    ];
    if (v.volumeCount > 1) {
      // A 4D series loads its first frame. Saying so beats letting a reader
      // wonder where the other frames went.
      rows.push(['4D series', `frame 1 of ${v.volumeCount}`]);
    }
    if (s.labels) {
      rows.push(['Structures', String(s.labels.structures.length)]);
      if (s.labels.overlapVoxels > 0) {
        rows.push(['Overlapping voxels', formatCount(s.labels.overlapVoxels)]);
      }
    }
    if (s.timings.downloadMs > 0) {
      const parts = [`scan ${(s.timings.downloadMs / 1000).toFixed(1)}s`];
      if (s.timings.labelsMs > 0) parts.push(`labels ${(s.timings.labelsMs / 1000).toFixed(1)}s`);
      if (s.timings.meshMs > 0) parts.push(`surfaces ${(s.timings.meshMs / 1000).toFixed(1)}s`);
      rows.push(['Load time', parts.join(', ')]);
    }
    for (const [k, val] of rows) {
      info.append(el('dt', undefined, k), el('dd', undefined, val));
    }
  });

  return root;
}

// ---------------------------------------------------------------------------

function buildDisplayPanel(deps: PanelDeps, updaters: Array<(s: AppState) => void>): HTMLElement {
  const { store } = deps;
  const { root, body } = section('Windowing');

  const presetSelect = select({
    label: 'Preset',
    options: [
      { value: '', label: 'Custom' },
      ...WINDOW_PRESETS.map((p) => ({
        value: p.id,
        label: p.hotkey ? `${p.name}  (${p.hotkey})` : p.name,
      })),
    ],
    value: 'abdomen',
    onChange: (id) => {
      if (id) deps.applyPreset(id);
    },
  });

  const levelSlider = slider({
    label: 'Level',
    min: -1000, max: 1500, step: 1, value: 40,
    format: (v) => `${Math.round(v)} HU`,
    onInput: (v) => {
      store.set((s) => ({ windowLevel: { ...s.windowLevel, level: v }, presetId: null }));
      deps.requestRender();
    },
  });

  const windowSlider = slider({
    label: 'Width',
    min: 1, max: 4000, step: 1, value: 400,
    format: (v) => `${Math.round(v)} HU`,
    onInput: (v) => {
      store.set((s) => ({ windowLevel: { ...s.windowLevel, window: v }, presetId: null }));
      deps.requestRender();
    },
  });

  const histogram = el('canvas', 'histogram');
  histogram.width = 512;
  histogram.height = 96;

  body.append(presetSelect.root, histogram, levelSlider.root, windowSlider.root);
  body.append(el('p', 'hint', 'Right drag on any slice adjusts window and level directly.'));

  const overlayHeader = el('div', 'control-label', 'Segmentation overlay');
  const showLabels = toggle({
    label: 'Show structures',
    value: true,
    onChange: (v) => { store.set({ showLabels: v }); deps.requestRender(); },
  });
  const outline = toggle({
    label: 'Outline only',
    value: false,
    hint: 'Draw structure borders instead of filled regions, so the underlying anatomy stays visible.',
    onChange: (v) => { store.set({ labelOutline: v }); deps.requestRender(); },
  });
  const opacity = slider({
    label: 'Overlay opacity',
    min: 0, max: 1, step: 0.01, value: 0.45,
    format: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { store.set({ labelOpacity: v }); deps.requestRender(); },
  });

  body.append(el('hr', 'divider'), overlayHeader, showLabels.root, outline.root, opacity.root);

  let histogramDrawnFor: unknown = null;
  updaters.push((s) => {
    presetSelect.update((s.presetId ?? '') as never);
    levelSlider.update(s.windowLevel.level);
    windowSlider.update(s.windowLevel.window);
    showLabels.update(s.showLabels);
    outline.update(s.labelOutline);
    opacity.update(s.labelOpacity);

    if (s.volume && histogramDrawnFor !== s.volume) {
      histogramDrawnFor = s.volume;
      drawHistogram(histogram, s);
    }
    if (s.volume) drawWindowOverlay(histogram, s);
  });

  return root;
}

/**
 * Voxel histogram behind the window controls.
 *
 * Plotted on a log scale because air occupies more than half of any abdominal
 * CT: on a linear scale the entire soft-tissue range, which is the part anyone
 * is windowing on, would be an invisible line along the axis.
 */
let histogramCache: { counts: Uint32Array; min: number; max: number } | null = null;

function drawHistogram(canvas: HTMLCanvasElement, state: AppState): void {
  const ctx = canvas.getContext('2d');
  if (!ctx || !state.volume) return;
  const bins = 256;

  const counts = new Uint32Array(bins);
  const { values, slope, intercept, min, max } = state.volume;
  const range = max - min;
  if (range <= 0) return;
  const scale = (bins - 1) / range;
  const stride = Math.max(1, Math.floor(values.length / 1_500_000));
  for (let i = 0; i < values.length; i += stride) {
    const v = (values[i] * slope + intercept - min) * scale;
    counts[v < 0 ? 0 : v > bins - 1 ? bins - 1 : v | 0]++;
  }
  histogramCache = { counts, min, max };
  canvas.dataset.ready = '1';
}

function drawWindowOverlay(canvas: HTMLCanvasElement, state: AppState): void {
  const ctx = canvas.getContext('2d');
  if (!ctx || !histogramCache || !state.volume) return;
  const { counts, min, max } = histogramCache;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  let peak = 0;
  for (const c of counts) peak = Math.max(peak, c);
  if (peak === 0) return;
  const logPeak = Math.log1p(peak);

  ctx.fillStyle = 'rgba(120, 170, 230, 0.35)';
  const bw = W / counts.length;
  for (let i = 0; i < counts.length; i++) {
    const h = (Math.log1p(counts[i]) / logPeak) * (H - 2);
    ctx.fillRect(i * bw, H - h, Math.ceil(bw), h);
  }

  // Shade the region the current window actually maps to visible grey.
  const lo = state.windowLevel.level - state.windowLevel.window / 2;
  const hi = state.windowLevel.level + state.windowLevel.window / 2;
  const toX = (hu: number) => ((hu - min) / (max - min)) * W;
  const x0 = Math.max(0, toX(lo));
  const x1 = Math.min(W, toX(hi));

  const grad = ctx.createLinearGradient(x0, 0, x1, 0);
  grad.addColorStop(0, 'rgba(255, 255, 255, 0.06)');
  grad.addColorStop(1, 'rgba(255, 255, 255, 0.22)');
  ctx.fillStyle = grad;
  ctx.fillRect(x0, 0, Math.max(x1 - x0, 1), H);

  ctx.strokeStyle = 'rgba(255, 214, 92, 0.9)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0 + 0.5, 0); ctx.lineTo(x0 + 0.5, H);
  ctx.moveTo(x1 - 0.5, 0); ctx.lineTo(x1 - 0.5, H);
  ctx.stroke();

  ctx.fillStyle = 'rgba(200, 215, 235, 0.75)';
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillText(`${Math.round(min)}`, 3, 3);
  ctx.textAlign = 'right';
  ctx.fillText(`${Math.round(max)} HU`, W - 3, 3);
}

// ---------------------------------------------------------------------------

function buildStructuresPanel(deps: PanelDeps, updaters: Array<(s: AppState) => void>): HTMLElement {
  const { store } = deps;
  const { root, body } = section('Structures');

  const bulk = el('div', 'row');
  bulk.append(
    button('All', () => {
      store.set((s) => ({ structures: s.structures.map((x) => ({ ...x, visible: true })) }));
      deps.requestRender();
    }),
    button('None', () => {
      store.set((s) => ({ structures: s.structures.map((x) => ({ ...x, visible: false })) }));
      deps.requestRender();
    }),
    button('Invert', () => {
      store.set((s) => ({ structures: s.structures.map((x) => ({ ...x, visible: !x.visible })) }));
      deps.requestRender();
    }),
  );
  body.append(bulk);

  const list = el('ul', 'structure-list');
  body.append(list);

  const empty = el('p', 'hint', 'No structures loaded.');
  body.append(empty);

  let signature = '';
  const rows = new Map<number, { li: HTMLElement; eye: HTMLElement; status: HTMLElement }>();

  updaters.push((s) => {
    const sig = s.structures.map((x) => `${x.index}:${x.key}`).join('|');
    empty.style.display = s.structures.length ? 'none' : '';

    if (sig !== signature) {
      signature = sig;
      list.textContent = '';
      rows.clear();

      for (const st of s.structures) {
        const li = el('li', 'structure');

        const swatch = el('span', 'swatch');
        swatch.style.background = rgbCss(st.color);

        const nameWrap = el('div', 'structure-main');
        const name = el('span', 'structure-name', st.name);
        const meta = el('span', 'structure-meta',
          `${formatVolume(st.volumeMl)}   ${Math.round(st.meanHu)} HU`);
        nameWrap.append(name, meta);

        const status = el('span', 'structure-status');

        const eye = el('button', 'eye');
        eye.type = 'button';
        eye.title = 'Show or hide this structure';
        eye.addEventListener('click', (ev) => {
          ev.stopPropagation();
          const cur = store.get().structures.find((x) => x.index === st.index);
          store.updateStructure(st.index, { visible: !(cur?.visible ?? true) });
          deps.requestRender();
        });

        li.title = `${st.name}\n${formatCount(st.voxelCount)} voxels\nmean ${Math.round(st.meanHu)} HU\nClick to centre the views on it`;
        li.addEventListener('click', () => deps.focusStructure(st.index));

        li.append(eye, swatch, nameWrap, status);
        list.append(li);
        rows.set(st.index, { li, eye, status });
      }
    }

    for (const st of s.structures) {
      const row = rows.get(st.index);
      if (!row) continue;
      row.li.classList.toggle('hidden-structure', !st.visible);
      row.eye.textContent = st.visible ? '◉' : '○';
      const ms = s.meshStatus[st.index] ?? 'none';
      row.status.textContent =
        ms === 'ready' ? '3D' : ms === 'building' ? '...' : ms === 'queued' ? '·' : ms === 'error' ? '!' : '';
      row.status.className = `structure-status status-${ms}`;
    }
  });

  return root;
}

// ---------------------------------------------------------------------------

function build3DPanel(deps: PanelDeps, updaters: Array<(s: AppState) => void>): HTMLElement {
  const { store } = deps;
  const { root, body } = section('3D view');

  const views = el('div', 'row wrap');
  const viewButtons: Array<[string, 'anterior' | 'posterior' | 'left' | 'right' | 'superior' | 'inferior']> = [
    ['Ant', 'anterior'], ['Post', 'posterior'], ['Left', 'left'],
    ['Right', 'right'], ['Sup', 'superior'], ['Inf', 'inferior'],
  ];
  for (const [label, view] of viewButtons) {
    views.append(button(label, () => {
      deps.camera.setStandardView(view);
      deps.requestRender();
    }, 'small'));
  }
  body.append(el('div', 'control-label', 'Standard views'), views);

  const showVolume = toggle({
    label: 'Volume rendering',
    value: true,
    onChange: (v) => { store.set({ show3DVolume: v }); deps.requestRender(); },
  });

  const mode = segmented<VolumeRenderMode>({
    label: 'Mode',
    options: [
      { value: 'composite', label: 'Composite', title: 'Front to back alpha compositing with a tissue transfer function' },
      { value: 'mip', label: 'MIP', title: 'Maximum intensity projection, useful for contrast-filled vessels' },
    ],
    value: 'composite',
    onChange: (v) => { store.set({ volumeMode: v }); deps.requestRender(); },
  });

  const density = slider({
    label: 'Density',
    min: 0.05, max: 2, step: 0.01, value: 0.55,
    format: (v) => v.toFixed(2),
    onInput: (v) => { store.set({ volumeDensity: v }); deps.requestRender(); },
  });

  const quality = slider({
    label: 'Quality',
    min: 0.25, max: 2, step: 0.05, value: 1,
    format: (v) => `${v.toFixed(2)}x`,
    onInput: (v) => { store.set({ volumeQuality: v }); deps.requestRender(); },
  });

  const shade = toggle({
    label: 'Gradient shading',
    value: true,
    hint: 'Light the volume using the local intensity gradient as a surface normal.',
    onChange: (v) => { store.set({ volumeShade: v }); deps.requestRender(); },
  });

  const labelBoost = slider({
    label: 'Structure emphasis',
    min: 0, max: 1, step: 0.01, value: 0.5,
    format: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { store.set({ volumeLabelBoost: v }); deps.requestRender(); },
  });

  body.append(showVolume.root, mode.root, density.root, quality.root, shade.root, labelBoost.root);

  const showMeshes = toggle({
    label: 'Structure surfaces',
    value: true,
    onChange: (v) => { store.set({ showMeshes: v }); deps.requestRender(); },
  });
  const meshOpacity = slider({
    label: 'Surface opacity',
    min: 0.05, max: 1, step: 0.01, value: 1,
    format: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { store.set({ meshOpacity: v }); deps.requestRender(); },
  });
  const slices3D = toggle({
    label: 'Slice planes in 3D',
    value: true,
    onChange: (v) => { store.set({ showSlicesIn3D: v }); deps.requestRender(); },
  });
  const bbox = toggle({
    label: 'Bounding box',
    value: false,
    onChange: (v) => { store.set({ showBoundingBox: v }); deps.requestRender(); },
  });

  body.append(el('hr', 'divider'), showMeshes.root, meshOpacity.root, slices3D.root, bbox.root);

  updaters.push((s) => {
    showVolume.update(s.show3DVolume);
    mode.update(s.volumeMode);
    density.update(s.volumeDensity);
    quality.update(s.volumeQuality);
    shade.update(s.volumeShade);
    labelBoost.update(s.volumeLabelBoost);
    showMeshes.update(s.showMeshes);
    meshOpacity.update(s.meshOpacity);
    slices3D.update(s.showSlicesIn3D);
    bbox.update(s.showBoundingBox);
  });

  return root;
}

// ---------------------------------------------------------------------------

function buildShortcutsPanel(): HTMLElement {
  const { root, body } = section('Controls', { collapsed: true });
  const rows: Array<[string, string]> = [
    ['Left drag', 'Move the crosshair (orbit in 3D)'],
    ['Right drag', 'Window and level'],
    ['Middle drag', 'Pan'],
    ['Wheel', 'Scroll slices (dolly in 3D)'],
    ['Ctrl + wheel', 'Zoom about the pointer'],
    ['Double click', 'Maximise or restore a pane'],
    ['1 to 9', 'Window presets'],
    ['Arrows', 'Step slice, Shift for 10'],
    ['A C S V F G', 'Axial, coronal, sagittal, 3D, four up, row'],
    ['L / O', 'Toggle structures / outline mode'],
    ['M / X / P', 'Surfaces / volume rendering / slice planes'],
    ['H', 'Toggle crosshair'],
    ['R', 'Reset views'],
    ['Esc', 'Clear measurements'],
  ];
  const dl = el('dl', 'info shortcuts');
  for (const [k, v] of rows) dl.append(el('dt', undefined, k), el('dd', undefined, v));
  body.append(dl);
  return root;
}

// ---------------------------------------------------------------------------

export function buildToolbar(
  root: HTMLElement,
  deps: PanelDeps,
  updaters: Array<(s: AppState) => void>,
): void {
  const { store } = deps;

  const layout = segmented<LayoutMode>({
    options: (['fourUp', 'axial', 'coronal', 'sagittal', 'volume', 'row'] as LayoutMode[]).map((m) => ({
      value: m,
      label: m === 'fourUp' ? '⊞' : m === 'row' ? '≡' : m === 'volume' ? '3D' : m[0].toUpperCase(),
      title: LAYOUT_LABELS[m],
    })),
    value: 'fourUp',
    onChange: (v) => { store.set({ layout: v }); deps.requestRender(); },
  });
  layout.root.classList.add('toolbar-group');

  const tool = segmented<Tool>({
    options: [
      { value: 'navigate', label: 'Navigate', title: 'Left drag moves the crosshair' },
      { value: 'window', label: 'Window', title: 'Left drag adjusts window and level' },
      { value: 'zoom', label: 'Zoom', title: 'Left drag zooms' },
      { value: 'pan', label: 'Pan', title: 'Left drag pans' },
      { value: 'ruler', label: 'Ruler', title: 'Left drag measures a distance in millimetres' },
    ],
    value: 'navigate',
    onChange: (v) => { store.set({ tool: v }); deps.requestRender(); },
  });
  tool.root.classList.add('toolbar-group');

  const reset = button('Reset', () => deps.resetView(), 'small');
  const snapshot = button('Snapshot', () => deps.saveSnapshot(), 'small');
  snapshot.title = 'Save the current viewports as a PNG, at full device resolution';

  root.append(layout.root, tool.root, reset, snapshot);

  updaters.push((s) => {
    layout.update(s.layout);
    tool.update(s.tool);
  });
}
