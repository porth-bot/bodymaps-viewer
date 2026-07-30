/**
 * Everything drawn on top of the GL canvas: crosshairs, orientation letters,
 * the scale bar, measurements and the corner readouts.
 *
 * A separate 2D canvas rather than more GL work, because text is the bulk of
 * it and Canvas2D renders text with proper hinting and subpixel positioning
 * that a texture-atlas font in GL would only approximate.
 */

import type { Structure } from '../core/types';
import { formatPatientPosition, labelAtVoxel, sampleVoxel, voxelToWorld } from '../core/volume';
import type { OrbitCamera } from '../camera/orbit';
import type { Renderer } from '../gl/renderer';
import { PLANES, planeUvToScreen, voxelToPlaneUv, type PlaneSpec } from '../gl/planes';
import type { ViewportRect } from '../gl/layout';
import type { AppState } from './store';

const FONT = '11px ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace';
const FONT_BOLD = 'bold 13px ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace';

const COLORS = {
  text: 'rgba(226, 235, 245, 0.92)',
  textDim: 'rgba(160, 178, 198, 0.72)',
  orientation: 'rgba(120, 200, 255, 0.85)',
  crosshair: 'rgba(255, 214, 92, 0.85)',
  crosshairSoft: 'rgba(255, 214, 92, 0.28)',
  ruler: 'rgba(120, 255, 190, 0.95)',
  activeBorder: 'rgba(92, 170, 255, 0.9)',
  border: 'rgba(255, 255, 255, 0.07)',
  panel: 'rgba(8, 12, 20, 0.72)',
};

/** Per-view accent, matching the slice-plane colours used by most viewers. */
const VIEW_ACCENT: Record<string, string> = {
  axial: 'rgba(255, 196, 84, 0.95)',
  coronal: 'rgba(120, 220, 130, 0.95)',
  sagittal: 'rgba(240, 130, 150, 0.95)',
};

export class Overlay {
  private ctx: CanvasRenderingContext2D;

  constructor(private canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not create the 2D overlay context');
    this.ctx = ctx;
  }

  private resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    // Draw in CSS pixels; the transform handles the device pixel ratio so text
    // stays crisp on retina without every coordinate being multiplied by hand.
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  draw(state: AppState, renderer: Renderer, camera: OrbitCamera): void {
    this.resize();
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.clientWidth, this.canvas.clientHeight);
    if (!state.volume) return;

    for (const rect of renderer.viewportRects) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(rect.x, rect.y, rect.width, rect.height);
      ctx.clip();
      ctx.translate(rect.x, rect.y);

      if (rect.view === 'volume') this.draw3D(state, renderer, rect, camera);
      else this.drawSlice(state, renderer, rect);

      ctx.restore();

      ctx.strokeStyle = state.hoverView === rect.view ? COLORS.activeBorder : COLORS.border;
      ctx.lineWidth = state.hoverView === rect.view ? 1.5 : 1;
      ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.width - 1, rect.height - 1);
    }
  }

  private drawSlice(state: AppState, renderer: Renderer, rect: ViewportRect): void {
    const ctx = this.ctx;
    const view = rect.view as 'axial' | 'coronal' | 'sagittal';
    const spec: PlaneSpec = PLANES[view];
    const volume = state.volume!;
    const t = renderer.viewTransformFor(rect, state.views);
    if (!t) return;

    const W = rect.width;
    const H = rect.height;

    if (state.showOrientationLabels) {
      ctx.font = FONT_BOLD;
      ctx.fillStyle = COLORS.orientation;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(spec.labels.left, 8, H / 2);
      ctx.textAlign = 'right';
      ctx.fillText(spec.labels.right, W - 8, H / 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(spec.labels.top, W / 2, 8);
      ctx.textBaseline = 'bottom';
      ctx.fillText(spec.labels.bottom, W / 2, H - 8);
    }

    if (state.showCrosshair) {
      const [u, v] = voxelToPlaneUv(spec, volume.dims, state.crosshair);
      const [cx, cy] = planeUvToScreen(t, u, v);
      const gap = 10;
      ctx.strokeStyle = COLORS.crosshairSoft;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, cy); ctx.lineTo(cx - gap, cy);
      ctx.moveTo(cx + gap, cy); ctx.lineTo(W, cy);
      ctx.moveTo(cx, 0); ctx.lineTo(cx, cy - gap);
      ctx.moveTo(cx, cy + gap); ctx.lineTo(cx, H);
      ctx.stroke();

      ctx.strokeStyle = COLORS.crosshair;
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.stroke();
    }

    this.drawMeasurements(state, t, view, spec);
    this.drawScaleBar(t, W, H);

    // Corner readouts, laid out like a clinical viewer: identity top-left,
    // geometry bottom-left, window/level bottom-right.
    const sliceIndex = Math.round(state.crosshair[spec.sliceAxis]);
    const nSlices = volume.dims[spec.sliceAxis];
    ctx.font = FONT;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = VIEW_ACCENT[view] ?? COLORS.text;
    ctx.fillText(spec.sliceName.toUpperCase(), 8, 8);
    ctx.fillStyle = COLORS.textDim;
    ctx.fillText(`${sliceIndex + 1} / ${nSlices}`, 8, 22);

    ctx.textBaseline = 'bottom';
    ctx.fillStyle = COLORS.textDim;
    const mmPerVoxel = volume.spacing[spec.sliceAxis];
    ctx.fillText(`${mmPerVoxel.toFixed(2)} mm slice`, 8, H - 8);

    ctx.textAlign = 'right';
    ctx.fillText(
      `W ${Math.round(state.windowLevel.window)}  L ${Math.round(state.windowLevel.level)}`,
      W - 8, H - 8,
    );
    if (t.pixelsPerMm > 0) {
      ctx.fillText(`${(state.views[view]?.zoom ?? 1).toFixed(2)}x`, W - 8, H - 22);
    }
  }

  private drawMeasurements(
    state: AppState,
    t: ReturnType<Renderer['viewTransformFor']>,
    view: 'axial' | 'coronal' | 'sagittal',
    spec: PlaneSpec,
  ): void {
    if (!t || !state.volume) return;
    const ctx = this.ctx;
    const dims = state.volume.dims;
    const current = Math.round(state.crosshair[spec.sliceAxis]);

    const all = state.pendingMeasurement
      ? [...state.measurements, state.pendingMeasurement]
      : state.measurements;

    for (const m of all) {
      if (m.view !== view) continue;
      // Only show a measurement on the slice it was drawn on. Showing them
      // everywhere turns a busy study into spaghetti.
      if (Math.round(m.slice) !== current) continue;

      const [ua, va] = voxelToPlaneUv(spec, dims, m.a);
      const [ub, vb] = voxelToPlaneUv(spec, dims, m.b);
      const [ax, ay] = planeUvToScreen(t, ua, va);
      const [bx, by] = planeUvToScreen(t, ub, vb);

      ctx.strokeStyle = COLORS.ruler;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();

      for (const [px, py] of [[ax, ay], [bx, by]]) {
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.stroke();
      }

      const label = `${m.lengthMm.toFixed(1)} mm`;
      ctx.font = FONT;
      const mx = (ax + bx) / 2;
      const my = (ay + by) / 2;
      const w = ctx.measureText(label).width;
      ctx.fillStyle = COLORS.panel;
      ctx.fillRect(mx + 6, my - 16, w + 8, 15);
      ctx.fillStyle = COLORS.ruler;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(label, mx + 10, my - 14);
    }
  }

  /**
   * Scale bar with a 1/2/5 step. A bar is more honest than printing a zoom
   * percentage, because it stays correct no matter how the browser scales the
   * canvas.
   */
  private drawScaleBar(
    t: ReturnType<Renderer['viewTransformFor']>,
    W: number,
    H: number,
  ): void {
    if (!t || t.pixelsPerMm <= 0) return;
    const ctx = this.ctx;
    const targetPx = Math.min(120, W * 0.28);
    const rawMm = targetPx / t.pixelsPerMm;
    const pow = Math.pow(10, Math.floor(Math.log10(Math.max(rawMm, 1e-6))));
    const norm = rawMm / pow;
    const nice = norm >= 5 ? 5 : norm >= 2 ? 2 : 1;
    const mm = nice * pow;
    const px = mm * t.pixelsPerMm;
    if (!isFinite(px) || px < 8) return;

    const x1 = W - 16;
    const x0 = x1 - px;
    const y = H - 34;

    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x0, y); ctx.lineTo(x1, y);
    ctx.moveTo(x0, y - 4); ctx.lineTo(x0, y + 4);
    ctx.moveTo(x1, y - 4); ctx.lineTo(x1, y + 4);
    ctx.stroke();

    ctx.font = FONT;
    ctx.fillStyle = COLORS.text;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(mm >= 10 ? `${(mm / 10).toFixed(mm % 10 === 0 ? 0 : 1)} cm` : `${mm} mm`, (x0 + x1) / 2, y - 6);
  }

  private draw3D(state: AppState, renderer: Renderer, rect: ViewportRect, camera: OrbitCamera): void {
    const ctx = this.ctx;
    ctx.font = FONT;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = COLORS.text;
    ctx.fillText('3D', 8, 8);

    ctx.fillStyle = COLORS.textDim;
    const bits: string[] = [];
    if (state.show3DVolume) bits.push(state.volumeMode === 'mip' ? 'MIP' : 'volume');
    if (state.showMeshes && renderer.meshCount > 0) bits.push(`${renderer.meshCount} surfaces`);
    if (state.showSlicesIn3D) bits.push('slices');
    ctx.fillText(bits.join(' + ') || 'empty', 8, 22);

    const ms = renderer.cpuFrameMs;
    if (ms > 0) {
      ctx.textAlign = 'right';
      ctx.fillText(`${ms.toFixed(1)} ms CPU`, rect.width - 8, 8);
    }

    this.drawAxisGizmo(camera, rect);
  }

  /**
   * Small R/A/S axis indicator. It is the fastest way to answer "which side am
   * I looking at" in a 3D view, and it stays readable where an orientation
   * cube would need real geometry.
   */
  private drawAxisGizmo(camera: OrbitCamera, rect: ViewportRect): void {
    const ctx = this.ctx;
    const cx = rect.width - 46;
    const cy = rect.height - 46;
    const r = 26;

    const az = camera.azimuth;
    const el = camera.elevation;

    // Project each anatomical axis into the camera's screen basis. This mirrors
    // the lookAt the renderer uses, so the gizmo can never disagree with the
    // scene it labels.
    //
    // `right` is cross(forward, worldUp) normalised, which for worldUp = +S
    // reduces to (-cos az, sin az, 0). Getting that sign backwards mirrors the
    // gizmo, and a left/right indicator that lies is worse than none at all.
    const ce = Math.cos(el), se = Math.sin(el);
    const sa = Math.sin(az), ca = Math.cos(az);
    const forward: [number, number, number] = [-sa * ce, -ca * ce, -se];
    const right: [number, number, number] = [-ca, sa, 0];
    const up: [number, number, number] = [
      right[1] * forward[2] - right[2] * forward[1],
      right[2] * forward[0] - right[0] * forward[2],
      right[0] * forward[1] - right[1] * forward[0],
    ];

    const axes: Array<{ v: [number, number, number]; pos: string; neg: string; color: string }> = [
      { v: [1, 0, 0], pos: 'R', neg: 'L', color: 'rgba(255,120,130,0.95)' },
      { v: [0, 1, 0], pos: 'A', neg: 'P', color: 'rgba(130,230,150,0.95)' },
      { v: [0, 0, 1], pos: 'S', neg: 'I', color: 'rgba(130,180,255,0.95)' },
    ];

    ctx.save();
    ctx.font = FONT_BOLD;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const a of axes) {
      const sx = a.v[0] * right[0] + a.v[1] * right[1] + a.v[2] * right[2];
      const sy = a.v[0] * up[0] + a.v[1] * up[1] + a.v[2] * up[2];
      const depth = a.v[0] * forward[0] + a.v[1] * forward[1] + a.v[2] * forward[2];

      // Which letter goes where is decided purely by which end of the axis it
      // is: the +v end is always the positive letter. Depth only controls the
      // fade, so the arm pointing away from the camera recedes. Letting depth
      // choose the letter instead mirrors the gizmo whenever an axis tips past
      // the camera plane, which is exactly when a viewer relies on it.
      for (const end of [1, -1] as const) {
        const near = end * depth < 0;
        ctx.globalAlpha = near ? 1 : 0.3;
        ctx.strokeStyle = a.color;
        ctx.lineWidth = near ? 1.6 : 1;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + end * sx * r, cy - end * sy * r);
        ctx.stroke();

        ctx.fillStyle = a.color;
        ctx.fillText(end === 1 ? a.pos : a.neg, cx + end * sx * (r + 8), cy - end * sy * (r + 8));
      }
    }
    ctx.restore();
  }
}

/** Data-probe line for the status bar: HU, structure name and patient coordinates. */
export function describeProbe(state: AppState): string {
  const { volume, labels, probe } = state;
  if (!volume || !probe) return '';
  const hu = sampleVoxel(volume, probe[0], probe[1], probe[2]);
  if (hu === null) return '';

  const world = voxelToWorld(volume, [Math.round(probe[0]), Math.round(probe[1]), Math.round(probe[2])]);
  const parts = [
    `${hu.toFixed(0)} HU`,
    `voxel ${Math.round(probe[0])}, ${Math.round(probe[1])}, ${Math.round(probe[2])}`,
    formatPatientPosition(world),
  ];

  if (labels) {
    const idx = labelAtVoxel(labels, probe[0], probe[1], probe[2]);
    const s: Structure | undefined = labels.structures.find((st) => st.index === idx);
    if (s) parts.splice(1, 0, s.name);
  }
  return parts.join('   |   ');
}
