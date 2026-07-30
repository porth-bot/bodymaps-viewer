/**
 * Viewport layout.
 *
 * All four views share a single WebGL2 canvas and are separated with
 * gl.viewport/gl.scissor. One canvas means one context, which means the 25 MB
 * volume texture is uploaded once instead of once per view. Four contexts
 * would need four copies and would hit the browser's context limit as soon as
 * anything else on the page wanted one.
 */

import type { ViewKind } from '../core/types';

export type LayoutMode = 'fourUp' | 'axial' | 'coronal' | 'sagittal' | 'volume' | 'row';

/** Rect in CSS pixels with a top-left origin, matching DOM coordinates. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ViewportRect extends Rect {
  view: ViewKind;
}

export const LAYOUT_LABELS: Record<LayoutMode, string> = {
  fourUp: 'Four up',
  axial: 'Axial only',
  coronal: 'Coronal only',
  sagittal: 'Sagittal only',
  volume: '3D only',
  row: 'Three slices',
};

/** Gap between panes, in CSS pixels. */
const GAP = 2;

export function computeLayout(mode: LayoutMode, width: number, height: number): ViewportRect[] {
  const w = Math.max(1, width);
  const h = Math.max(1, height);

  if (mode === 'fourUp') {
    const halfW = (w - GAP) / 2;
    const halfH = (h - GAP) / 2;
    const right = halfW + GAP;
    const bottom = halfH + GAP;
    return [
      { view: 'axial', x: 0, y: 0, width: halfW, height: halfH },
      { view: 'coronal', x: right, y: 0, width: w - right, height: halfH },
      { view: 'sagittal', x: 0, y: bottom, width: halfW, height: h - bottom },
      { view: 'volume', x: right, y: bottom, width: w - right, height: h - bottom },
    ];
  }

  if (mode === 'row') {
    const paneW = (w - GAP * 2) / 3;
    return [
      { view: 'axial', x: 0, y: 0, width: paneW, height: h },
      { view: 'coronal', x: paneW + GAP, y: 0, width: paneW, height: h },
      { view: 'sagittal', x: (paneW + GAP) * 2, y: 0, width: w - (paneW + GAP) * 2, height: h },
    ];
  }

  return [{ view: mode, x: 0, y: 0, width: w, height: h }];
}

export function rectAt(rects: ViewportRect[], x: number, y: number): ViewportRect | null {
  for (const r of rects) {
    if (x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height) return r;
  }
  return null;
}
