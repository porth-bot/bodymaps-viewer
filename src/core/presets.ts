/**
 * CT window/level presets, and the conversions around them.
 *
 * Presets are quoted the way a radiologist reads them off a console, "WW/WL",
 * width then centre, both in Hounsfield units. They follow Radiopaedia's CT
 * windowing reference (https://radiopaedia.org/articles/windowing-ct); where
 * 3D Slicer's VolumeDisplayPresets.json disagrees, its value is noted inline.
 * Institutions differ by a few tens of HU either way.
 */

import type { Preset, WindowLevel } from './types';

// Zero width divides by zero in the display transfer function.
const MIN_WINDOW = 1e-6;

/** Window used when the caller passes a range we cannot interpret at all. */
const FALLBACK_WINDOW = 2000;

// BDMAP_00000338 stores CT as int16 with scl_slope = 2000/65535 and scl_inter =
// 1000/65535, so the whole int16 range maps onto exactly [-1000, +1000] HU. Bone
// reaches +1300 and lung -1350, so both clip flat on this volume. That is the
// quantisation on write, not the windowing.
export const WINDOW_PRESETS: Preset[] = [
  // Digit order follows what an abdominal reader actually cycles through.
  // Slicer: 350/40.
  { id: 'soft-tissue', name: 'Soft tissue / abdomen', level: 50, window: 400, hotkey: '1' },
  { id: 'liver', name: 'Liver', level: 30, window: 150, hotkey: '2' },
  { id: 'mediastinum', name: 'Mediastinum', level: 50, window: 350, hotkey: '3' },
  // Narrower CTA windows exist for calcified plaque; 600/200 is the general one.
  { id: 'angio', name: 'CT angiography', level: 200, window: 600, hotkey: '4' },
  // Slicer: 1400/-500.
  { id: 'lung', name: 'Lung', level: -600, window: 1500, hotkey: '5' },
  // Slicer: 1000/400. 1800 is what gets taught for cortical detail.
  { id: 'bone', name: 'Bone', level: 400, window: 1800, hotkey: '6' },
  // Slicer: 100/50.
  { id: 'brain', name: 'Brain', level: 40, window: 80, hotkey: '7' },
  // Radiopaedia gives a range here (W 130-300, L 50-100); this is the midpoint.
  { id: 'subdural', name: 'Subdural', level: 75, window: 200, hotkey: '8' },
  { id: 'full-range', name: 'Full range', level: 0, window: 2000, hotkey: '9' },
];

/** Lower and upper HU bound of a window, in that order. */
export function windowToRange(wl: WindowLevel): [number, number] {
  const half = Math.abs(wl.window) / 2;
  return [wl.level - half, wl.level + half];
}

/**
 * Bounds may arrive in either order, so a drag-to-window can pass raw
 * start/end. Width comes back positive and finite: downstream divides by it.
 */
export function rangeToWindow(lo: number, hi: number): WindowLevel {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    return { level: 0, window: FALLBACK_WINDOW };
  }
  const low = Math.min(lo, hi);
  const high = Math.max(lo, hi);
  // Halve each bound before summing: `low + high` overflows to Infinity for
  // finite endpoints near the float64 limit, and halving afterwards keeps it
  // there. Both forms are exact in binary floating point everywhere else.
  const level = low / 2 + high / 2;
  const width = high - low;
  // The subtraction can overflow where the sum does not, and Math.max would
  // carry the NaN straight past the MIN_WINDOW floor.
  if (!Number.isFinite(width)) return { level, window: FALLBACK_WINDOW };
  return { level, window: Math.max(width, MIN_WINDOW) };
}

/**
 * Default window for a volume no clinical preset fits: MR, PET, an unlabelled
 * float volume. `percentileLow`/`percentileHigh` are fractions of the
 * `min`..`max` span, NOT true histogram percentiles, since this only gets the
 * two extrema and has to run before any histogram exists. Trimming 2% off each
 * end stops one hot voxel or a metal artefact flattening the image. Callers
 * holding a histogram should pass real percentiles to `rangeToWindow` instead.
 */
export function autoWindow(
  min: number,
  max: number,
  percentileLow = 0.02,
  percentileHigh = 0.98,
): WindowLevel {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { level: 0, window: FALLBACK_WINDOW };
  }

  const low = Math.min(min, max);
  const high = Math.max(min, max);
  const span = high - low;
  if (span <= 0) return { level: low, window: MIN_WINDOW };
  // Two finite extrema can still span more than float64 can subtract, and the
  // trim below would put both bounds at Infinity. Nothing to trim off a range
  // that wide anyway.
  if (!Number.isFinite(span)) return rangeToWindow(low, high);

  let fracLow = clamp01(percentileLow, 0);
  let fracHigh = clamp01(percentileHigh, 1);
  if (fracLow > fracHigh) [fracLow, fracHigh] = [fracHigh, fracLow];

  const lo = low + span * fracLow;
  const hi = low + span * fracHigh;
  // Equal fractions, or close enough that the span collapses, give a zero-width
  // window: a hard black/white split rather than an image.
  if (hi - lo < span * 1e-3) return rangeToWindow(low, high);
  return rangeToWindow(lo, hi);
}

/** NaN/Infinity takes `fallback`, rather than collapsing the window to a sliver. */
function clamp01(v: number, fallback: number): number {
  if (!Number.isFinite(v)) return fallback;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
