/**
 * Radiological window/level presets and the conversions around them.
 *
 * A CT window is a linear ramp: everything at or below `level - window/2`
 * renders black, everything at or above `level + window/2` renders white.
 * Clinicians quote windows as "WW/WL" (width then centre), both in Hounsfield
 * units, so the numbers below are the ones a radiologist would read off a
 * console rather than anything rescaled for the renderer.
 *
 * Values are the widely taught defaults from Radiopaedia's CT windowing
 * reference (https://radiopaedia.org/articles/windowing-ct), cross-checked
 * against LITFL's abdominal CT window series
 * (https://litfl.com/abdominal-ct-windows-advanced/) and Radiology Cafe's CT
 * overview (https://www.radiologycafe.com/radiology-basics/imaging-modalities/ct-overview/).
 * 3D Slicer ships slightly narrower defaults in
 * Modules/Loadable/Volumes/Resources/VolumeDisplayPresets.json (CT-Abdomen
 * 350/40, CT-Bone 1000/400, CT-Lung 1400/-500, CT-Brain 100/50); where the two
 * disagree we follow the clinical convention and note Slicer's value inline,
 * because the viewer is read by people who learned the Radiopaedia numbers.
 *
 * Exact preset numbers vary by a few tens of HU between institutions. These are
 * the mid-range consensus values, not a claim that any single site uses them.
 *
 * A note on this dataset specifically: BDMAP_00000338 stores CT as int16 with
 * scl_slope = 2000/65535 and scl_inter = 1000/65535, so the entire int16 range
 * maps onto exactly [-1000, +1000] HU. The bone preset reaches +1300 HU and the
 * lung preset reaches -1350 HU, both past the ends of the stored range, so both
 * clip flat at the extremes on this volume. That is a property of how the data
 * was quantised on write, not a defect in the windowing code. Volumes stored at
 * full HU depth use the same presets and show the extra contrast.
 */

import type { Preset, WindowLevel } from './types';

/**
 * Smallest window we will ever hand back. A zero-width window makes the display
 * transfer function divide by zero, so degenerate ranges collapse to this
 * instead, which renders the constant volume as flat mid-grey.
 */
const MIN_WINDOW = 1e-6;

/** Window used when the caller passes a range we cannot interpret at all. */
const FALLBACK_WINDOW = 2000;

export const WINDOW_PRESETS: Preset[] = [
  // Ordered so the digits a user reaches for first are the ones an abdominal
  // CT reader actually cycles through.
  { id: 'soft-tissue', name: 'Soft tissue / abdomen', level: 50, window: 400, hotkey: '1' },
  { id: 'liver', name: 'Liver', level: 30, window: 150, hotkey: '2' },
  { id: 'mediastinum', name: 'Mediastinum', level: 50, window: 350, hotkey: '3' },
  // Radiopaedia lists 600/200 for vascular and cardiac work. Narrower CTA
  // windows exist for calcified plaque, but 600/200 is the general default.
  { id: 'angio', name: 'CT angiography', level: 200, window: 600, hotkey: '4' },
  // Slicer uses 1400/-500 for the same job.
  { id: 'lung', name: 'Lung', level: -600, window: 1500, hotkey: '5' },
  // Slicer uses 1000/400; 1800 is the value taught for reading cortical detail.
  { id: 'bone', name: 'Bone', level: 400, window: 1800, hotkey: '6' },
  { id: 'brain', name: 'Brain', level: 40, window: 80, hotkey: '7' },
  // Radiopaedia gives subdural as a range (W 130-300, L 50-100); 200/75 is the
  // midpoint and the value most departments preset.
  { id: 'subdural', name: 'Subdural', level: 75, window: 200, hotkey: '8' },
  // Spans exactly the [-1000, +1000] HU this dataset can represent, so it is
  // the "show me everything that is there" escape hatch.
  { id: 'full-range', name: 'Full range', level: 0, window: 2000, hotkey: '9' },
];

/** Lower and upper HU bound of a window, in that order. */
export function windowToRange(wl: WindowLevel): [number, number] {
  const half = Math.abs(wl.window) / 2;
  return [wl.level - half, wl.level + half];
}

/**
 * Inverse of `windowToRange`. Accepts the bounds in either order so a
 * drag-to-window interaction can pass raw start/end values.
 */
export function rangeToWindow(lo: number, hi: number): WindowLevel {
  const low = Math.min(lo, hi);
  const high = Math.max(lo, hi);
  return { level: (low + high) / 2, window: Math.max(high - low, MIN_WINDOW) };
}

/**
 * A usable default window for a volume that no clinical preset fits, such as
 * MR, PET or an unlabelled float volume.
 *
 * `percentileLow` and `percentileHigh` are fractions of the `min`..`max` span,
 * not true histogram percentiles: this function deliberately takes only the two
 * extrema so it can run before any histogram exists. Trimming 2% off each end
 * keeps a single hot voxel or a metal artefact from flattening the whole image.
 * A caller that has already built a histogram should skip this and pass its own
 * percentile values straight to `rangeToWindow`.
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

  let fracLow = clamp01(percentileLow, 0);
  let fracHigh = clamp01(percentileHigh, 1);
  if (fracLow > fracHigh) [fracLow, fracHigh] = [fracHigh, fracLow];

  const lo = low + span * fracLow;
  const hi = low + span * fracHigh;
  // Both fractions equal (or so close that the span collapses) would give a
  // zero-width window, which shows a hard black/white split instead of an
  // image, so fall back to the untrimmed range.
  if (hi - lo < span * 1e-3) return rangeToWindow(low, high);
  return rangeToWindow(lo, hi);
}

/** `fallback` is used for NaN/Infinity so one bad argument cannot silently
 * collapse the window to a sliver at one end of the range. */
function clamp01(v: number, fallback: number): number {
  if (!Number.isFinite(v)) return fallback;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
