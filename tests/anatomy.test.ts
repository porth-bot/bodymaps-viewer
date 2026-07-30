import { describe, expect, it } from 'vitest';

import { ANATOMY, lookupAnatomy, paletteColor } from '../src/core/anatomy';
import type { AnatomyEntry } from '../src/core/anatomy';
import { WINDOW_PRESETS, autoWindow, rangeToWindow, windowToRange } from '../src/core/presets';

/** The nine masks that ship with BDMAP_00000338. */
const SAMPLE_KEYS = [
  'aorta',
  'gall_bladder',
  'kidney_left',
  'kidney_right',
  'liver',
  'pancreas',
  'postcava',
  'spleen',
  'stomach',
] as const;

// --- colour distance -------------------------------------------------------
// Overlay readability is a perceptual question, so the distinctness assertions
// use CIE76 dE in CIELAB rather than raw RGB, which badly underrates how close
// two dark colours look. dE 2.3 is one just-noticeable difference.

function srgbToLab(rgb: readonly [number, number, number]): [number, number, number] {
  const linear = rgb.map((c) => {
    const v = c / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
  const [r, g, b] = linear;
  const x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047;
  const y = r * 0.2126729 + g * 0.7151522 + b * 0.072175;
  const z = (r * 0.0193339 + g * 0.119192 + b * 0.9503041) / 1.08883;
  const f = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : (841 / 108) * t + 4 / 29);
  const [fx, fy, fz] = [f(x), f(y), f(z)];
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

function deltaE(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  const la = srgbToLab(a);
  const lb = srgbToLab(b);
  return Math.hypot(la[0] - lb[0], la[1] - lb[1], la[2] - lb[2]);
}

function rgbDistance(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function expectValidColor(color: readonly number[], context: string): void {
  expect(color, context).toHaveLength(3);
  for (const channel of color) {
    expect(Number.isInteger(channel), `${context} channel ${channel}`).toBe(true);
    expect(channel, `${context} channel ${channel}`).toBeGreaterThanOrEqual(0);
    expect(channel, `${context} channel ${channel}`).toBeLessThanOrEqual(255);
  }
}

describe('window presets', () => {
  it('all have a positive width and a finite level', () => {
    expect(WINDOW_PRESETS.length).toBeGreaterThanOrEqual(8);
    for (const preset of WINDOW_PRESETS) {
      expect(preset.window, preset.id).toBeGreaterThan(0);
      expect(Number.isFinite(preset.level), preset.id).toBe(true);
      expect(preset.name.length, preset.id).toBeGreaterThan(0);
    }
  });

  it('have unique ids and unique digit hotkeys', () => {
    const ids = WINDOW_PRESETS.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);

    const hotkeys = WINDOW_PRESETS.map((p) => p.hotkey).filter((h): h is string => h !== undefined);
    expect(new Set(hotkeys).size).toBe(hotkeys.length);
    for (const hotkey of hotkeys) expect(hotkey).toMatch(/^[1-9]$/);
  });

  it('cover the presets a CT reader expects to find', () => {
    const ids = new Set(WINDOW_PRESETS.map((p) => p.id));
    for (const id of ['soft-tissue', 'liver', 'lung', 'bone', 'brain', 'mediastinum', 'angio', 'full-range']) {
      expect(ids.has(id), `missing preset ${id}`).toBe(true);
    }
  });

  it('places the standard values where clinical convention puts them', () => {
    const byId = new Map(WINDOW_PRESETS.map((p) => [p.id, p]));
    // Spot-check the two that a reader would notice immediately if wrong.
    expect(byId.get('lung')).toMatchObject({ level: -600, window: 1500 });
    expect(byId.get('brain')).toMatchObject({ level: 40, window: 80 });
    // The full-range preset must span exactly what this dataset can store.
    expect(windowToRange(byId.get('full-range')!)).toEqual([-1000, 1000]);
  });
});

describe('window and range conversion', () => {
  it('round-trips window -> range -> window for every preset', () => {
    for (const preset of WINDOW_PRESETS) {
      const [lo, hi] = windowToRange(preset);
      const back = rangeToWindow(lo, hi);
      expect(back.level, preset.id).toBeCloseTo(preset.level, 10);
      expect(back.window, preset.id).toBeCloseTo(preset.window, 10);
    }
  });

  it('round-trips range -> window -> range', () => {
    const ranges: Array<[number, number]> = [
      [-1000, 1000],
      [-150, 250],
      [0, 1],
      [-45.5, 12.25],
      [1e-3, 2e-3],
    ];
    for (const [lo, hi] of ranges) {
      const [backLo, backHi] = windowToRange(rangeToWindow(lo, hi));
      expect(backLo).toBeCloseTo(lo, 10);
      expect(backHi).toBeCloseTo(hi, 10);
    }
  });

  it('accepts reversed bounds', () => {
    expect(rangeToWindow(250, -150)).toEqual(rangeToWindow(-150, 250));
  });
});

describe('autoWindow', () => {
  const cases: Array<[number, number]> = [
    [-1000, 1000],
    [0, 4095],
    [-3.5, 7.25],
    [100, 101],
  ];

  it('produces a positive width inside the data range', () => {
    for (const [min, max] of cases) {
      const wl = autoWindow(min, max);
      expect(wl.window, `${min}..${max}`).toBeGreaterThan(0);
      const [lo, hi] = windowToRange(wl);
      expect(lo, `${min}..${max}`).toBeGreaterThanOrEqual(min);
      expect(hi, `${min}..${max}`).toBeLessThanOrEqual(max);
      expect(hi).toBeGreaterThan(lo);
    }
  });

  it('honours explicit trim fractions', () => {
    const wl = autoWindow(0, 100, 0.1, 0.9);
    expect(windowToRange(wl)).toEqual([10, 90]);
  });

  it('tolerates reversed and out-of-range fractions', () => {
    expect(windowToRange(autoWindow(0, 100, 0.9, 0.1))).toEqual([10, 90]);
    const clamped = autoWindow(0, 100, -5, 5);
    expect(windowToRange(clamped)).toEqual([0, 100]);
  });

  it('still returns a usable window for degenerate input', () => {
    for (const wl of [autoWindow(42, 42), autoWindow(NaN, 10), autoWindow(0, Infinity)]) {
      expect(wl.window).toBeGreaterThan(0);
      expect(Number.isFinite(wl.level)).toBe(true);
    }
  });
});

describe('lookupAnatomy', () => {
  it('resolves every mask in the sample study', () => {
    for (const key of SAMPLE_KEYS) {
      const entry = lookupAnatomy(key);
      expect(entry.key, key).toBe(key);
      expect(entry.name.length, key).toBeGreaterThan(0);
      expect(ANATOMY[key], key).toBe(entry);
    }
  });

  it('gives the sample structures their clinical names', () => {
    expect(lookupAnatomy('postcava').name).toBe('Inferior vena cava');
    expect(lookupAnatomy('gall_bladder').name).toBe('Gallbladder');
    expect(lookupAnatomy('kidney_left').name).toBe('Kidney (left)');
  });

  it('is forgiving about key formatting', () => {
    const canonical = lookupAnatomy('kidney_left');
    for (const alias of [
      'kidney_left',
      'kidney-left',
      'Kidney Left',
      'left kidney',
      'KIDNEY_LEFT',
      '  kidney   left  ',
      'kidney_left.nii.gz',
      'kidney_l',
      'Left_Kidney',
    ]) {
      expect(lookupAnatomy(alias), alias).toBe(canonical);
    }
  });

  it('resolves the common synonyms for the same structure', () => {
    expect(lookupAnatomy('gallbladder')).toBe(lookupAnatomy('gall_bladder'));
    expect(lookupAnatomy('inferior_vena_cava')).toBe(lookupAnatomy('postcava'));
    expect(lookupAnatomy('IVC')).toBe(lookupAnatomy('postcava'));
    expect(lookupAnatomy('oesophagus')).toBe(lookupAnatomy('esophagus'));
    expect(lookupAnatomy('adrenal left')).toBe(lookupAnatomy('adrenal_gland_left'));
    expect(lookupAnatomy('clavicle_right')).toBe(lookupAnatomy('clavicula_right'));
  });

  it('gives unknown keys a stable colour and a readable name', () => {
    const first = lookupAnatomy('mystery_organ');
    const second = lookupAnatomy('Mystery Organ');
    expect(second.color).toEqual(first.color);
    expect(first.name).toBe('Mystery organ');
    expect(lookupAnatomy('weird_thing_left').name).toBe('Weird thing (left)');
    expectValidColor(first.color, 'mystery_organ');
  });

  it('spreads unknown keys across mostly different colours', () => {
    const keys = Array.from({ length: 200 }, (_, i) => `unmapped_structure_${i}`);
    const colors = keys.map((k) => lookupAnatomy(k).color.join(','));
    const distinct = new Set(colors).size;
    expect(distinct).toBeGreaterThanOrEqual(Math.floor(keys.length * 0.9));
  });
});

describe('paletteColor', () => {
  it('returns valid triples for arbitrary indices', () => {
    for (const n of [0, 1, 7, 63, 4095, -3, 1e6, NaN, Infinity]) {
      expectValidColor(paletteColor(n), `paletteColor(${n})`);
    }
  });

  it('keeps adjacent indices far apart', () => {
    for (let n = 0; n < 64; n++) {
      expect(deltaE(paletteColor(n), paletteColor(n + 1)), `${n} vs ${n + 1}`).toBeGreaterThan(25);
    }
  });
});

describe('anatomy colours', () => {
  it('are all valid 0-255 triples', () => {
    for (const [key, entry] of Object.entries(ANATOMY)) {
      expectValidColor(entry.color, key);
      expect(entry.key, key).toBe(key);
    }
  });

  it('keeps the nine sample structures visually separable', () => {
    const entries: AnatomyEntry[] = SAMPLE_KEYS.map((k) => lookupAnatomy(k));
    let worstDeltaE = Infinity;
    let worstRgb = Infinity;
    let worstPair = '';
    for (let i = 0; i < entries.length; i++) {
      for (let j = i + 1; j < entries.length; j++) {
        const d = deltaE(entries[i].color, entries[j].color);
        if (d < worstDeltaE) {
          worstDeltaE = d;
          worstPair = `${entries[i].key} vs ${entries[j].key}`;
        }
        worstRgb = Math.min(worstRgb, rgbDistance(entries[i].color, entries[j].color));
      }
    }
    // dE 15 is roughly six just-noticeable differences, comfortably readable at
    // the partial opacity an overlay is drawn with.
    expect(worstDeltaE, `closest pair: ${worstPair}`).toBeGreaterThanOrEqual(15);
    expect(worstRgb, `closest pair: ${worstPair}`).toBeGreaterThanOrEqual(40);
  });

  it('gives the two kidneys different colours', () => {
    expect(lookupAnatomy('kidney_left').color).not.toEqual(lookupAnatomy('kidney_right').color);
  });

  it('keeps neighbouring members of a bony series apart', () => {
    // A rib cage or a lumbar spine rendered in one flat ivory reads as a single
    // region, so consecutive members have to differ even though the standard
    // LUT gives the whole series one colour.
    const series: string[][] = [
      ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'].map((v) => `vertebrae_${v}`),
      ['L1', 'L2', 'L3', 'L4', 'L5'].map((v) => `vertebrae_${v}`),
      Array.from({ length: 12 }, (_, i) => `rib_left_${i + 1}`),
      Array.from({ length: 12 }, (_, i) => `rib_right_${i + 1}`),
    ];
    for (const keys of series) {
      for (let i = 1; i < keys.length; i++) {
        const gap = deltaE(ANATOMY[keys[i - 1]].color, ANATOMY[keys[i]].color);
        expect(gap, `${keys[i - 1]} vs ${keys[i]}`).toBeGreaterThan(2.3);
      }
    }
  });

  it('keeps every bilateral pair in the table apart', () => {
    for (const key of Object.keys(ANATOMY)) {
      if (!key.endsWith('_left')) continue;
      const right = ANATOMY[`${key.slice(0, -5)}_right`];
      if (!right) continue;
      expect(deltaE(ANATOMY[key].color, right.color), key).toBeGreaterThan(8);
    }
  });
});
