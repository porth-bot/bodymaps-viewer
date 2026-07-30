/**
 * Names and colours for the anatomical structures the viewer can overlay.
 *
 * Colour source
 * -------------
 * The base is 3D Slicer's GenericAnatomyColors table
 * (Slicer/Base/Logic/Resources/ColorFiles/GenericAnatomyColors.txt), which is
 * also what backs the SlicerTotalSegmentator terminology file
 * (lassoan/SlicerTotalSegmentator, Resources/SegmentationCategoryTypeModifier-TotalSegmentator.term.json).
 * TotalSegmentator itself ships no colour table; its published colours are the
 * ones that terminology mapping resolves to, so those two files together are
 * "the TotalSegmentator LUT" in practice. Key spellings follow TotalSegmentator
 * v2 (wasserth/TotalSegmentator, totalsegmentator/map_to_binary.py) except for
 * the nine structures in the sample study, which keep their AbdomenAtlas names
 * (MrGiovanni/AbdomenAtlas) since that is what the files on disk are called.
 *
 * Where we depart from the LUT, and why
 * ------------------------------------
 * GenericAnatomyColors is tuned for realistic 3D surface renders, not for flat
 * 2D overlays, and it has three properties that make it unusable verbatim:
 *
 *   1. Bilateral pairs share one colour. Both kidneys are 185,102,83; every rib
 *      is 253,232,158. A viewer that has to tell left from right cannot use
 *      that, so the contralateral side takes a lightness step (see
 *      `contralateral`).
 *   2. A whole tissue class is often one colour. All twelve thoracic vertebrae
 *      are 226,202,134; the myocardium and all four cardiac chambers sit inside
 *      dE 5 of each other. Members of a series are spread along a ramp (see
 *      `seriesShade`) and crowded groups are given an explicit ladder.
 *   3. The abdominal viscera are all one warm salmon family. Slicer's liver
 *      (221,130,101) and stomach (216,132,105) are CIE76 dE 3.7 apart, roughly
 *      one and a half just-noticeable differences, and the kidneys sit inside
 *      the same wedge as liver and aorta.
 *
 * Every departure keeps the structure inside a colour family a reader would
 * expect (bone stays osseous, arteries stay red, veins stay blue) and is marked
 * RETUNED below with the LUT value it replaces. After the retune the tightest
 * pair among the nine sample structures is Slicer's own liver/aorta at dE 18.2,
 * so no departure of ours is the limiting factor there.
 *
 * Separation is scoped to structures that can share a slice. Across the whole
 * 132-entry table the closest pair is dE 3.2 (two rungs of the same rib ramp),
 * and the closest pair from two different structures is dE 3.4 (clavicle
 * against hip bone). That floor is deliberate: chasing global separation pushes
 * bone off ivory and muscle off red, and a reader notices a lime-green rib far
 * faster than two similar tans that never appear together. `tests/anatomy.test.ts`
 * pins both numbers, so a retune that lowers them fails the suite rather than
 * making this paragraph quietly wrong.
 */

export interface AnatomyEntry {
  key: string;
  name: string;
  color: [number, number, number];
}

type RGB = [number, number, number];
type Def = readonly [key: string, name: string, color: RGB];

/**
 * Lightness step applied to the right side of a bilateral pair. It normally
 * darkens, but a colour that is already dark would sink into the CT background
 * and read as a hole rather than a label, so those get lightened instead.
 */
const CONTRALATERAL_STEP = 0.28;
const CONTRALATERAL_FLOOR = 110;

function contralateral(c: RGB): RGB {
  const darker: RGB = [
    Math.round(c[0] * (1 - CONTRALATERAL_STEP)),
    Math.round(c[1] * (1 - CONTRALATERAL_STEP)),
    Math.round(c[2] * (1 - CONTRALATERAL_STEP)),
  ];
  if (Math.max(darker[0], darker[1], darker[2]) >= CONTRALATERAL_FLOOR) return darker;
  return [
    Math.round(c[0] + (255 - c[0]) * CONTRALATERAL_STEP),
    Math.round(c[1] + (255 - c[1]) * CONTRALATERAL_STEP),
    Math.round(c[2] + (255 - c[2]) * CONTRALATERAL_STEP),
  ];
}

function clampByte(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)));
}

/** Emit `<stem>_left` / `<stem>_right`, deriving the right colour if not given. */
function pair(stem: string, name: string, left: RGB, right?: RGB): Def[] {
  return [
    [`${stem}_left`, `${name} (left)`, left],
    [`${stem}_right`, `${name} (right)`, right ?? contralateral(left)],
  ];
}

// --- vertebrae and ribs ----------------------------------------------------
// These are long uniform series in the LUT: every cervical vertebra is one
// ivory, every rib another. A sagittal slice shows a whole series at once, so
// each member gets its own rung of a ramp rather than one of two alternating
// shades, which would put six identical vertebrae on screen together.

/**
 * How far the last rung of a series drops below the LUT colour, and how far the
 * warm/cool tilt swings across the ramp. Twelve rungs separated by lightness
 * alone would need a drop of about half, running the last ribs down into a
 * brown that reads as a hole in the overlay rather than a label, so the ramp
 * moves along two axes at once and neither has to travel far.
 */
const SERIES_LIGHTNESS_SPAN = 0.26;
const SERIES_WARMTH_SPAN = 0.34;

/** Colour for member `n` (0-based) of a `count`-long series. */
function seriesShade(base: RGB, n: number, count: number): RGB {
  if (count < 2) return base;
  const p = rampPosition(n, count) / (count - 1);
  const lightness = 1 - SERIES_LIGHTNESS_SPAN * p;
  // The tilt rides on blue alone: pushing red or green instead would clip
  // against 255 on the near-white cervical ivory and flatten the bright half of
  // the ramp back into duplicates.
  const warmth = 1 + SERIES_WARMTH_SPAN * (p - 0.5);
  return [
    clampByte(base[0] * lightness),
    clampByte(base[1] * lightness),
    clampByte(base[2] * lightness * warmth),
  ];
}

/**
 * Which rung member `n` occupies. The ramp is walked in two passes, even
 * members then odd ones, so T7 and T8 land half a ramp apart while every member
 * still gets a rung to itself. Walking it in order would instead put the
 * smallest step exactly where it matters most, between two vertebrae that touch.
 */
function rampPosition(n: number, count: number): number {
  const half = Math.ceil(count / 2);
  return n % 2 === 0 ? n / 2 : half + (n - 1) / 2;
}

const VERTEBRA_REGIONS: ReadonlyArray<readonly [prefix: string, count: number, color: RGB]> = [
  ['C', 7, [255, 255, 207]], // Slicer cervical_vertebral_column
  ['T', 12, [226, 202, 134]], // Slicer thoracic_vertebral_column
  ['L', 5, [212, 188, 102]], // Slicer lumbar_vertebral_column
  ['S', 1, [212, 208, 122]], // SlicerTotalSegmentator, S1 vertebra
];

function vertebraDefs(): Def[] {
  const out: Def[] = [];
  for (const [prefix, count, color] of VERTEBRA_REGIONS) {
    for (let n = 1; n <= count; n++) {
      const shade = seriesShade(color, n - 1, count);
      out.push([`vertebrae_${prefix}${n}`, `Vertebra ${prefix}${n}`, shade]);
    }
  }
  return out;
}

// RETUNED, warmed a shade off Slicer's 253,232,158 so that neither the base
// nor its alternating shade lands on the thoracic vertebrae it sits beside.
const RIB_COLOR: RGB = [252, 222, 176];

const RIBS_PER_SIDE = 12;

function ribDefs(): Def[] {
  const out: Def[] = [];
  const sides: ReadonlyArray<readonly [side: 'left' | 'right', color: RGB]> = [
    ['left', RIB_COLOR],
    ['right', contralateral(RIB_COLOR)],
  ];
  for (const [side, base] of sides) {
    for (let n = 1; n <= RIBS_PER_SIDE; n++) {
      const shade = seriesShade(base, n - 1, RIBS_PER_SIDE);
      out.push([`rib_${side}_${n}`, `Rib ${n} (${side})`, shade]);
    }
  }
  return out;
}

// --- the table -------------------------------------------------------------

const DEFS: readonly Def[] = [
  // Abdominal viscera.
  ['liver', 'Liver', [221, 130, 101]], // Slicer liver
  ['spleen', 'Spleen', [157, 108, 162]], // Slicer spleen
  ['pancreas', 'Pancreas', [249, 180, 111]], // Slicer pancreas
  // RETUNED, Slicer stomach 216,132,105 is dE 3.7 from liver and the two share
  // a border on every upper abdominal slice.
  ['stomach', 'Stomach', [120, 190, 170]],
  ['gall_bladder', 'Gallbladder', [139, 150, 98]], // Slicer gallbladder
  // RETUNED, Slicer duodenum 255,253,229 is near-white and disappears against
  // bone and contrast.
  ['duodenum', 'Duodenum', [245, 205, 175]],
  ['small_bowel', 'Small bowel', [205, 167, 142]], // Slicer small_bowel
  // RETUNED, Slicer colon 204,168,143 is dE 1 from small bowel.
  ['colon', 'Colon', [150, 104, 44]],
  ['rectum', 'Rectum', [190, 96, 96]], // no LUT entry
  // RETUNED, Slicer esophagus 211,171,143 lands on small bowel.
  ['esophagus', 'Oesophagus', [226, 122, 168]],

  // Urinary tract and pelvis.
  // RETUNED pair. Slicer gives both kidneys 185,102,83, which collides with
  // liver and aorta; the right kidney abuts the liver on most axial slices.
  ...pair('kidney', 'Kidney', [165, 74, 58], [112, 46, 40]),
  ...pair('kidney_cyst', 'Kidney cyst', [205, 205, 100]), // Slicer cyst
  // RETUNED, Slicer's pale peach 249,186,150 lands on the psoas the glands sit
  // against at T12, and these are small structures that need to carry.
  ...pair('adrenal_gland', 'Adrenal gland', [252, 208, 78]),
  ['urinary_bladder', 'Urinary bladder', [222, 154, 132]], // Slicer urinary_bladder
  // RETUNED, Slicer prostate 230,158,140 is dE 3 from urinary bladder and sits
  // directly beneath it.
  ['prostate', 'Prostate', [188, 120, 152]],
  // RETUNED from 247,182,164, which lands on the psoas and glutes the ureters
  // run over; the right side is given explicitly so it clears the prostate.
  ...pair('ureter', 'Ureter', [236, 146, 190], [158, 86, 132]),

  // Arteries. Slicer keeps the whole arterial tree in one red, so branches that
  // touch the aorta are separated by lightness.
  ['aorta', 'Aorta', [224, 97, 76]], // Slicer aorta
  // RETUNED, the celiac trunk arises straight off the aorta and must not read
  // as part of it.
  ['celiac_trunk', 'Coeliac trunk', [176, 32, 60]],
  ['pulmonary_artery', 'Pulmonary artery', [0, 122, 171]], // SlicerTotalSegmentator
  ['brachiocephalic_trunk', 'Brachiocephalic trunk', [196, 121, 79]], // SlicerTotalSegmentator
  // RETUNED from the generic artery red 216,101,79, which is dE 4 from aorta.
  ...pair('subclavian_artery', 'Subclavian artery', [176, 60, 52]),
  ...pair('common_carotid_artery', 'Common carotid artery', [246, 146, 110]),
  // RETUNED from the generic artery red, which is dE 4 from the left kidney the
  // vessels run beside at the L4 level.
  ...pair('iliac_artery', 'Common iliac artery', [214, 84, 100], [166, 52, 68]),

  // Veins.
  ['postcava', 'Inferior vena cava', [0, 151, 206]], // Slicer vein / SlicerTotalSegmentator IVC
  ['superior_vena_cava', 'Superior vena cava', [0, 141, 226]], // SlicerTotalSegmentator
  // RETUNED, the LUT reuses the IVC blue here and the two meet at the porta
  // hepatis.
  ['portal_vein_and_splenic_vein', 'Portal and splenic vein', [76, 118, 190]],
  ['hepatic_vessel', 'Hepatic vessels', [104, 190, 226]], // no LUT entry
  ['pulmonary_vein', 'Pulmonary vein', [195, 45, 25]], // SlicerTotalSegmentator
  ...pair('brachiocephalic_vein', 'Brachiocephalic vein', [64, 176, 220]),
  ...pair('iliac_vena', 'Common iliac vein', [120, 190, 235]),

  // Heart. The whole-organ label keeps Slicer's colour; the chamber labels are a
  // RETUNED ladder, because Slicer puts the myocardium and all four chambers
  // inside dE 5 of one another and the chamber task renders them together.
  ['heart', 'Heart', [206, 110, 84]], // Slicer heart
  ['heart_myocardium', 'Myocardium', [186, 92, 68]],
  ['heart_atrium_left', 'Left atrium', [250, 178, 156]],
  ['heart_atrium_right', 'Right atrium', [236, 140, 88]],
  ['heart_ventricle_left', 'Left ventricle', [152, 55, 13]], // Slicer left_ventricle_of_heart
  ['heart_ventricle_right', 'Right ventricle', [176, 104, 124]],
  ['atrial_appendage_left', 'Left atrial appendage', [188, 92, 124]], // no LUT entry
  ['pericardium', 'Pericardium', [255, 244, 209]], // Slicer pericardium

  // Airway and lungs.
  ['trachea', 'Trachea', [182, 228, 255]], // Slicer trachea
  // RETUNED. Slicer's lung beige sits inside the ivory rib cage that lungs are
  // always displayed against, so the lung family moves to the pale blue-grey
  // that aerated lung is conventionally rendered in. Lobe sets lightness, side
  // sets hue.
  ['lung_left', 'Lung (left)', [176, 158, 196]],
  ['lung_right', 'Lung (right)', [128, 156, 182]],
  ['lung_upper_lobe_left', 'Upper lobe (left lung)', [146, 130, 168]],
  ['lung_lower_lobe_left', 'Lower lobe (left lung)', [206, 190, 232]],
  ['lung_upper_lobe_right', 'Upper lobe (right lung)', [96, 126, 150]],
  ['lung_middle_lobe_right', 'Middle lobe (right lung)', [150, 180, 204]],
  ['lung_lower_lobe_right', 'Lower lobe (right lung)', [188, 214, 232]],

  // Endocrine and neural.
  ['thyroid_gland', 'Thyroid gland', [220, 160, 30]], // SlicerTotalSegmentator thyroid
  ['brain', 'Brain', [250, 250, 225]], // Slicer brain
  ['spinal_cord', 'Spinal cord', [244, 214, 49]], // Slicer spinal_cord

  // Axial skeleton, on Slicer's warm ivory.
  ...vertebraDefs(),
  ...ribDefs(),
  ['sacrum', 'Sacrum', [172, 144, 62]], // RETUNED from 212,188,102, shared with L5
  ['sternum', 'Sternum', [250, 200, 104]], // RETUNED from 244,217,154, dE 6 from the thoracic spine
  ['costal_cartilages', 'Costal cartilage', [111, 184, 210]], // SlicerTotalSegmentator
  ['skull', 'Skull', [241, 213, 144]], // Slicer skull
  ['face', 'Face', [255, 218, 185]], // SlicerTotalSegmentator head
  // RETUNED to a cooler bone grey. The girdles overlie the ribs and the spine on
  // the same slices, and the LUT gives scapula, hip and the lumbar spine one
  // shared ivory, so the appendicular skeleton is shifted off the axial hue
  // while staying recognisably osseous. The shoulder needs six separable shades
  // (three bones, two sides), so those pairs are given explicitly.
  ...pair('humerus', 'Humerus', [244, 240, 226], [176, 172, 163]),
  ...pair('scapula', 'Scapula', [166, 161, 138], [112, 108, 92]),
  ...pair('clavicula', 'Clavicle', [204, 198, 172], [136, 131, 112]),
  ...pair('femur', 'Femur', [255, 238, 170]), // SlicerTotalSegmentator femur
  ...pair('hip', 'Hip bone', [184, 174, 142]),

  // Muscle. The LUT puts erector spinae, iliopsoas and all three gluteals
  // within dE 5 of one another; they are spread across the same red-brown
  // family by lightness so the pelvic overlay stays readable.
  ...pair('autochthon', 'Erector spinae', [205, 118, 105]),
  ...pair('iliopsoas', 'Iliopsoas', [240, 168, 132]),
  ...pair('gluteus_maximus', 'Gluteus maximus', [150, 74, 60]),
  ...pair('gluteus_medius', 'Gluteus medius', [186, 120, 62]),
  ...pair('gluteus_minimus', 'Gluteus minimus', [224, 168, 96]),
];

/**
 * Every lookup table below is null-prototyped. The keys come from file names a
 * user dropped on the page, and on a plain object a mask called "constructor"
 * would resolve through Object.prototype and hand back a function where an
 * entry is expected. `normalise` lowercases, so "constructor" is the only
 * inherited member that can survive it, but one is enough to abort a load.
 */
function emptyTable<T>(): Record<string, T> {
  return Object.create(null) as Record<string, T>;
}

/**
 * Alternative spellings seen in the wild, mapped to the canonical key. Keys and
 * values are both in normalised form. Left/right word order is handled
 * separately in `lookupAnatomy` so this table does not need a row per side.
 */
const ALIASES: Readonly<Record<string, string>> = Object.assign(emptyTable<string>(), {
  gallbladder: 'gall_bladder',
  inferior_vena_cava: 'postcava',
  vena_cava_inferior: 'postcava',
  ivc: 'postcava',
  svc: 'superior_vena_cava',
  vena_cava_superior: 'superior_vena_cava',
  oesophagus: 'esophagus',
  intestine: 'small_bowel',
  small_intestine: 'small_bowel',
  bowel: 'small_bowel',
  large_bowel: 'colon',
  large_intestine: 'colon',
  bladder: 'urinary_bladder',
  portal_vein: 'portal_vein_and_splenic_vein',
  splenic_vein: 'portal_vein_and_splenic_vein',
  portal_and_splenic_vein: 'portal_vein_and_splenic_vein',
  hepatic_vessels: 'hepatic_vessel',
  celiac_artery: 'celiac_trunk',
  coeliac_trunk: 'celiac_trunk',
  coeliac_artery: 'celiac_trunk',
  brachiocephalic_artery: 'brachiocephalic_trunk',
  myocardium: 'heart_myocardium',
  left_atrium: 'heart_atrium_left',
  right_atrium: 'heart_atrium_right',
  left_ventricle: 'heart_ventricle_left',
  right_ventricle: 'heart_ventricle_right',
  thyroid: 'thyroid_gland',
  costal_cartilage: 'costal_cartilages',
  head: 'face',
  // Stems of bilateral pairs. `resolveCanonical` reattaches the side, so one
  // row here covers both "clavicle" and "clavicle_left".
  adrenal: 'adrenal_gland',
  suprarenal_gland: 'adrenal_gland',
  clavicle: 'clavicula',
  erector_spinae: 'autochthon',
  deep_muscle_of_back: 'autochthon',
  psoas: 'iliopsoas',
  iliac_vein: 'iliac_vena',
  common_iliac_vein: 'iliac_vena',
  common_iliac_artery: 'iliac_artery',
  hip_bone: 'hip',
  cyst_kidney: 'kidney_cyst',
  renal_cyst: 'kidney_cyst',
});

function buildAnatomy(): Record<string, AnatomyEntry> {
  const table = emptyTable<AnatomyEntry>();
  for (const [key, name, color] of DEFS) {
    if (table[key]) throw new Error(`anatomy: duplicate key ${key}`);
    table[key] = Object.freeze({ key, name, color: Object.freeze(color) as RGB });
  }
  return table;
}

export const ANATOMY: Record<string, AnatomyEntry> = buildAnatomy();

const SIDE_WORDS = new Set(['left', 'right', 'l', 'r']);
const FILE_SUFFIXES = ['_nii_gz', '_nii', '_gz', '_mask', '_seg', '_label'];

/** Canonical keys indexed by their normalised form, e.g. vertebrae_l3 -> vertebrae_L3. */
const BY_NORMALISED: Readonly<Record<string, string>> = (() => {
  const index = emptyTable<string>();
  for (const key of Object.keys(ANATOMY)) index[normalise(key)] = key;
  return index;
})();

/**
 * Collapse a mask name to a comparison key: lower case, every run of
 * non-alphanumerics becomes one underscore, and any file-extension or
 * segmentation-qualifier tail is dropped. "Kidney Left", "kidney-left" and
 * "KIDNEY_LEFT.nii.gz" all land on "kidney_left".
 */
function normalise(raw: string): string {
  let s = raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  // Stripping runs to a fixed point because a real export carries an extension
  // and a qualifier at once ("liver_mask.nii.gz"), and taking only the outer
  // one leaves a name no table can match. Each pass shortens `s` by at least
  // three characters and never empties it, so this terminates.
  let stripped = true;
  while (stripped) {
    stripped = false;
    for (const suffix of FILE_SUFFIXES) {
      if (s.length > suffix.length && s.endsWith(suffix)) {
        s = s.slice(0, -suffix.length);
        stripped = true;
        break;
      }
    }
  }
  return s;
}

/** "left_kidney" -> "kidney_left", so English word order resolves too. */
function moveSideToEnd(normalised: string): string | null {
  const parts = normalised.split('_');
  if (parts.length < 2) return null;
  const first = parts[0];
  if (!SIDE_WORDS.has(first)) return null;
  const side = first === 'l' ? 'left' : first === 'r' ? 'right' : first;
  return [...parts.slice(1), side].join('_');
}

/** "kidney_l" -> "kidney_left". */
function expandSideAbbreviation(normalised: string): string | null {
  const parts = normalised.split('_');
  if (parts.length < 2) return null;
  const last = parts[parts.length - 1];
  if (last !== 'l' && last !== 'r') return null;
  return [...parts.slice(0, -1), last === 'l' ? 'left' : 'right'].join('_');
}

const unknownCache = new Map<string, AnatomyEntry>();

/**
 * Resolve a mask name to its display entry. Never throws: an unrecognised name
 * gets a deterministic palette colour and a title-cased label, so a mask this
 * table has not heard of still renders and still legends correctly.
 */
export function lookupAnatomy(key: string): AnatomyEntry {
  const normalised = normalise(key);

  for (const candidate of resolutionCandidates(normalised)) {
    const canonical = resolveCanonical(candidate);
    if (canonical) return ANATOMY[canonical];
  }

  const cached = unknownCache.get(normalised);
  if (cached) return cached;
  const entry: AnatomyEntry = Object.freeze({
    key: normalised,
    name: titleCase(normalised),
    color: Object.freeze(paletteColor(hashKey(normalised))) as RGB,
  });
  unknownCache.set(normalised, entry);
  return entry;
}

function resolveCanonical(candidate: string): string | undefined {
  const direct = BY_NORMALISED[candidate];
  if (direct) return direct;

  const aliased = ALIASES[candidate];
  if (aliased && BY_NORMALISED[aliased]) return BY_NORMALISED[aliased];

  const sided = /^(.+)_(left|right)$/.exec(candidate);
  if (sided) {
    const stem = ALIASES[sided[1]];
    if (stem && BY_NORMALISED[`${stem}_${sided[2]}`]) return BY_NORMALISED[`${stem}_${sided[2]}`];
  }
  return undefined;
}

/** Forms to try, cheapest first. */
function resolutionCandidates(normalised: string): string[] {
  const out = [normalised];
  const reordered = moveSideToEnd(normalised);
  if (reordered) out.push(reordered);
  const expanded = expandSideAbbreviation(normalised);
  if (expanded) out.push(expanded);
  if (reordered) {
    const both = expandSideAbbreviation(reordered);
    if (both) out.push(both);
  }
  return out;
}

function titleCase(normalised: string): string {
  if (!normalised) return 'Unknown';
  const parts = normalised.split('_');
  const last = parts[parts.length - 1];
  const side = last === 'left' || last === 'right' ? parts.pop() : null;
  const label = parts
    .map((w, i) => (i === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w))
    .join(' ');
  return side ? `${label} (${side})` : label;
}

/** FNV-1a, 32-bit. Cheap, well spread over short ASCII names, and stable. */
function hashKey(key: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0) % 4096;
}

// --- fallback palette ------------------------------------------------------

/**
 * 1/phi. Stepping the hue by an irrational fraction of the circle keeps
 * consecutive indices about 137.5 degrees apart, so no two neighbours land in
 * the same hue family no matter how many structures are requested.
 */
const GOLDEN_RATIO_CONJUGATE = 0.618033988749895;
const PALETTE_SATURATION = 0.62;
const PALETTE_VALUE = 0.95;

export function paletteColor(n: number): [number, number, number] {
  const index = Number.isFinite(n) ? Math.trunc(n) : 0;
  // The +1 offset keeps index 0 off pure red, which a reader would take for a
  // vessel.
  const hue = fract((index + 1) * GOLDEN_RATIO_CONJUGATE);
  return hsvToRgb(hue, PALETTE_SATURATION, PALETTE_VALUE);
}

function fract(x: number): number {
  return ((x % 1) + 1) % 1;
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const sector = Math.floor(h * 6) % 6;
  const f = h * 6 - Math.floor(h * 6);
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let rgb: [number, number, number];
  switch (sector) {
    case 0: rgb = [v, t, p]; break;
    case 1: rgb = [q, v, p]; break;
    case 2: rgb = [p, v, t]; break;
    case 3: rgb = [p, q, v]; break;
    case 4: rgb = [t, p, v]; break;
    default: rgb = [v, p, q]; break;
  }
  return [Math.round(rgb[0] * 255), Math.round(rgb[1] * 255), Math.round(rgb[2] * 255)];
}
