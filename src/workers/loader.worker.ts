/// <reference lib="webworker" />
/**
 * Loads a case off the network or off disk: download, gunzip, parse, reorient,
 * pack labels. Off the main thread because gunzipping a 16 MB scan and walking
 * 12 million voxels nine times freezes the UI for seconds.
 */

import { gunzipIfNeeded, parseNifti } from '../core/nifti';
import { reorientLike, reorientToRAS } from '../core/orientation';
import { buildLabelVolume, expandMaskFile, type MaskInput } from '../core/labelmap';
import { normalizeForGpu } from '../core/volume';
import type { LabelVolume, Volume } from '../core/types';

export interface LoadSource {
  key: string;
  url?: string;
  file?: File;
}

export interface LoadRequest {
  type: 'load';
  ct: LoadSource;
  masks: LoadSource[];
  /** Echoed back so a stale load can be discarded when the user switches cases. */
  token: number;
}

export type LoaderMessage =
  | { type: 'progress'; token: number; stage: string; fraction: number | null; detail?: string }
  | { type: 'volume'; token: number; volume: Volume; normalized: Float32Array; ms: number }
  | { type: 'labels'; token: number; labels: LabelVolume; ms: number }
  | { type: 'error'; token: number; message: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

function post(msg: LoaderMessage, transfer: Transferable[] = []): void {
  ctx.postMessage(msg, transfer);
}

async function readSource(src: LoadSource): Promise<ArrayBuffer> {
  if (src.file) return src.file.arrayBuffer();
  if (!src.url) throw new Error(`Source "${src.key}" has neither a URL nor a file`);
  const res = await fetch(src.url);
  if (!res.ok) {
    throw new Error(`Could not fetch ${src.url} (HTTP ${res.status} ${res.statusText})`);
  }
  return res.arrayBuffer();
}

async function loadVolume(src: LoadSource): Promise<Volume> {
  const raw = await readSource(src);
  const plain = await gunzipIfNeeded(raw);
  return reorientToRAS(parseNifti(plain));
}

/**
 * Name the file in the error. Parser and gunzip failures say what went wrong
 * but not where, so a case with a dozen structures reported "unexpected EOF"
 * with no way to tell which file to look at.
 */
function blame(key: string, err: unknown): Error {
  const message = err instanceof Error ? err.message : String(err);
  return new Error(`${key}: ${message}`);
}

ctx.onmessage = async (event: MessageEvent<LoadRequest>) => {
  const req = event.data;
  if (req.type !== 'load') return;
  const { token } = req;

  try {
    const tStart = performance.now();
    post({ type: 'progress', token, stage: 'Downloading scan', fraction: 0, detail: req.ct.key });

    let volume: Volume;
    try {
      volume = await loadVolume(req.ct);
    } catch (err) {
      throw blame(req.ct.file?.name ?? req.ct.url ?? req.ct.key, err);
    }
    const normalized = normalizeForGpu(volume);
    const volumeMs = performance.now() - tStart;

    post({ type: 'progress', token, stage: 'Uploading scan', fraction: 0.35 });

    // The scan goes out before any mask is touched, so anatomy is on screen in
    // about a second instead of after the whole case. `normalized` is
    // transferred; `values` is cloned, the mask statistics below still need it.
    post({ type: 'volume', token, volume, normalized, ms: volumeMs }, [normalized.buffer]);

    if (req.masks.length === 0) return;

    const tMasks = performance.now();
    // Fetch concurrently, parse serially: the network is the bottleneck, and
    // parsing two 12 million voxel masks at once only doubles peak memory.
    const buffers = await Promise.all(
      req.masks.map(async (m, i) => {
        const buf = await readSource(m);
        post({
          type: 'progress', token,
          stage: 'Downloading structures',
          fraction: 0.35 + 0.35 * ((i + 1) / req.masks.length),
          detail: m.key,
        });
        return { key: m.key, buf };
      }),
    );

    const masks: MaskInput[] = [];
    for (let i = 0; i < buffers.length; i++) {
      const { key, buf } = buffers[i];
      post({
        type: 'progress', token,
        stage: 'Parsing structures',
        fraction: 0.7 + 0.25 * (i / buffers.length),
        detail: key,
      });
      try {
        const plain = await gunzipIfNeeded(buf);
        const image = parseNifti(plain);
        // One file can hold one structure or many: TotalSegmentator's default
        // output is a combined map of values 1..N, and AbdomenAtlas ships both
        // layouts, so it has to be decided from the data, not the filename.
        masks.push(...expandMaskFile(key, reorientLike(image, volume), undefined));
      } catch (err) {
        throw blame(key, err);
      }
      if (masks.length > 255) {
        throw new Error(
          `This case describes more than 255 structures, which is more than a ` +
          `single-byte label volume can hold.`,
        );
      }
    }

    post({ type: 'progress', token, stage: 'Packing structures', fraction: 0.97 });
    const labels = buildLabelVolume({ reference: volume, masks });
    post({ type: 'labels', token, labels, ms: performance.now() - tMasks }, [labels.values.buffer]);
  } catch (err) {
    post({ type: 'error', token, message: err instanceof Error ? err.message : String(err) });
  }
};
