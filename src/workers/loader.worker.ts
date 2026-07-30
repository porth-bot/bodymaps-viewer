/// <reference lib="webworker" />
/**
 * Loads a case off the network or off disk: download, gunzip, parse, reorient,
 * pack labels.
 *
 * All of it happens here rather than on the main thread because gunzipping a
 * 16 MB scan and walking 12 million voxels nine times would freeze the UI for
 * several seconds, and the whole point of the viewer is that it stays
 * responsive while a case loads.
 */

import { gunzipIfNeeded, parseNifti } from '../core/nifti';
import { reorientLike, reorientToRAS } from '../core/orientation';
import { buildLabelVolume, type MaskInput } from '../core/labelmap';
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

ctx.onmessage = async (event: MessageEvent<LoadRequest>) => {
  const req = event.data;
  if (req.type !== 'load') return;
  const { token } = req;

  try {
    const tStart = performance.now();
    post({ type: 'progress', token, stage: 'Downloading scan', fraction: 0, detail: req.ct.key });

    const volume = await loadVolume(req.ct);
    const normalized = normalizeForGpu(volume);
    const volumeMs = performance.now() - tStart;

    post({ type: 'progress', token, stage: 'Uploading scan', fraction: 0.35 });

    // The scan goes out before any mask is touched so the user sees anatomy
    // within a second instead of waiting on the whole case. `normalized` is
    // transferred; `values` is cloned because the mask statistics below still
    // need it here.
    post({ type: 'volume', token, volume, normalized, ms: volumeMs }, [normalized.buffer]);

    if (req.masks.length === 0) return;

    const tMasks = performance.now();
    // Fetch concurrently, parse serially. The files are small and the network
    // is the bottleneck, but parsing two 12 million voxel masks at once just
    // doubles peak memory for no gain.
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
      const plain = await gunzipIfNeeded(buf);
      const image = parseNifti(plain);
      masks.push({ key, values: reorientLike(image, volume) });
    }

    post({ type: 'progress', token, stage: 'Packing structures', fraction: 0.97 });
    const labels = buildLabelVolume({ reference: volume, masks });
    post({ type: 'labels', token, labels, ms: performance.now() - tMasks }, [labels.values.buffer]);
  } catch (err) {
    post({ type: 'error', token, message: err instanceof Error ? err.message : String(err) });
  }
};
