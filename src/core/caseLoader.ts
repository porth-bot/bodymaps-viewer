/**
 * Main-thread side of loading: drives the loader worker and a small mesh
 * worker pool. The pool is capped at three because surface extraction is
 * memory bound, not compute bound (each worker holds its own 12 MB copy of the
 * label volume), so one per core buys little and costs a lot of RAM.
 */

import type { LabelVolume, Mesh, Structure, Volume } from './types';
import type { LoaderMessage, LoadRequest, LoadSource } from '../workers/loader.worker';
import type { MeshBuildRequest, MeshInit, MeshMessage } from '../workers/mesh.worker';

export interface CaseSources {
  name: string;
  ct: LoadSource;
  masks: LoadSource[];
}

export interface CaseCallbacks {
  onProgress(stage: string, fraction: number | null, detail?: string): void;
  onVolume(volume: Volume, normalized: Float32Array, ms: number): void;
  onLabels(labels: LabelVolume, ms: number): void;
  onMesh(mesh: Mesh, ms: number): void;
  onMeshError(index: number, message: string): void;
  onAllMeshes(totalMs: number): void;
  onError(message: string): void;
}

const MAX_MESH_WORKERS = 3;

export class CaseLoader {
  private loader: Worker | null = null;
  private meshWorkers: Worker[] = [];
  private meshQueue: MeshBuildRequest[] = [];
  private busy = new Set<Worker>();
  private token = 0;
  private meshStart = 0;
  private meshOutstanding = 0;

  constructor(private callbacks: CaseCallbacks) {}

  /** Cancels anything in flight and starts a new case. */
  load(sources: CaseSources): void {
    this.cancel();
    this.token++;
    const token = this.token;

    this.loader = new Worker(new URL('../workers/loader.worker.ts', import.meta.url), {
      type: 'module',
    });

    this.loader.onmessage = (event: MessageEvent<LoaderMessage>) => {
      const msg = event.data;
      // Stale messages from a case switched mid-load would overwrite the new scan.
      if (msg.token !== this.token) return;

      switch (msg.type) {
        case 'progress':
          this.callbacks.onProgress(msg.stage, msg.fraction, msg.detail);
          break;
        case 'volume':
          this.callbacks.onVolume(msg.volume, msg.normalized, msg.ms);
          break;
        case 'labels':
          this.callbacks.onLabels(msg.labels, msg.ms);
          break;
        case 'error':
          this.callbacks.onError(msg.message);
          break;
      }
    };

    this.loader.onerror = (e) => {
      this.callbacks.onError(`Loader worker failed: ${e.message}`);
    };

    const request: LoadRequest = { type: 'load', ct: sources.ct, masks: sources.masks, token };
    this.loader.postMessage(request);
  }

  /** Structures go out largest first, so the slowest jobs start earliest and the pool drains evenly. */
  buildMeshes(labels: LabelVolume, structures: Structure[]): void {
    this.stopMeshWorkers();
    const nonEmpty = structures.filter((s) => s.voxelCount > 0);
    if (nonEmpty.length === 0) {
      this.callbacks.onAllMeshes(0);
      return;
    }

    this.meshStart = performance.now();
    const count = Math.max(1, Math.min(MAX_MESH_WORKERS, (navigator.hardwareConcurrency || 4) - 1));

    for (let i = 0; i < count; i++) {
      const worker = new Worker(new URL('../workers/mesh.worker.ts', import.meta.url), {
        type: 'module',
      });
      worker.onmessage = (event: MessageEvent<MeshMessage>) => this.onMeshMessage(worker, event.data);
      worker.onerror = (e) => {
        this.busy.delete(worker);
        this.callbacks.onMeshError(-1, `Mesh worker failed: ${e.message}`);
        this.pump();
      };
      const init: MeshInit = {
        type: 'init',
        dims: labels.dims,
        spacing: labels.spacing,
        // Each worker needs its own copy: SharedArrayBuffer would avoid it but
        // needs cross-origin isolation headers, which GitHub Pages cannot set.
        values: labels.values.slice(),
      };
      worker.postMessage(init, [init.values.buffer]);
      this.meshWorkers.push(worker);
    }

    this.meshQueue = [...nonEmpty]
      .sort((a, b) => b.voxelCount - a.voxelCount)
      .map((s) => ({
        type: 'build' as const,
        index: s.index,
        bounds: s.bounds,
        token: this.token,
      }));
    this.meshOutstanding = this.meshQueue.length;
    this.pump();
  }

  private onMeshMessage(worker: Worker, msg: MeshMessage): void {
    this.busy.delete(worker);
    if (msg.token === this.token) {
      if (msg.type === 'mesh') {
        const mesh: Mesh = {
          index: msg.index,
          positions: msg.positions,
          normals: msg.normals,
          indices: msg.indices,
          bounds: msg.bounds,
          triangleCount: msg.triangleCount,
        };
        this.callbacks.onMesh(mesh, msg.ms);
      } else {
        this.callbacks.onMeshError(msg.index, msg.message);
      }
      this.meshOutstanding--;
      if (this.meshOutstanding <= 0 && this.meshQueue.length === 0) {
        this.callbacks.onAllMeshes(performance.now() - this.meshStart);
      }
    }
    this.pump();
  }

  private pump(): void {
    for (const worker of this.meshWorkers) {
      if (this.busy.has(worker)) continue;
      const job = this.meshQueue.shift();
      if (!job) break;
      this.busy.add(worker);
      worker.postMessage(job);
    }
  }

  private stopMeshWorkers(): void {
    for (const w of this.meshWorkers) w.terminate();
    this.meshWorkers = [];
    this.busy.clear();
    this.meshQueue = [];
    this.meshOutstanding = 0;
  }

  cancel(): void {
    this.token++;
    if (this.loader) {
      this.loader.terminate();
      this.loader = null;
    }
    this.stopMeshWorkers();
  }
}

/** Sources for the sample case bundled with the app. */
export function sampleCase(base: string): CaseSources {
  const structures = [
    'aorta', 'gall_bladder', 'kidney_left', 'kidney_right', 'liver',
    'pancreas', 'postcava', 'spleen', 'stomach',
  ];
  const root = `${base}data/BDMAP_00000338`;
  return {
    name: 'BDMAP_00000338',
    ct: { key: 'ct', url: `${root}/ct.nii.gz` },
    masks: structures.map((key) => ({ key, url: `${root}/segmentations/${key}.nii.gz` })),
  };
}

/**
 * Mirrors how these datasets are actually laid out: one file whose name looks
 * like the scan, everything else a structure named after its file. Anything
 * under a "segmentations" or "masks" folder is a structure whatever its name.
 */
export function sourcesFromFiles(files: File[]): CaseSources | null {
  const nifti = files.filter((f) => /\.nii(\.gz)?$/i.test(f.name));
  if (nifti.length === 0) return null;

  const pathOf = (f: File) => ((f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name);
  const isSegPath = (f: File) => /(^|\/)(segmentations?|masks?|labels?)\//i.test(pathOf(f));
  const looksLikeScan = (f: File) => /(^|\/)(ct|scan|image|volume)\.nii(\.gz)?$/i.test(pathOf(f));

  let ctFile = nifti.find((f) => looksLikeScan(f) && !isSegPath(f));
  if (!ctFile) ctFile = nifti.find((f) => !isSegPath(f));
  // Last resort, the largest file: a scan is int16 and a mask int8 over the
  // same grid, so the scan compresses far worse and is reliably biggest.
  if (!ctFile) ctFile = nifti.reduce((a, b) => (a.size >= b.size ? a : b));

  const masks = nifti
    .filter((f) => f !== ctFile)
    .map((f) => ({ key: f.name.replace(/\.nii(\.gz)?$/i, ''), file: f }));

  const folder = pathOf(ctFile).split('/').slice(0, -1).filter((p) => !/^segmentations?$/i.test(p)).pop();

  return {
    name: folder || ctFile.name.replace(/\.nii(\.gz)?$/i, ''),
    ct: { key: 'ct', file: ctFile },
    masks,
  };
}
