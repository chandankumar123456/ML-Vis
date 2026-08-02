// src/lib/exporters/recorder.ts
import type { SnapshotRun } from '../../engine/types';
import { snapshotToPng } from './pngExporter';

/**
 * Record a run as a PNG sequence (max 60 frames).
 * Each frame = one snapshot rendered by the provided render callback.
 * GIF/MP4 via WebCodecs arrives in a later wave.
 */
export function recordRun(
  run: SnapshotRun,
  render: (snapshotIndex: number) => HTMLCanvasElement | null,
  maxFrames = 60
): string[] {
  const frames: string[] = [];
  const stride = Math.max(1, Math.ceil(run.snapshots.length / maxFrames));
  for (let i = 0; i < run.snapshots.length; i += stride) {
    const c = render(i);
    // null canvas (e.g. view not mounted) → skip frame, count reflects captured frames
    if (c) frames.push(snapshotToPng(c));
  }
  return frames;
}
