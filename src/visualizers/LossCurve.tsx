// src/visualizers/LossCurve.tsx
import { useEffect, useRef } from 'react';
import { CanvasStage, cssVar, type Bounds } from '../lib/canvas/CanvasStage';
import { useContainerSize } from '../lib/canvas/useContainerSize';
import type { SnapshotRun } from '../engine/types';
import { usePlaybackStore } from '../store/playbackStore';

// IMPORTANT: registered visualizers receive ViewProps ({ run, params, snapshot, subscribe }) from ViewHost.
// Cursor-dependent components must read `cursor` from the playback store, NOT from props (ViewHost does not pass it).
// Intentionally NOT a ViewProps component: mounted by wrapper views (e.g.
// registerView('loss', (p) => <LossCurve run={p.run} metricKey="cost" />)),
// so it takes explicit props instead of ViewProps.
export function LossCurve({ run, metricKey = 'cost' }: {
  run: SnapshotRun | null; metricKey?: string;
}) {
  const [ref, size] = useContainerSize(600, 300);
  const stageRef = useRef<CanvasStage | null>(null);
  const cursor = usePlaybackStore((s) => s.cursor);

  useEffect(() => {
    if (!ref.current) return;
    stageRef.current = new CanvasStage(size.w, size.h);
    stageRef.current.canvas.style.display = 'block';
    ref.current.replaceChildren(stageRef.current.canvas);
  }, [size]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !run) return;
    const values = run.snapshots.map((s) => s.metrics[metricKey] ?? NaN);
    const finite = values.filter(Number.isFinite);
    if (finite.length < 2) return;
    const b: Bounds = { x: [0, values.length - 1], y: [Math.min(...finite), Math.max(...finite)] };
    stage.setBounds(b, size.w, size.h);
    stage.clear(size.w, size.h, 'transparent');
    stage.drawPath(values.map((v, i) => [i, v] as [number, number]), '#3b82f6', 2);
    const fg = cssVar('--fg', '#0f172a');
    if (cursor < run.snapshots.length) {
      const v = run.snapshots[cursor].metrics[metricKey];
      if (Number.isFinite(v)) stage.drawCircle(cursor, v, 6, '#f59e0b', fg);
    }
  }, [run, cursor, metricKey, size]);

  return <div ref={ref} style={{ width: '100%', height: 300 }} />;
}
