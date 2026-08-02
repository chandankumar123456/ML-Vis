// src/visualizers/LossCurve.tsx
import { useEffect, useRef, useState } from 'react';
import { CanvasStage, type Bounds } from '../lib/canvas/CanvasStage';
import type { SnapshotRun } from '../engine/types';
import { usePlaybackStore } from '../store/playbackStore';

// IMPORTANT: registered visualizers receive ViewProps ({ run, params, snapshot, subscribe }) from ViewHost.
// Cursor-dependent components must read `cursor` from the playback store, NOT from props (ViewHost does not pass it).
export function LossCurve({ run, metricKey = 'cost' }: {
  run: SnapshotRun | null; metricKey?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const stageRef = useRef<CanvasStage | null>(null);
  const [size, setSize] = useState({ w: 600, h: 300 });
  const cursor = usePlaybackStore((s) => s.cursor);

  useEffect(() => {
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0) setSize({ w: width, h: height });
    });
    if (ref.current) ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

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
    if (cursor < run.snapshots.length) {
      const v = run.snapshots[cursor].metrics[metricKey];
      if (Number.isFinite(v)) stage.drawCircle(cursor, v, 6, '#f59e0b', 'var(--fg)');
    }
  }, [run, cursor, metricKey, size]);

  return <div ref={ref} style={{ width: '100%', height: 300 }} />;
}
