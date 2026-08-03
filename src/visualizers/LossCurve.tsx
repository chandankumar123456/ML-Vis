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

// series colors: series 1 keeps the original single-line blue; series 2 is a
// distinct green so the two loss curves (train vs test) read apart.
const LOSS_SERIES_COLORS = ['#3b82f6', '#22c55e'] as const;
// bar heights normalize against this fixed pixel budget (labels + legend take
// the rest of the 300px container — no layout measurement needed).
const LOSS_BAR_MAX = 120;

export function LossCurve({ run, metricKey = 'cost', metricKey2 }: {
  run: SnapshotRun | null; metricKey?: string; metricKey2?: string;
}) {
  const [ref, size] = useContainerSize(600, 300);
  const plotRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<CanvasStage | null>(null);
  const cursor = usePlaybackStore((s) => s.cursor);

  // ≤2 snapshots + a second metric → grouped train/test bars; otherwise lines.
  // Bar mode only exists for the two-series case — a single metricKey keeps the
  // original line behavior even on ≤2-snapshot runs.
  const snapCount = run?.snapshots.length ?? 0;
  const barMode = Boolean(metricKey2) && snapCount > 0 && snapCount <= 2;

  useEffect(() => {
    // plotRef only exists in line mode (bar mode renders LossBars instead), so
    // no canvas is ever created for bar runs.
    const host = plotRef.current;
    if (!host) return;
    stageRef.current = new CanvasStage(size.w, size.h);
    stageRef.current.canvas.style.display = 'block';
    host.replaceChildren(stageRef.current.canvas);
  }, [size, barMode]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !run || barMode) return;
    const values = run.snapshots.map((s) => s.metrics[metricKey] ?? NaN);
    const values2 = metricKey2 ? run.snapshots.map((s) => s.metrics[metricKey2] ?? NaN) : null;
    const finite = values.filter(Number.isFinite);
    const finite2 = values2 ? values2.filter(Number.isFinite) : [];
    // single-key: identical to the original `finite.length < 2` early-return;
    // two-series: draw when EITHER series has a plottable line.
    if (finite.length < 2 && finite2.length < 2) return;
    const y = finite.concat(finite2);
    const b: Bounds = { x: [0, values.length - 1], y: [Math.min(...y), Math.max(...y)] };
    stage.setBounds(b, size.w, size.h);
    stage.clear(size.w, size.h, 'transparent');
    stage.drawPath(values.map((v, i) => [i, v] as [number, number]), LOSS_SERIES_COLORS[0], 2);
    if (values2) {
      stage.drawPath(values2.map((v, i) => [i, v] as [number, number]), LOSS_SERIES_COLORS[1], 2);
    }
    const fg = cssVar('--fg', '#0f172a');
    if (cursor < run.snapshots.length) {
      const v = run.snapshots[cursor].metrics[metricKey];
      if (Number.isFinite(v)) stage.drawCircle(cursor, v, 6, '#f59e0b', fg);
    }
  }, [run, cursor, metricKey, metricKey2, size, barMode]);

  return (
    <div ref={ref} style={{ width: '100%', height: 300, position: 'relative' }} data-loss-mode={barMode ? 'bars' : 'line'}>
      {barMode ? (
        <LossBars run={run} metricKey={metricKey} metricKey2={metricKey2} />
      ) : (
        <div ref={plotRef} style={{ width: '100%', height: '100%' }} />
      )}
      {!barMode && metricKey2 && (
        <div className="loss-legend loss-legend-overlay">
          <span className="loss-legend-item">
            <span className="loss-legend-swatch" style={{ background: LOSS_SERIES_COLORS[0] }} />{metricKey}
          </span>
          <span className="loss-legend-item">
            <span className="loss-legend-swatch" style={{ background: LOSS_SERIES_COLORS[1] }} />{metricKey2}
          </span>
        </div>
      )}
    </div>
  );
}

/** Grouped bar fallback for single-shot runs: one train/test pair per snapshot. */
function LossBars({ run, metricKey, metricKey2 }: {
  run: SnapshotRun | null; metricKey: string; metricKey2?: string;
}) {
  if (!run) return null;
  const key2 = metricKey2 ?? '';
  const rows = run.snapshots.map((s) => [s.metrics[metricKey] ?? NaN, s.metrics[key2] ?? NaN] as const);
  const max = Math.max(...rows.flat().filter(Number.isFinite), 1e-9);
  return (
    <div className="loss-bars">
      <div className="loss-bars-chart">
        {rows.map(([v1, v2], i) => (
          <div key={i} className="loss-bar-group">
            <div className="loss-bar-pair">
              <div className="loss-bar-col">
                <span className="loss-bar-value">{Number.isFinite(v1) ? v1.toFixed(2) : '—'}</span>
                <div className="loss-bar" data-key={metricKey} data-value={v1}
                  style={{ height: barHeight(v1, max), background: LOSS_SERIES_COLORS[0] }} />
              </div>
              <div className="loss-bar-col">
                <span className="loss-bar-value">{Number.isFinite(v2) ? v2.toFixed(2) : '—'}</span>
                <div className="loss-bar" data-key={key2} data-value={v2}
                  style={{ height: barHeight(v2, max), background: LOSS_SERIES_COLORS[1] }} />
              </div>
            </div>
            <span className="loss-step">step {i}</span>
          </div>
        ))}
      </div>
      <div className="loss-legend">
        <span className="loss-legend-item">
          <span className="loss-legend-swatch" style={{ background: LOSS_SERIES_COLORS[0] }} />{metricKey}
        </span>
        <span className="loss-legend-item">
          <span className="loss-legend-swatch" style={{ background: LOSS_SERIES_COLORS[1] }} />{metricKey2}
        </span>
      </div>
    </div>
  );
}

function barHeight(v: number, max: number): number {
  return Number.isFinite(v) && v > 0 ? Math.max(2, (v / max) * LOSS_BAR_MAX) : 0;
}
