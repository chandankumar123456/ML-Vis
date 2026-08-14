// src/visualizers/Eigenviewer.tsx
// 2-D data cloud + rotatable candidate axis (registered as 'eigenviewer').
// Reads 'point' / 'axis' / 'projection' visual commands from the current
// snapshot: the axis line through the data centroid, per-point projection
// guides, projected dots on the axis, a 1-D projection strip, and
// variance-explained bars for the axis and its orthogonal complement (the two
// "PCs" of the current rotation). Reconstruction mode projects back to 2-D and
// draws residual error lines to the originals. The slider is a LOCAL override:
// while untouched the view is driven by the snapshot's axis command (or the
// params.angleDeg topic hint); dragging it recomputes everything live.
// Zoom/pan reuse CanvasStage (ScatterPlot convention); pure math lives in
// ./eigenview with its own unit tests.
import { useEffect, useMemo, useRef, useState } from 'react';
import { CanvasStage, type Bounds } from '../lib/canvas/CanvasStage';
import { useContainerSize } from '../lib/canvas/useContainerSize';
import {
  axisUnit, axisExtent, centroid, pointsBounds, resolveAngleDeg, sceneProjections,
  varianceFractions, axisCoordinate, type SceneProjection,
} from './eigenview';
import type { Params, SimState, VisualCommand } from '../engine/types';

const AXIS_DEFAULT = '#64748b';
const GUIDE_COLOR = 'rgba(100,116,139,0.45)';
const RECON_ERROR = '#f43f5e';
const POINT_FALLBACK = '#2563eb';
const DEFAULT_DOMAIN: Bounds = { x: [-7, 7], y: [-7, 7] };
const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

export function Eigenviewer({ snapshot, params }: {
  snapshot?: SimState | null;
  params: Params;
}) {
  const [ref, size] = useContainerSize(600, 400);
  const stageRef = useRef<CanvasStage | null>(null);
  const [overrideDeg, setOverrideDeg] = useState<number | null>(null);
  const [mode, setMode] = useState<'project' | 'reconstruct'>('project');

  // The whole scene (points, colors, effective angle, projections, bars) is a
  // pure memo of (snapshot, params, overrideDeg) — slider drags and snapshot
  // scrubs both invalidate exactly what they must. Deterministic: computed via
  // ./eigenview helpers only.
  const scene = useMemo(() => {
    const visuals = snapshot?.visuals ?? [];
    const pointCmds = visuals.filter((v) => v.type === 'point');
    const axisCmd = visuals.find((v) => v.type === 'axis');
    const projCmds = visuals.filter((v) => v.type === 'projection');
    const points = pointCmds.map((v) => [v.x as number, v.y as number] as [number, number]);
    const angleDeg = resolveAngleDeg(visuals, params, overrideDeg);
    const angle = angleDeg * DEG2RAD;
    const c = centroid(points);
    const u = axisUnit(angle);
    const projections = sceneProjections(points, c, projCmds, angle, overrideDeg);
    const colors = points.map((_, i) => pointColor(pointCmds[i], projections[i]));
    const bars = varianceFractions(points, c, angle).map((b) => ({
      angle: normalizeDeg(b.angle * RAD2DEG),
      fraction: b.fraction,
    }));
    return {
      points, colors, bars, projections, c, u, angleDeg,
      axisColor: typeof axisCmd?.color === 'string' ? axisCmd.color : AXIS_DEFAULT,
    };
  }, [snapshot, params, overrideDeg]);

  const hasPoints = scene.points.length > 0;

  // CanvasStage + zoom/pan wiring, gated on hasPoints so the empty state never
  // mounts a canvas. Re-created on resize (ScatterPlot convention).
  useEffect(() => {
    const host = ref.current;
    if (!host || !hasPoints) { stageRef.current = null; return; }
    const stage = new CanvasStage(size.w, size.h);
    stage.canvas.style.display = 'block';
    stage.canvas.setAttribute('aria-label',
      '2-D data cloud projected onto a rotating candidate axis, with variance bars and 1-D projection strip');
    host.replaceChildren(stage.canvas);
    stageRef.current = stage;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = stage.canvas.getBoundingClientRect();
      stage.zoomAt(e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1.1 : 0.9);
    };
    let panning = false, lx = 0, ly = 0;
    const onDown = (e: PointerEvent) => { panning = true; lx = e.clientX; ly = e.clientY; };
    const onMove = (e: PointerEvent) => {
      if (!panning) return;
      stage.panBy(e.clientX - lx, e.clientY - ly);
      lx = e.clientX; ly = e.clientY;
    };
    const onUp = () => { panning = false; };
    stage.canvas.addEventListener('wheel', onWheel, { passive: false });
    stage.canvas.addEventListener('pointerdown', onDown);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      stage.canvas.removeEventListener('wheel', onWheel);
      stage.canvas.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [size, hasPoints]);

  // Paint the scene: axis through the centroid, then per-point projection
  // guides / reconstruction error lines depending on the display mode.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const b = pointsBounds(scene.points) ?? DEFAULT_DOMAIN;
    stage.setBounds(b, size.w, size.h);
    stage.clear(size.w, size.h, 'transparent');
    const L = axisExtent(scene.points, scene.c);
    stage.drawPath([
      [scene.c[0] - scene.u[0] * L, scene.c[1] - scene.u[1] * L],
      [scene.c[0] + scene.u[0] * L, scene.c[1] + scene.u[1] * L],
    ], scene.axisColor, 1.5);
    scene.projections.forEach((pr, i) => {
      if (mode === 'project') {
        stage.drawPath([pr.from, pr.to], GUIDE_COLOR, 1);
        stage.drawCircle(pr.to[0], pr.to[1], 3, scene.colors[i], '#ffffff');
      } else {
        stage.drawPath([pr.from, pr.to], RECON_ERROR, 1.5);
        stage.drawCircle(pr.to[0], pr.to[1], 3, scene.colors[i], '#ffffff');
        stage.drawCircle(pr.from[0], pr.from[1], 2.5, fade(scene.colors[i]));
      }
    });
    if (mode === 'project') {
      scene.points.forEach((p, i) => stage.drawCircle(p[0], p[1], 4.5, scene.colors[i]));
    }
  }, [scene, size, mode]);

  if (!hasPoints) {
    return (
      <div className="eigenview eigenview-empty" role="status">
        eigenviewer: no data
      </div>
    );
  }

  // strip dot positions: normalized projected coordinate along the axis
  const ts = scene.projections.map((pr) => axisCoordinate(pr.to, scene.c, scene.u));
  const tMin = ts.length > 0 ? Math.min(...ts) : 0;
  const tSpan = (ts.length > 0 ? Math.max(...ts) : 0) - tMin || 1;
  const pct = (t: number) => ((t - tMin) / tSpan) * 100;

  return (
    <div className="eigenview" data-testid="eigenview"
      data-mode={mode} data-angle-deg={scene.angleDeg}
      data-using-snapshot={overrideDeg === null}>
      <div className="eigenview-controls">
        <label className="eigenview-angle">
          <span>axis angle</span>
          <input type="range" min={0} max={180} step={1} value={scene.angleDeg}
            aria-label="candidate axis angle" data-testid="eigen-axis-slider"
            onChange={(e) => setOverrideDeg(Number(e.currentTarget.value))} />
          <span className="eigenview-angle-value">{scene.angleDeg}°</span>
        </label>
        {overrideDeg !== null && (
          <button type="button" data-testid="eigen-follow-button"
            onClick={() => setOverrideDeg(null)}>
            follow topic
          </button>
        )}
        <div className="eigenview-modes" role="group" aria-label="display mode">
          <button type="button" data-testid="eigen-mode-project"
            className={mode === 'project' ? 'active' : ''}
            onClick={() => setMode('project')}>project</button>
          <button type="button" data-testid="eigen-mode-reconstruct"
            className={mode === 'reconstruct' ? 'active' : ''}
            onClick={() => setMode('reconstruct')}>reconstruct</button>
        </div>
      </div>
      <div ref={ref} className="eigenview-canvas" style={{ width: '100%', height: 400 }} />
      <div className="eigenview-strip" data-testid="eigen-strip">
        <span className="eigenview-strip-label">1-D projection</span>
        <div className="eigenview-strip-track">
          {scene.projections.map((pr, i) => {
            const t = axisCoordinate(pr.to, scene.c, scene.u);
            return (
              <span key={i} className="eigenview-strip-dot" data-testid="eigen-strip-dot"
                data-t={tDisplay(t)}
                style={{ left: `${pct(t)}%`, background: scene.colors[i] }} />
            );
          })}
        </div>
      </div>
      <div className="eigenview-bars" data-testid="eigen-bars">
        {scene.bars.map((b, i) => (
          <div key={i} className="eigenview-bar" data-testid="eigen-bar"
            data-angle={b.angle} data-fraction={b.fraction.toFixed(4)}
            data-label={`${b.angle}°`}>
            <span className="eigenview-bar-fill" style={{ width: `${b.fraction * 100}%` }} />
            <span className="eigenview-bar-label">{b.angle}° {Math.round(b.fraction * 100)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function pointColor(pointCmd: VisualCommand | undefined, pr: SceneProjection): string {
  if (typeof pointCmd?.color === 'string') return pointCmd.color;
  if (pr?.color) return pr.color;
  return POINT_FALLBACK;
}

/** Normalize an angle to [0, 180) — the axis is a line, θ ≡ θ + 180°. */
function normalizeDeg(deg: number): number {
  return ((deg % 180) + 180) % 180;
}

/**
 * Strip dot coordinate at 2 decimals, pre-rounded to 4 so floating-point
 * drift from the deg→rad conversion (e.g. 90·π/180 vs Math.PI/2) can never
 * surface as a spurious "-0.00".
 */
function tDisplay(t: number): string {
  return (Math.round(t * 1e4) / 1e4).toFixed(2);
}

/** 35%-alpha variant of a #rrggbb color (used for faded originals). */
function fade(hex: string): string {
  const full = hex.length === 4 ? `#${hex.slice(1).split('').map((c) => c + c).join('')}` : hex;
  if (!/^#[0-9a-fA-F]{6}$/.test(full)) return hex;
  const n = parseInt(full.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, 0.35)`;
}