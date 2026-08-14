// src/visualizers/ClusterAnimator.tsx
// K-means step renderer (registered as 'cluster-animator'). Reads the current
// snapshot's 'point' / 'centroid' / 'assignment' visual commands: data points,
// centroids, a thin line from each assigned point to its centroid (the point
// is colored with the centroid's color), and a loss readout ("J = 123.45")
// resolved from snapshot.metrics (preferring topic.lossMetricKey, then any
// loss/cost/j-prefixed key) with a {type:'text'} command fallback.
//
// "Animated convergence" is purely snapshot-driven: topics animate by emitting
// consecutive snapshots with stable centroid ids; the view records each
// centroid's position after every draw in a ref and renders a faint trail at
// the PREVIOUS snapshot's positions, so scrubbing the run shows the centroids
// migrating toward their final clusters.
//
// Defensive: NaN/∞ points/centroids/assignment points are skipped, assignment
// commands whose centroid id is missing are skipped entirely, and a snapshot
// with no plottable points/centroids renders an empty state.
import { useEffect, useMemo, useRef } from 'react';
import { fitBounds, type Bounds } from '../lib/canvas/CanvasStage';
import { useContainerSize } from '../lib/canvas/useContainerSize';
import type { Params, SimState, TopicModule } from '../engine/types';

const PAD = 40;
const POINT_R = 4.5;
const CENTROID_R = 6;
const TRAIL_R = 3;
const FALLBACK_POINT = '#2563eb';
const FALLBACK_CENTROID = '#dc2626';
const CENTROID_STROKE = '#ffffff';
const TRAIL_COLOR = 'rgba(148,163,184,0.45)';

export interface ClusterScenePoint {
  id?: string;
  x: number;
  y: number;
  color?: string;
}

export interface ClusterSceneCentroid {
  id: string;
  x: number;
  y: number;
  color?: string;
}

export interface ClusterSceneAssignment {
  point: [number, number];
  centroidId: string;
  color?: string;
}

export interface ClusterScene {
  points: ClusterScenePoint[];
  centroids: ClusterSceneCentroid[];
  assignments: ClusterSceneAssignment[];
  loss: { text: string } | null;
}

/**
 * Pure scene builder: filters point/centroid commands with non-finite coords,
 * keeps assignments whose point is finite (centroid resolution happens at draw
 * time so dangling centroid ids stay tolerated), and resolves the loss
 * readout. Deterministic and DOM-free so it can be unit-tested directly.
 */
export function buildClusterScene(snapshot?: SimState | null, topic?: TopicModule): ClusterScene {
  if (!snapshot) return { points: [], centroids: [], assignments: [], loss: null };
  const points: ClusterScenePoint[] = [];
  const centroids: ClusterSceneCentroid[] = [];
  const assignments: ClusterSceneAssignment[] = [];
  for (const cmd of snapshot.visuals) {
    if (cmd.type === 'point') {
      const x = cmd.x as number;
      const y = cmd.y as number;
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      points.push({
        id: typeof cmd.id === 'string' ? cmd.id : undefined,
        x, y,
        color: typeof cmd.color === 'string' ? cmd.color : undefined,
      });
    } else if (cmd.type === 'centroid') {
      const x = cmd.x as number;
      const y = cmd.y as number;
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (typeof cmd.id !== 'string') continue; // assignments need a stable id
      centroids.push({
        id: cmd.id,
        x, y,
        color: typeof cmd.color === 'string' ? cmd.color : undefined,
      });
    } else if (cmd.type === 'assignment') {
      const point = cmd.point;
      if (!Array.isArray(point) || point.length < 2) continue;
      const px = point[0] as number;
      const py = point[1] as number;
      if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      if (typeof cmd.centroidId !== 'string') continue;
      assignments.push({
        point: [px, py],
        centroidId: cmd.centroidId,
        color: typeof cmd.color === 'string' ? cmd.color : undefined,
      });
    }
  }
  return { points, centroids, assignments, loss: resolveLoss(snapshot, topic) };
}

/**
 * Resolve the loss caption. Precedence: the topic's lossMetricKey (when the
 * metric exists and is finite), then the first metrics key starting with
 * loss/cost/j, then the first {type:'text'} command. Returns null when the
 * snapshot carries nothing usable.
 */
export function resolveLoss(
  snapshot: SimState | null | undefined,
  topic?: TopicModule,
): { text: string } | null {
  if (!snapshot) return null;
  const metrics = snapshot.metrics ?? {};
  if (topic?.lossMetricKey) {
    const v = metrics[topic.lossMetricKey];
    if (typeof v === 'number' && Number.isFinite(v)) {
      return { text: `${prettifyMetric(topic.lossMetricKey)} = ${v.toFixed(2)}` };
    }
  }
  for (const key of Object.keys(metrics)) {
    if (/^(loss|cost|j)/i.test(key)) {
      const v = metrics[key];
      if (typeof v === 'number' && Number.isFinite(v)) {
        return { text: `${prettifyMetric(key)} = ${v.toFixed(2)}` };
      }
    }
  }
  for (const cmd of snapshot.visuals) {
    if (cmd.type === 'text' && typeof cmd.text === 'string') {
      return { text: cmd.text };
    }
  }
  return null;
}

/** 'j' → 'J', 'loss' → 'Loss', 'inertia' → 'Inertia'. */
function prettifyMetric(key: string): string {
  return key.length === 0 ? key : key[0].toUpperCase() + key.slice(1);
}

/**
 * Fit a padded world-space domain around points + centroids (ScatterPlot-style
 * 10% + 0.5 padding); falls back to a symmetric default window when there is
 * nothing plottable.
 */
export function clusterBounds(
  points: [number, number][],
  centroids: [number, number][],
): Bounds {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  const touch = (x: number, y: number) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  };
  for (const p of points) touch(p[0], p[1]);
  for (const c of centroids) touch(c[0], c[1]);
  if (!Number.isFinite(x0)) return { x: [-7, 7], y: [-7, 7] };
  const padX = (x1 - x0) * 0.1 + 0.5;
  const padY = (y1 - y0) * 0.1 + 0.5;
  return { x: [x0 - padX, x1 + padX], y: [y0 - padY, y1 + padY] };
}

export function ClusterAnimator({ snapshot, params, topic }: {
  snapshot?: SimState | null;
  params: Params;
  topic?: TopicModule;
}) {
  const [ref, size] = useContainerSize(600, 400);
  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<{ host: HTMLDivElement; canvas: HTMLCanvasElement } | null>(null);
  // Last-rendered centroid positions keyed by centroid id: the faint
  // convergence trail is drawn from THIS map, then the map is replaced with
  // the current snapshot's positions — so the next snapshot's draw shows the
  // centroids where they were one step earlier.
  const trailRef = useRef<Map<string, [number, number]>>(new Map());

  const scene = useMemo(() => buildClusterScene(snapshot, topic), [snapshot, topic]);
  const hasPlottable = scene.points.length > 0 || scene.centroids.length > 0;

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !hasPlottable) return;
    const dpr = window.devicePixelRatio || 1;
    let canvas = canvasRef.current?.host === host ? canvasRef.current.canvas : null;
    if (!canvas) {
      const created = document.createElement('canvas');
      created.className = 'cluster-canvas';
      created.setAttribute('data-testid', 'cluster-canvas');
      created.setAttribute('aria-label',
        'k-means step: data points, centroids, assignment lines and a loss readout');
      created.style.display = 'block';
      host.replaceChildren(created);
      canvasRef.current = { host, canvas: created };
      canvas = created;
    }
    canvas.width = Math.max(1, size.w * dpr);
    canvas.height = Math.max(1, size.h * dpr);
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${size.h}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    draw(ctx, scene, trailRef.current, size, dpr);
    // record current centroid positions AFTER drawing (params in deps by
    // ScatterPlot convention — redraw on param change)
    const next = new Map<string, [number, number]>();
    for (const c of scene.centroids) next.set(c.id, [c.x, c.y]);
    trailRef.current = next;
  }, [scene, size, params, hasPlottable]);

  if (!hasPlottable) {
    return (
      <div className="cluster-animator cluster-empty" role="status">
        cluster-animator: no data
      </div>
    );
  }

  return (
    <div ref={ref} className="cluster-animator" data-testid="cluster-animator"
      data-point-count={scene.points.length} data-centroid-count={scene.centroids.length}
      style={{ width: '100%', height: 400, position: 'relative' }}>
      <div ref={hostRef} style={{ width: '100%', height: '100%' }} />
      {scene.loss && (
        <div className="cluster-loss" data-testid="cluster-loss" data-loss-text={scene.loss.text}
          style={{ position: 'absolute', top: 4, left: 8 }}>
          {scene.loss.text}
        </div>
      )}
    </div>
  );
}

function draw(ctx: CanvasRenderingContext2D, scene: ClusterScene,
  trail: Map<string, [number, number]>,
  size: { w: number; h: number }, dpr: number): void {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size.w, size.h);

  // Fit the domain over everything visible INCLUDING the previous centroids
  // (trail marks must never fall outside the padded canvas).
  const trailPts = [...trail.values()];
  const t = fitBounds(clusterBounds(
    [
      ...scene.points.map((p) => [p.x, p.y] as [number, number]),
      ...scene.assignments.map((a) => a.point),
    ],
    [...scene.centroids.map((c) => [c.x, c.y] as [number, number]), ...trailPts],
  ), size.w, size.h, PAD);
  const w2s = (x: number, y: number): [number, number] => [x * t.scale + t.tx, y * t.scale + t.ty];
  const centroidById = new Map(scene.centroids.map((c) => [c.id, c]));

  // faint trail at the PREVIOUS snapshot's centroid positions
  for (const [, [tx, ty]] of trail) {
    const [sx, sy] = w2s(tx, ty);
    ctx.beginPath();
    ctx.arc(sx, sy, TRAIL_R, 0, Math.PI * 2);
    ctx.fillStyle = TRAIL_COLOR;
    ctx.fill();
  }

  // assignment lines + the assigned point colored with its centroid's color
  for (const a of scene.assignments) {
    const cent = centroidById.get(a.centroidId);
    if (!cent) continue; // missing/dangling centroid — skip the assignment
    const [px, py] = w2s(a.point[0], a.point[1]);
    const [cx, cy] = w2s(cent.x, cent.y);
    const color = a.color ?? cent.color ?? FALLBACK_CENTROID;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(cx, cy);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(px, py, POINT_R, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  // data points
  for (const p of scene.points) {
    const [sx, sy] = w2s(p.x, p.y);
    ctx.beginPath();
    ctx.arc(sx, sy, POINT_R, 0, Math.PI * 2);
    ctx.fillStyle = p.color ?? FALLBACK_POINT;
    ctx.fill();
  }

  // centroids: filled circle + white outline ring
  for (const c of scene.centroids) {
    const [sx, sy] = w2s(c.x, c.y);
    const color = c.color ?? FALLBACK_CENTROID;
    ctx.beginPath();
    ctx.arc(sx, sy, CENTROID_R, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(sx, sy, CENTROID_R, 0, Math.PI * 2);
    ctx.strokeStyle = CENTROID_STROKE;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}
