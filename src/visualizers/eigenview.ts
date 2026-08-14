// src/visualizers/eigenview.ts
// Pure, DOM-free math helpers for the eigenviewer registry view (Task 15):
// centroid, axis unit vector, orthogonal projection, residual, projected
// coordinate, variance along an axis, explained-variance fractions for the two
// orthogonal 2-D axes (the "per-PC" bar split), world-space bounds over the
// point cloud, the effective-angle resolution (slider override > snapshot axis
// command > params hint > 0), and the per-point projection scene.
import type { Bounds } from '../lib/canvas/CanvasStage';
import type { Params, VisualCommand } from '../engine/types';

/** Axis extent margin: the drawn axis line spans the cloud × 1.1 so its caps
 * stay clear of the outer points. */
const EXTENT_FACTOR = 1.1;
/** Radians → degrees (used by the angle-resolution + slider labels). */
const RAD2DEG = 180 / Math.PI;

/** Arithmetic mean of the points; [0, 0] for an empty list. */
export function centroid(points: [number, number][]): [number, number] {
  const n = points.length;
  if (n === 0) return [0, 0];
  let sx = 0, sy = 0;
  for (const [x, y] of points) { sx += x; sy += y; }
  return [sx / n, sy / n];
}

/** Unit vector pointing along `angle` (radians, from the +x axis). */
export function axisUnit(angle: number): [number, number] {
  return [Math.cos(angle), Math.sin(angle)];
}

/** Half-length of the candidate axis line around `c` (cloud radius × 1.1). */
export function axisExtent(points: [number, number][], c: [number, number]): number {
  let r = 0;
  for (const [x, y] of points) r = Math.max(r, Math.hypot(x - c[0], y - c[1]));
  return r * EXTENT_FACTOR;
}

/** Orthogonal projection of `p` onto the axis line through `c` along `u`. */
export function projectPoint(p: [number, number], c: [number, number], u: [number, number]): [number, number] {
  const t = (p[0] - c[0]) * u[0] + (p[1] - c[1]) * u[1];
  return [c[0] + t * u[0], c[1] + t * u[1]];
}

/** Perpendicular distance from `p` to the axis line (reconstruction error). */
export function residualLength(p: [number, number], c: [number, number], u: [number, number]): number {
  const q = projectPoint(p, c, u);
  return Math.hypot(p[0] - q[0], p[1] - q[1]);
}

/** Signed projection coordinate t = (p − c)·u along the axis. */
export function axisCoordinate(p: [number, number], c: [number, number], u: [number, number]): number {
  return (p[0] - c[0]) * u[0] + (p[1] - c[1]) * u[1];
}

/** Population variance of the 1-D projections of `points` onto `u`. */
export function varianceAlong(points: [number, number][], c: [number, number], u: [number, number]): number {
  const n = points.length;
  if (n === 0) return 0;
  let sum = 0, sum2 = 0;
  for (const p of points) {
    const t = axisCoordinate(p, c, u);
    sum += t;
    sum2 += t * t;
  }
  return Math.max(0, sum2 / n - (sum / n) ** 2);
}

export interface VarianceBar { angle: number; fraction: number }

/**
 * Explained-variance split between `angle` and its orthogonal complement —
 * the two 2-D "PCs" of the current rotation. Fractions always sum to 1 (the
 * two directions span the plane); a degenerate zero-variance cloud degrades to
 * an even split so the bars never show NaN.
 */
export function varianceFractions(points: [number, number][], c: [number, number], angle: number): VarianceBar[] {
  const u = axisUnit(angle);
  const v1 = varianceAlong(points, c, u);
  const v2 = varianceAlong(points, c, [u[1] * -1, u[0]]);
  const tot = v1 + v2;
  const f1 = tot > 0 ? v1 / tot : 0.5;
  return [
    { angle, fraction: f1 },
    { angle: angle + Math.PI / 2, fraction: 1 - f1 },
  ];
}

/**
 * World-space domain over the point cloud with the ScatterPlot padding
 * convention (10% + 0.5 per axis); null when no points exist.
 */
export function pointsBounds(points: [number, number][]): Bounds | null {
  if (points.length === 0) return null;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [x, y] of points) {
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  const padX = (x1 - x0) * 0.1 + 0.5, padY = (y1 - y0) * 0.1 + 0.5;
  return { x: [x0 - padX, x1 + padX], y: [y0 - padY, y1 + padY] };
}

/**
 * Effective axis angle (degrees, rounded for the integer slider UI).
 * Precedence: user slider override > snapshot axis command (radians) >
 * params.angleDeg hint > 0.
 */
export function resolveAngleDeg(
  visuals: VisualCommand[], params: Params, overrideDeg: number | null,
): number {
  if (overrideDeg !== null && Number.isFinite(overrideDeg)) return Math.round(overrideDeg);
  const axis = visuals.find((v) => v.type === 'axis');
  const rad = axis?.angle;
  if (typeof rad === 'number' && Number.isFinite(rad)) return Math.round(rad * RAD2DEG);
  const hint = params.angleDeg;
  if (typeof hint === 'number' && Number.isFinite(hint)) return Math.round(hint);
  return 0;
}

export interface SceneProjection {
  from: [number, number];
  to: [number, number];
  residual: number;
  color?: string;
}

/**
 * The per-point projection scene. Snapshot `projection` commands are
 * authoritative (they carry the topic's own geometry, e.g. LDA's direction and
 * per-class colors) UNLESS the user is dragging the slider — an override
 * recomputes every projection from the points and the current axis so the view
 * stays live. `angle` is the EFFECTIVE angle in radians.
 */
export function sceneProjections(
  points: [number, number][], c: [number, number],
  projCmds: VisualCommand[], angle: number, overrideDeg: number | null,
): SceneProjection[] {
  const useCommands = overrideDeg === null && projCmds.length > 0;
  if (useCommands) {
    return projCmds.map((cmd) => ({
      from: cmd.point as [number, number],
      to: cmd.onto as [number, number],
      residual: typeof cmd.residual === 'number' ? cmd.residual : 0,
      color: typeof cmd.color === 'string' ? cmd.color : undefined,
    }));
  }
  const u = axisUnit(angle);
  return points.map((p) => ({
    from: p,
    to: projectPoint(p, c, u),
    residual: residualLength(p, c, u),
  }));
}