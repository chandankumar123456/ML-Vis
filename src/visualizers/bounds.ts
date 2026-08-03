// src/visualizers/bounds.ts
// Pure, DOM-free helpers shared by the decision-boundary renderer:
//  - boundsOfVisuals: fits the world-space domain over a snapshot's visual
//    commands (points/lines/arrows/circles), mirroring ScatterPlot's convention.
//  - safeClassify: calls a topic classifier without ever throwing, so a
//    misbehaving topic can never crash the grid render.
import type { Bounds } from '../lib/canvas/CanvasStage';
import type { Classifier } from '../registry/viewRegistry';
import type { Params, VisualCommand } from '../engine/types';

/**
 * Conservative world-space domain covering every plottable visual command.
 * Circles contribute their full extent (x±r, y±r) so distance rings are not
 * clipped by a domain fit to their centers alone. Each axis is padded by
 * 10% + 0.5 (the ScatterPlot convention) so glyphs sit clear of the canvas
 * edge. Returns null when nothing plottable exists (callers fall back to a
 * default domain).
 *
 * NOTE: DecisionBoundary's empty-case contract is `null` (→ default domain),
 * while ScatterPlot's boundsOf returns [0,1]×[0,1] — the divergence is why the
 * two files still carry near-identical loops instead of sharing this helper.
 */
export function boundsOfVisuals(cmds: VisualCommand[]): Bounds | null {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  const touch = (x: number, y: number) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  };
  for (const c of cmds) {
    if (c.type === 'point') touch(c.x as number, c.y as number);
    if (c.type === 'line' && Array.isArray(c.points)) {
      for (const [x, y] of c.points as [number, number][]) touch(x, y);
    }
    if (c.type === 'arrow') { touch(c.x1 as number, c.y1 as number); touch(c.x2 as number, c.y2 as number); }
    if (c.type === 'circle') {
      const r = c.r as number;
      touch((c.x as number) - r, (c.y as number) - r);
      touch((c.x as number) + r, (c.y as number) + r);
    }
  }
  if (!Number.isFinite(x0)) return null;
  const padX = (x1 - x0) * 0.1 + 0.5, padY = (y1 - y0) * 0.1 + 0.5;
  return { x: [x0 - padX, x1 + padX], y: [y0 - padY, y1 + padY] };
}

/**
 * Classify a world-space point without ever throwing: an undefined classifier
 * and any classifier exception both degrade to class 0 (the default background
 * class), so a throwing topic cannot crash the whole canvas render. Non-finite
 * results are treated the same way.
 */
export function safeClassify(classifier: Classifier | undefined, x: number, y: number, params: Params): number {
  if (!classifier) return 0;
  try {
    const cls = classifier(x, y, params);
    return Number.isFinite(cls) ? cls : 0;
  } catch {
    return 0;
  }
}
