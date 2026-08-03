// src/visualizers/DecisionBoundary.tsx
// 2D class-region renderer (registered as 'decision-boundary'). Samples a
// topic-registered classifier — viewRegistry.registerClassifier(topicId, fn)
// where fn = (x, y, params) => class index — on an offscreen GRID×GRID lattice,
// paints it as ImageData, then upscales onto the visible canvas with drawImage.
// Overlays: a solid decision line, optional dashed margin lines parallel to it,
// and optional highlighted support vectors (used by later SVM topics).
import { useEffect, useMemo, useRef } from 'react';
import { useContainerSize } from '../lib/canvas/useContainerSize';
import { cssVar, fitBounds, type Bounds } from '../lib/canvas/CanvasStage';
import { getClassifier, type Classifier } from '../registry/viewRegistry';
import { boundsOfVisuals, safeClassify } from './bounds';
import type { Params, SimState, TopicModule } from '../engine/types';

// Offscreen lattice resolution: GRID² classifier calls per redraw, then one
// drawImage upscale — keeps per-frame cost constant regardless of canvas size.
const GRID = 50;
// Overlay breathing room (pixels) — matches CanvasStage.fitBounds' default pad
// so the decision line / support vectors sit clear of the canvas edges.
const OVERLAY_PAD = 40;
// Domain used when a snapshot has no plottable data (or no snapshot at all):
// a symmetric window around the origin, matching the default scatter extent.
const DEFAULT_DOMAIN: Bounds = { x: [-7, 7], y: [-7, 7] };
// Overlay colors are fixed hex — canvas cannot consume CSS var() directly.
const DECISION_COLOR = '#3b82f6';
const MARGIN_COLOR = '#f59e0b';
const SV_FILL = '#f59e0b';
const SV_STROKE = '#ffffff';
// Classes 2+ cycle this fixed palette (class 0/1 resolve the --cat1/--cat2 CSS
// vars at draw time so the colorblind palette overrides keep working).
const EXTRA_CLASS_COLORS: [number, number, number][] = [
  [22, 163, 74], // #16a34a
  [147, 51, 234], // #9333ea
  [245, 158, 11], // #f59e0b
  [8, 145, 178], // #0891b2
];

export function DecisionBoundary({ snapshot, params, topic, supportVectors, marginLines }: {
  snapshot?: SimState | null;
  params: Params;
  topic?: TopicModule;
  supportVectors?: [number, number][];
  marginLines?: { offset: number }[];
}) {
  const [ref, size] = useContainerSize(600, 400);
  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<{ host: HTMLDivElement; canvas: HTMLCanvasElement } | null>(null);

  const classifier = getClassifier(topic?.id ?? '');
  const hasClassifier = classifier !== undefined;

  // Merge the topic params with the snapshot's algorithm state ONCE per
  // snapshot, reusing the exact same object for every grid cell and overlay
  // probe — topic classifiers may cache fit state keyed on the params reference
  // (e.g. naive-bayes), so rebuilding it per cell would miss that cache every
  // call. Only a new snapshot or changed params produce a new object.
  const effParams = useMemo(
    () => (snapshot ? { ...params, ...snapshot.algorithm } : params),
    [snapshot, params]
  );

  // World-space domain shared by the grid lattice and the overlay transform.
  const bounds = useMemo(() => resolveBounds(snapshot), [snapshot]);

  // Classify the whole GRID×GRID lattice once per (snapshot, classifier,
  // merged params): scrubbing or panel re-renders reuse the cached grid instead
  // of recomputing 2500 classifications per frame. Invalidates only when the
  // displayed snapshot, the registered classifier, or the merged params change.
  const grid = useMemo(() => {
    if (!classifier) return null;
    const [x0, x1] = bounds.x;
    const [y0, y1] = bounds.y;
    const cells = new Float64Array(GRID * GRID);
    for (let iy = 0; iy < GRID; iy++) {
      // cell centers — never sample exactly on the domain edge
      const y = y0 + ((iy + 0.5) / GRID) * (y1 - y0);
      for (let ix = 0; ix < GRID; ix++) {
        const x = x0 + ((ix + 0.5) / GRID) * (x1 - x0);
        cells[iy * GRID + ix] = safeClassify(classifier, x, y, effParams);
      }
    }
    return cells;
  }, [snapshot, classifier, effParams]);

  // Paint whenever the snapshot, params, classifier, overlays or size change.
  // The grid is memoized above, so re-paints (e.g. panel re-renders, resize)
  // reuse the cached classifications instead of re-running the classifier per
  // cell. The classifier receives the CURRENT topic params with the snapshot's
  // algorithm state merged in, so classification reflects the step the viewer
  // is scrubbing to.
  useEffect(() => {
    const host = hostRef.current;
    if (!host || !classifier || !grid) return;
    const dpr = window.devicePixelRatio || 1;
    let canvas = canvasRef.current?.host === host ? canvasRef.current.canvas : null;
    if (canvas === null) {
      const created = document.createElement('canvas');
      created.className = 'decision-canvas';
      created.setAttribute('data-decision-grid', String(GRID));
      created.setAttribute('aria-label',
        'decision boundary classification grid: 50 × 50 sampled class regions');
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
    draw(ctx, classifier, bounds, effParams, grid, supportVectors, marginLines, size);
  }, [snapshot, params, size, classifier, bounds, effParams, grid, supportVectors, marginLines]);

  if (!hasClassifier) {
    return (
      <div className="decision-boundary decision-empty" role="status">
        decision-boundary: no classifier
      </div>
    );
  }

  return (
    <div ref={ref} className="decision-boundary" style={{ width: '100%', height: 400 }}>
      <div ref={hostRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}

function draw(ctx: CanvasRenderingContext2D, classifier: Classifier, bounds: Bounds,
  effParams: Params, grid: Float64Array,
  supportVectors: [number, number][] | undefined,
  marginLines: { offset: number }[] | undefined,
  size: { w: number; h: number }): void {
  const dpr = window.devicePixelRatio || 1;
  const [x0, x1] = bounds.x;
  const [y0, y1] = bounds.y;

  // ---- class regions: GRID×GRID ImageData upscaled to the full canvas ----
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, size.w * dpr, size.h * dpr);
  const off = document.createElement('canvas');
  off.width = GRID;
  off.height = GRID;
  const offCtx = off.getContext('2d');
  if (!offCtx) return;
  const img = new ImageData(GRID, GRID);
  const palette = resolvePalette();
  for (let iy = 0; iy < GRID; iy++) {
    for (let ix = 0; ix < GRID; ix++) {
      const cls = grid[iy * GRID + ix];
      const [r, g, b] = classColor(palette, cls);
      const o = (iy * GRID + ix) * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = 255;
    }
  }
  offCtx.putImageData(img, 0, 0);
  // Nearest-neighbor upscale: with smoothing left on, the GRID² region image
  // blurs into gradients between adjacent class cells.
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, GRID, GRID, 0, 0, size.w * dpr, size.h * dpr);
  ctx.imageSmoothingEnabled = true;

  // ---- overlays in CSS-pixel space (padded transform, like ScatterPlot) ----
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const t = fitBounds(bounds, size.w, size.h, OVERLAY_PAD);
  const w2s = (x: number, y: number): [number, number] => [x * t.scale + t.tx, y * t.scale + t.ty];

  const bl = boundaryLine(classifier, bounds, effParams);
  if (bl) {
    const [px, py] = w2s(bl.p[0], bl.p[1]);
    const [dx, dy] = bl.d;
    // half-diagonal span — long enough that the line exits the domain box
    const r = Math.hypot((x1 - x0) / 2, (y1 - y0) / 2);
    drawLine(ctx, px - dx * r, py - dy * r, px + dx * r, py + dy * r, DECISION_COLOR, 2);
    if (marginLines) {
      for (const m of marginLines) {
        // parallel line offset along the boundary normal (−dy, dx)
        const ox = -dy * m.offset;
        const oy = dx * m.offset;
        drawLine(ctx, px + ox - dx * r, py + oy - dy * r, px + ox + dx * r, py + oy + dy * r,
          MARGIN_COLOR, 1.5, [6, 4]);
      }
    }
  }
  if (supportVectors) {
    for (const [sx, sy] of supportVectors) {
      const [cx, cy] = w2s(sx, sy);
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fillStyle = SV_FILL;
      ctx.fill();
      ctx.strokeStyle = SV_STROKE;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}

function drawLine(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number,
  color: string, width: number, dash?: number[]): void {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash ?? []);
  ctx.stroke();
}

// ---- domain inference ----
// Follow ScatterPlot's convention: fit the visible data (points/lines/arrows/
// circles) with the same 10% + 0.5 padding, so class regions align with the
// data on the shared world-space convention; fall back to DEFAULT_DOMAIN when
// the snapshot has nothing plottable. Bounds fitting lives in ./bounds
// (boundsOfVisuals) — shared with the regression tests and unit-testable
// without a DOM. Kept consistent with scatter-plot by design.
function resolveBounds(snapshot: SimState | null | undefined): Bounds {
  const b = snapshot ? boundsOfVisuals(snapshot.visuals) : null;
  return b ?? DEFAULT_DOMAIN;
}

// ---- class colors ----
function resolvePalette(): [number, number, number][] {
  return [
    hexToRgb(cssVar('--cat1', '#2563eb')) ?? [37, 99, 235],
    hexToRgb(cssVar('--cat2', '#dc2626')) ?? [220, 38, 38],
    ...EXTRA_CLASS_COLORS,
  ];
}

// Maps a classifier output to a palette entry: class index 0/1 use the two
// categorical vars; non-finite or negative outputs clamp to class 0; classes 2+
// cycle the fixed extra palette.
function classColor(palette: [number, number, number][], cls: number): [number, number, number] {
  const ci = Number.isFinite(cls) && cls >= 0 ? Math.floor(cls) : 0;
  if (ci <= 1) return palette[ci];
  return palette[2 + ((ci - 2) % (palette.length - 2))];
}

function hexToRgb(hex: string): [number, number, number] | null {
  const six = /^#([0-9a-fA-F]{6})$/.exec(hex);
  if (six) {
    const v = [parseInt(six[1].slice(0, 2), 16), parseInt(six[1].slice(2, 4), 16), parseInt(six[1].slice(4, 6), 16)];
    return [v[0], v[1], v[2]];
  }
  const three = /^#([0-9a-fA-F]{3})$/.exec(hex);
  if (three) {
    const v = three[1].split('').map((c) => parseInt(c + c, 16));
    return [v[0], v[1], v[2]];
  }
  return null;
}

// ---- boundary fitting for overlays ----
// Fits the classifier's decision boundary to a line: one boundary point from a
// midline scan + the boundary normal from a local gradient probe. Supports both
// hard (class 0/1) and signed-distance classifiers; returns null when no
// boundary crosses the visible domain (a uniform region), so callers skip the
// line overlays.
function boundaryLine(classifier: Classifier, bounds: Bounds, params: Params):
  { p: [number, number]; d: [number, number] } | null {
  const p = boundaryPoint(classifier, bounds, params);
  if (!p) return null;
  const n = boundaryNormal(classifier, p, bounds, params);
  if (!n) return null;
  return { p, d: [n[1], -n[0]] };
}

function boundaryPoint(classifier: Classifier, bounds: Bounds, params: Params): [number, number] | null {
  const [x0, x1] = bounds.x;
  const [y0, y1] = bounds.y;
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const STEPS = 101;
  const tH = scanCrossing((t: number) => safeClassify(classifier, x0 + (x1 - x0) * t, cy, params), STEPS);
  if (tH !== null) return [x0 + (x1 - x0) * tH, cy];
  const tV = scanCrossing((t: number) => safeClassify(classifier, cx, y0 + (y1 - y0) * t, params), STEPS);
  if (tV !== null) return [cx, y0 + (y1 - y0) * tV];
  return null;
}

// First sign-flip of f over t ∈ [0, 1]; returns the crossing's midpoint
// (null when f never flips sign in the domain).
function scanCrossing(f: (t: number) => number, steps: number): number | null {
  const s = (v: number) => (v < 0 ? -1 : v > 0 ? 1 : 0);
  let prev = s(f(0));
  for (let i = 1; i <= steps; i++) {
    const cur = s(f(i / steps));
    if (prev !== cur) return (i - 0.5) / steps;
    prev = cur;
  }
  return null;
}

function boundaryNormal(classifier: Classifier, p: [number, number], bounds: Bounds, params: Params):
  [number, number] | null {
  const [x0, x1] = bounds.x;
  const [y0, y1] = bounds.y;
  const eps = Math.max(x1 - x0, y1 - y0) * 1e-2;
  const gx = safeClassify(classifier, p[0] + eps, p[1], params) - safeClassify(classifier, p[0] - eps, p[1], params);
  const gy = safeClassify(classifier, p[0], p[1] + eps, params) - safeClassify(classifier, p[0], p[1] - eps, params);
  const len = Math.hypot(gx, gy);
  if (!Number.isFinite(len) || len < 1e-12) return null;
  return [gx / len, gy / len];
}
