// src/topics/svm-soft-margin/module.ts
// Task 12 (Wave 3): SVM with a SOFT margin — 2D binary classification.
//
// Data: the same two-cluster setup the svm-hard-margin sibling uses (class 0 on
// the left, class 1 on the right, seeded jitter) PLUS an optional deterministic
// OUTLIER that pushes one class-0 point deep into class-1 territory, making the
// dataset non-separable — the entire C-tradeoff story. (svm-hard-margin was
// being written in parallel when this module landed; the generator below
// follows the plan's shared description so the two datasets are compatible.)
//
// SOLVER (what is implemented, exactly — reviewers please read):
//   The soft-margin problem min ½‖w‖² + C·Σξᵢ s.t. yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ, ξᵢ ≥ 0
//   is NOT solved by a QP. In 2D a separating hyperplane is a line; we search
//   its ORIENTATION θ ∈ [0, 2π) (normal n = (cosθ, sinθ), w = s·n, s = ‖w‖):
//     • Coarse grid: 360 orientations, 1° steps.
//     • For a fixed orientation the remaining subproblem over (s, b),
//         h(s, b) = ½s² + C·Σᵢ max(0, 1 − yᵢ(s·(n·xᵢ) + b)),
//       is convex JOINTLY in (s, b) (hinge = max of two affine functions).
//       - For a fixed s the offset b is minimized EXACTLY: g(b) = Σᵢ max(0,
//         yᵢ(kᵢ − b)) with kᵢ = yᵢ − s·(n·xᵢ) is convex piecewise-linear, so its
//         minimizer sits at one of the n breakpoints b = kᵢ — we evaluate all n
//         (O(n²), n ≤ 30) and take the leftmost minimum (deterministic).
//       - The scale s is minimized by golden-section search over s ∈ [1e-4, 100]
//         (40 iterations → bracket shrinks by 0.618⁴⁰ ≈ 6e-9). Partial
//         minimization of a jointly convex function is convex, so h(s) is
//         unimodal and golden section is exact to float precision.
//     • The 3 best coarse orientations are refined with a golden-section search
//       over θ within ±1° (wrapped mod 2π), 30 iterations (≈3e-7 relative).
//     • The global best (θ, s, b) wins. Everything is deterministic: seeded
//       data, fixed grids, exact inner solves — no RNG, no QP library.
//   Termination: golden-section runs a FIXED iteration count (no data-dependent
//   convergence loop), so the solver always terminates in
//   360·40·O(n²) + 3·30·40·O(n²) ≈ 1.6e7 ops on n = 20.
//   Precision: on the default separable dataset the C = 1000 fit reproduces the
//   independent hard-margin reference (solveHardMarginReference — exhaustive
//   orientation search at 0.025°) within ~0.2% (asserted in testCases.test.ts).
//
// SWEEP: one snapshot per C on the log grid C_GRID up to the slider value; the
// LAST snapshot is always exactly the slider's C (mirrors ridge's λ-sweep and
// knn's k-sweep). Scrubbing the run IS the C-animation: the margin band
// shrinks, the slack lines shrink, the objective ½‖w‖² + C·Σξ climbs.
import type { TopicModule, Params, SimState, VisualCommand, MathStep, ParamValue } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { registerClassifier } from '../../registry/viewRegistry';
import { svmSoftTestCases } from './testCases';
import { svmSoftFormulas } from './formulas';
import { svmSoftDerivations } from './derivations';
import { svmSoftMistakes } from './mistakes';
import { svmSoftQuestions } from './questions';
import { svmSoftComparisons } from './comparisons';
import { svmSoftFailureDemos } from './failures';

export interface SvmPoint { x: number; y: number; cls: number; yLabel: number; }

// ===== constants =====
// Log-spaced C sweep (mirrors ridge's λ grid / knn's k grid). The run emits one
// snapshot per grid value below the slider's C, then the exact slider C last.
export const C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000];

const DOMAIN = 5;          // points are clamped to [−5, 5]²
const CLS_COLORS = ['#3b82f6', '#ef4444'];   // class 0 / class 1 (--cat1/--cat2 defaults)
const DECISION_COLOR = '#64748b';
const MARGIN_COLOR = '#f59e0b';
const SLACK_COLOR = '#f97316';
const SUPPORT_RING_COLOR = '#f59e0b';

// solver knobs (documented above)
const TWO_PI = Math.PI * 2;
const S_LO = 1e-4;
const S_HI = 100;
const GOLDEN_S_ITERS = 40;
const COARSE_ANGLES = 360;
const TOP_CANDIDATES = 3;
const GOLDEN_THETA_ITERS = 30;
const THETA_WINDOW = Math.PI / 180;          // ±1° refinement window

// evaluation tolerances (used by evaluateFit + metrics)
const XI_TOL = 1e-6;        // ξ above this ⇒ "violated" (inside the margin band)
const SUPPORT_TOL = 0.05;   // |y·f − 1| below this (and ξ ≈ 0) ⇒ support vector

/** Mulberry32 — deterministic PRNG so runs are reproducible. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(v: number): number { return Math.max(-DOMAIN, Math.min(DOMAIN, v)); }

/**
 * Deterministic two-cluster data (class 0 left / class 1 right, y = ±1 labels)
 * with per-axis jitter `spread`. With the `outlier` toggle ON, the LAST class-0
 * point is pushed to x = margin + outlierStrength·spread (its jittered y is
 * kept) — a deterministic non-separability that C must cope with.
 */
export function generatePoints(p: Params): SvmPoint[] {
  const nPerClass = (p.nPerClass as number) ?? 10;
  const margin = (p.margin as number) ?? 1.5;
  const spread = (p.spread as number) ?? 0.5;
  const outlier = p.outlier === true;
  const outlierStrength = (p.outlierStrength as number) ?? 3;
  const rng = mulberry32((p.seed as number) ?? 42);
  const pts: SvmPoint[] = [];
  for (let cls = 0; cls < 2; cls++) {
    const cx = cls === 0 ? -margin : margin;
    for (let i = 0; i < nPerClass; i++) {
      const x = clamp(cx + (rng() - 0.5) * 2 * spread);
      const y = clamp((rng() - 0.5) * 2 * spread);
      pts.push({ x, y, cls, yLabel: cls === 0 ? -1 : 1 });
    }
  }
  if (outlier) {
    const idx = nPerClass - 1;                       // last class-0 point
    pts[idx] = { ...pts[idx], x: clamp(margin + outlierStrength * spread) };
  }
  return pts;
}

// Point-set memoization (knn pattern): DecisionBoundary resolves the classifier
// 2500× per snapshot; the simulation reaches the same cached array per params.
const POINTS_CACHE = new Map<string, SvmPoint[]>();
const POINTS_CACHE_MAX = 64;

function pointsCacheKey(p: Params): string {
  return [
    'seed', (p.seed as number) ?? 42,
    'nPerClass', (p.nPerClass as number) ?? 10,
    'margin', (p.margin as number) ?? 1.5,
    'spread', (p.spread as number) ?? 0.5,
    'outlier', p.outlier === true ? 1 : 0,
    'outlierStrength', (p.outlierStrength as number) ?? 3,
  ].join('|');
}

export function getPoints(p: Params): SvmPoint[] {
  const key = pointsCacheKey(p);
  let pts = POINTS_CACHE.get(key);
  if (!pts) {
    pts = generatePoints(p);
    if (POINTS_CACHE.size >= POINTS_CACHE_MAX) POINTS_CACHE.clear();
    POINTS_CACHE.set(key, pts);
  }
  return pts;
}

// ===== pure math helpers =====

/** Hinge loss for a signed margin: max(0, 1 − y·f). */
export function hinge(yf: number): number {
  return Math.max(0, 1 - yf);
}

/** Per-point evaluation of a fitted hyperplane (w·x + b = 0). */
export interface PointEval {
  xi: number;          // slack = hinge(y·f)
  yf: number;          // signed margin y·(w·x + b)
  violated: boolean;   // ξ > XI_TOL  — strictly inside the margin band
  inside: boolean;     // 0 < ξ < 1  — inside the band but on the correct side
  misclassified: boolean; // ξ ≥ 1   — on the wrong side of the boundary
  support: boolean;    // on the margin: ξ ≈ 0 and |y·f − 1| < SUPPORT_TOL
}

export function evaluateFit(points: SvmPoint[], w1: number, w2: number, b: number): PointEval[] {
  return points.map((pt) => {
    const yf = pt.yLabel * (w1 * pt.x + w2 * pt.y + b);
    const xi = hinge(yf);
    return {
      xi, yf,
      violated: xi > XI_TOL,
      inside: xi > XI_TOL && xi < 1,
      misclassified: xi >= 1,
      support: xi <= XI_TOL && Math.abs(yf - 1) < SUPPORT_TOL,
    };
  });
}

// ===== solver =====

export interface SoftFit {
  theta: number;        // orientation of the unit normal (radians)
  s: number;            // ‖w‖
  w1: number; w2: number; b: number;
  slackSum: number;     // Σξᵢ (hinge loss sum)
  objective: number;    // ½‖w‖² + C·Σξᵢ
  margin: number;       // 2/‖w‖
}

/** Golden-section minimizer of a unimodal function over [lo, hi]. */
function goldenSection(f: (x: number) => number, lo: number, hi: number, iters: number): { x: number; v: number } {
  const R = (Math.sqrt(5) - 1) / 2;   // 0.618…
  let a = lo, b = hi;
  let c = b - R * (b - a);
  let d = a + R * (b - a);
  let fc = f(c), fd = f(d);
  for (let i = 0; i < iters; i++) {
    if (fc < fd) {
      b = d; d = c; fd = fc;
      c = b - R * (b - a); fc = f(c);
    } else {
      a = c; c = d; fc = fd;
      d = a + R * (b - a); fd = f(d);
    }
  }
  const x = (a + b) / 2;
  return { x, v: f(x) };
}

/**
 * Exact minimizer of g(b) = Σᵢ max(0, yᵢ(kᵢ − b)) — convex piecewise-linear in
 * b, so the minimum lies at a breakpoint b = kᵢ (or on a plateau whose value is
 * attained at its end breakpoints). Evaluating at all n breakpoints and taking
 * the leftmost minimum is exact and deterministic.
 */
function minimizeHingeG(k: number[], y: number[]): number {
  let best = Infinity;
  let bestB = k[0];
  for (let j = 0; j < k.length; j++) {
    const b = k[j];
    let g = 0;
    for (let i = 0; i < k.length; i++) {
      const d = y[i] * (k[i] - b);
      if (d > 0) g += d;
    }
    if (g < best) { best = g; bestB = b; }
  }
  return bestB;
}

interface LineSolve { s: number; b: number; slackSum: number; objective: number; }

/**
 * For a fixed orientation: minimize ½s² + C·Σξᵢ over (s > 0, b). The offset b*
 * is exact for each s (breakpoint minimizer above); s is minimized by golden
 * section over [S_LO, S_HI] (h is convex in s by partial minimization).
 */
function solveOrientation(z: number[], y: number[], C: number): LineSolve {
  const k = new Array<number>(z.length);
  const cost = (s: number): number => {
    for (let i = 0; i < z.length; i++) k[i] = y[i] - s * z[i];
    const b = minimizeHingeG(k, y);
    let xi = 0;
    for (let i = 0; i < z.length; i++) {
      const h = 1 - y[i] * (s * z[i] + b);
      if (h > 0) xi += h;
    }
    return 0.5 * s * s + C * xi;
  };
  const s = goldenSection(cost, S_LO, S_HI, GOLDEN_S_ITERS).x;
  for (let i = 0; i < z.length; i++) k[i] = y[i] - s * z[i];
  const b = minimizeHingeG(k, y);
  let slackSum = 0;
  for (let i = 0; i < z.length; i++) {
    const h = 1 - y[i] * (s * z[i] + b);
    if (h > 0) slackSum += h;
  }
  return { s, b, slackSum, objective: 0.5 * s * s + C * slackSum };
}

/**
 * Deterministic 2D soft-margin fit (see the module header for the full
 * algorithm). `C` must be > 0.
 */
export function solveSoftMargin(points: SvmPoint[], C: number): SoftFit {
  if (!(C > 0)) throw new Error('svm-soft-margin: C must be > 0');
  const y = points.map((pt) => pt.yLabel);
  const project = (n1: number, n2: number): number[] => points.map((pt) => n1 * pt.x + n2 * pt.y);

  const candidates: { theta: number; v: number }[] = [];
  for (let t = 0; t < COARSE_ANGLES; t++) {
    const theta = (TWO_PI * t) / COARSE_ANGLES;
    const sol = solveOrientation(project(Math.cos(theta), Math.sin(theta)), y, C);
    candidates.push({ theta, v: sol.objective });
  }
  candidates.sort((a, b) => a.v - b.v);

  const wrap = (a: number) => ((a % TWO_PI) + TWO_PI) % TWO_PI;
  let best: { theta: number; sol: LineSolve } | null = null;
  for (let c = 0; c < Math.min(TOP_CANDIDATES, candidates.length); c++) {
    const base = candidates[c].theta;
    const fTheta = (t: number): number => {
      const w = wrap(t);
      return solveOrientation(project(Math.cos(w), Math.sin(w)), y, C).objective;
    };
    const theta = goldenSection(fTheta, base - THETA_WINDOW, base + THETA_WINDOW, GOLDEN_THETA_ITERS).x;
    const tw = wrap(theta);
    const sol = solveOrientation(project(Math.cos(tw), Math.sin(tw)), y, C);
    if (!best || sol.objective < best.sol.objective) best = { theta: tw, sol };
  }
  const b0 = best as { theta: number; sol: LineSolve };
  const n1 = Math.cos(b0.theta), n2 = Math.sin(b0.theta);
  const s = b0.sol.s;
  return {
    theta: b0.theta, s,
    w1: s * n1, w2: s * n2, b: b0.sol.b,
    slackSum: b0.sol.slackSum,
    objective: 0.5 * s * s + C * b0.sol.slackSum,
    margin: 2 / s,
  };
}

/**
 * INDEPENDENT hard-margin reference (used by tests to verify the C → ∞ limit):
 * exhaustive orientation search at 0.025° over [0, 2π). For each orientation the
 * two classes' projections must be separable (max z₀ < min z₁); the max-margin
 * fit is then closed-form: s = 2/(min z₁ − max z₀), b = 1 − s·min z₁, support
 * points = the class-0 point at max z₀ and the class-1 point at min z₁.
 * Returns null when the data is not separable in any orientation.
 */
export interface HardMarginRef {
  theta: number; s: number; w1: number; w2: number; b: number;
  margin: number; supports: number[];
}

export function solveHardMarginReference(points: SvmPoint[], steps = 14400): HardMarginRef | null {
  let bestGap = -Infinity;
  let bestTheta = 0;
  for (let t = 0; t < steps; t++) {
    const theta = (TWO_PI * t) / steps;
    const n1 = Math.cos(theta), n2 = Math.sin(theta);
    let maxZ0 = -Infinity, minZ1 = Infinity;
    for (const pt of points) {
      const z = n1 * pt.x + n2 * pt.y;
      if (pt.yLabel === 1) { if (z < minZ1) minZ1 = z; }
      else if (z > maxZ0) maxZ0 = z;
    }
    const gap = minZ1 - maxZ0;
    if (gap > bestGap) { bestGap = gap; bestTheta = theta; }
  }
  if (!(bestGap > 0)) return null;
  const n1 = Math.cos(bestTheta), n2 = Math.sin(bestTheta);
  let maxZ0 = -Infinity, minZ1 = Infinity;
  let sup0 = -1, sup1 = -1;
  points.forEach((pt, i) => {
    const z = n1 * pt.x + n2 * pt.y;
    if (pt.yLabel === 1) { if (z < minZ1) { minZ1 = z; sup1 = i; } }
    else if (z > maxZ0) { maxZ0 = z; sup0 = i; }
  });
  const s = 2 / bestGap;
  return {
    theta: bestTheta, s,
    w1: s * n1, w2: s * n2, b: 1 - s * minZ1,
    margin: 2 / s,
    supports: [sup0, sup1],
  };
}

// ===== snapshot building =====

function metricsOf(points: SvmPoint[], fit: SoftFit, evals: PointEval[], C: number): Record<string, number> {
  const violated = evals.filter((e) => e.violated).length;
  const inside = evals.filter((e) => e.inside).length;
  const misclassified = evals.filter((e) => e.misclassified).length;
  const support = evals.filter((e) => e.support).length;
  return {
    C, w1: fit.w1, w2: fit.w2, b: fit.b, s: fit.s,
    margin: fit.margin,
    objective: fit.objective,
    hingeLoss: fit.slackSum,     // Σξᵢ — alias of slackSum: slack ξᵢ ≡ hinge loss max(0, 1−y·f)
    slackSum: fit.slackSum,      // identical by construction: slack ξᵢ ≡ hinge loss
    violatedCount: violated,
    insideMarginCount: inside,
    misclassifiedCount: misclassified,
    // FREE support vectors only (ξ ≈ 0, on the band). Bounded support vectors
    // (ξ > 0 ⇒ αᵢ = C, per the KKT derivation) ARE the violated points and are
    // reported separately as violatedCount — the narration says so explicitly.
    freeSupportCount: support,
    nPoints: points.length,
  };
}

function algorithmOf(fit: SoftFit, C: number): Record<string, ParamValue> {
  return {
    mode: 'svm-soft-margin', C,
    w1: fit.w1, w2: fit.w2, b: fit.b,
    s: fit.s, objective: fit.objective,
  };
}

// ---- visuals ----
type Box = { x0: number; y0: number; x1: number; y1: number };

/** Data-driven view box: point bbox expanded by ≥ 0.75 so helper lines stay in view. */
function visualBox(points: SvmPoint[]): Box {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const pt of points) {
    x0 = Math.min(x0, pt.x); x1 = Math.max(x1, pt.x);
    y0 = Math.min(y0, pt.y); y1 = Math.max(y1, pt.y);
  }
  const dx = Math.max(0.75, (x1 - x0) * 0.25);
  const dy = Math.max(0.75, (y1 - y0) * 0.25);
  return { x0: x0 - dx, x1: x1 + dx, y0: y0 - dy, y1: y1 + dy };
}

/** Liang–Barsky clip of the ray p0 + t·d (t ≥ 0) to the box; returns the visible segment. */
function clipSegment(p0: [number, number], d: [number, number], box: Box): [number, number][] | null {
  let t0 = 0, t1 = Infinity;
  const axes: [number, number, number, number][] = [
    [p0[0], d[0], box.x0, box.x1],
    [p0[1], d[1], box.y0, box.y1],
  ];
  for (const [p, dir, lo, hi] of axes) {
    if (Math.abs(dir) < 1e-12) {
      if (p < lo || p > hi) return null;
      continue;
    }
    let a = (lo - p) / dir, b = (hi - p) / dir;
    if (a > b) { const tmp = a; a = b; b = tmp; }
    t0 = Math.max(t0, a);
    t1 = Math.min(t1, b);
    if (t0 > t1) return null;
  }
  return [[p0[0] + t0 * d[0], p0[1] + t0 * d[1]], [p0[0] + t1 * d[0], p0[1] + t1 * d[1]]];
}

/** Segment of the level line w·x + b = level inside the box (clipped to view). */
function levelLineSegment(fit: SoftFit, level: number, box: Box): [number, number][] | null {
  const c = (level - fit.b) / (fit.s * fit.s);
  const p0: [number, number] = [c * fit.w1, c * fit.w2];
  const d: [number, number] = [-fit.w2 / fit.s, fit.w1 / fit.s];
  return clipSegment(p0, d, box);
}

/** Perpendicular slack line from a violated point to its foot on the decision line. */
function slackSegment(pt: SvmPoint, fit: SoftFit, box: Box): [number, number][] | null {
  const f = fit.w1 * pt.x + fit.w2 * pt.y + fit.b;
  const d: [number, number] = [-(f / (fit.s * fit.s)) * fit.w1, -(f / (fit.s * fit.s)) * fit.w2];
  return clipSegment([pt.x, pt.y], d, box);
}

/**
 * Visual convention (scatter-plot based — DecisionBoundary in registerViews.tsx
 * is not wired to receive supportVectors/marginLines props for this topic, so
 * the margin band and slack geometry live in the scatter visuals):
 *   • decision line f = 0                — slate
 *   • margin band lines f = ±1           — amber (offset ±1/‖w‖ from the boundary)
 *   • slack lines: point → its foot on   — orange, only for violated points
 *     the decision line (length = signed
 *     distance |f|/‖w‖ to the boundary)
 *   • support-vector rings               — small amber circles around ξ ≈ 0, |y·f−1|≈0
 *   • violated + support points          — highlighted (drawn larger) via SimState.highlights
 */
function buildVisuals(points: SvmPoint[], fit: SoftFit, evals: PointEval[]): VisualCommand[] {
  const box = visualBox(points);
  const cmds: VisualCommand[] = [];
  evals.forEach((e, i) => {
    if (e.violated) {
      const seg = slackSegment(points[i], fit, box);
      if (seg) cmds.push({ type: 'line', id: `slack${i}`, points: seg, color: SLACK_COLOR });
    }
  });
  const decision = levelLineSegment(fit, 0, box);
  if (decision) cmds.push({ type: 'line', id: 'decision', points: decision, color: DECISION_COLOR });
  const mPlus = levelLineSegment(fit, 1, box);
  if (mPlus) cmds.push({ type: 'line', id: 'margin-plus', points: mPlus, color: MARGIN_COLOR });
  const mMinus = levelLineSegment(fit, -1, box);
  if (mMinus) cmds.push({ type: 'line', id: 'margin-minus', points: mMinus, color: MARGIN_COLOR });
  evals.forEach((e, i) => {
    if (e.support) cmds.push({ type: 'circle', id: `sup-ring${i}`, x: points[i].x, y: points[i].y, r: 0.35, color: SUPPORT_RING_COLOR });
  });
  points.forEach((pt, i) => cmds.push({ type: 'point', id: `d${i}`, x: pt.x, y: pt.y, color: CLS_COLORS[pt.cls] }));
  return cmds;
}

function highlightOf(evals: PointEval[]): SimState['highlights'] {
  const hl: SimState['highlights'] = [];
  evals.forEach((e, i) => {
    if (e.violated || e.support) hl.push({ panel: 'canvas', id: `d${i}`, intensity: 1 });
  });
  return hl;
}

function fmtC(C: number): string {
  return String(C);
}

function snapshotAt(points: SvmPoint[], C: number, first: boolean, prev?: SimState): SimState {
  const fit = solveSoftMargin(points, C);
  const evals = evaluateFit(points, fit.w1, fit.w2, fit.b);
  const m = metricsOf(points, fit, evals, C);

  // Only the first snapshot's explanation needs the independent hard-margin
  // reference (null when the data is non-separable, e.g. with the outlier on).
  const hardRef = first ? solveHardMarginReference(points) : null;

  const math: MathStep[] = [
    { latex: '\\min_{w,b,\\xi}\\; \\tfrac{1}{2}\\|w\\|^2 + C\\sum_{i=1}^{n} \\xi_i \\quad\\text{s.t.}\\quad y_i(w\\cdot x_i + b) \\ge 1 - \\xi_i,\\; \\xi_i \\ge 0', id: 'svm-soft-objective' },
    { latex: '\\xi_i = \\max(0,\\; 1 - y_i\\,(w\\cdot x_i + b))', id: 'svm-hinge-loss' },
  ];
  if (m.violatedCount > 0) {
    math.push({ latex: `\\Sigma\\xi = ${m.slackSum.toFixed(3)} \\;(\\text{over } ${m.violatedCount} \\text{ of } ${m.nPoints} \\text{ points in the band})` });
  }

  const firstWhy =
    `C = ${C} makes violations nearly free, so the solver prefers a very wide band: margin = ${m.margin.toFixed(2)} ` +
    `${hardRef ? `vs the hard-margin ${hardRef.margin.toFixed(2)}` : '(no hard-margin fit exists — the data is non-separable)'} ` +
    `(‖w‖ = ${m.s.toFixed(3)}), and it even accepts a handful of misclassified points (${m.misclassifiedCount} of ${m.nPoints}) rather than paying for a tight fit. ${m.violatedCount} of ${m.nPoints} points pay slack ξᵢ (Σξ = ${m.slackSum.toFixed(3)}) — the soft-margin tradeoff at the cheap-slack end.`;
  let why = firstWhy;
  const changed: string[] = [];
  if (!first && prev) {
    const prevC = (prev.algorithm.C as number) ?? NaN;
    const prevMargin = prev.metrics.margin as number | undefined;
    const prevObj = prev.metrics.objective as number | undefined;
    changed.push(`C → ${fmtC(C)}`);
    if (prevMargin !== undefined && prevMargin !== m.margin) changed.push(`margin → ${m.margin.toFixed(2)}`);
    if (prevObj !== undefined && prevObj !== m.objective) changed.push(`objective → ${m.objective.toFixed(3)}`);
    // Only claim the margin band "tightens" when it actually moves: on separable
    // data the fit is already at the hard-margin optimum, so most C steps leave
    // the band (and the fit) untouched — only the price per unit of slack rises.
    const marginMoved = prevMargin !== undefined && prevMargin.toFixed(3) !== m.margin.toFixed(3);
    why =
      `Raising C from ${fmtC(prevC)} to ${fmtC(C)} makes each unit of slack ${C === 0 ? 0 : (C / (prevC || 1)).toFixed(0)}× more expensive. ` +
      (marginMoved
        ? `The solver tightens the margin band (${prevMargin !== undefined ? prevMargin.toFixed(2) : '—'} → ${m.margin.toFixed(2)}) to push points back outside it. `
        : `Here the margin band does not move (${prevMargin !== undefined ? prevMargin.toFixed(2) : '—'} → ${m.margin.toFixed(2)}) — the optimum is unchanged because no rotation would reduce total slack, so the fit stays put and only the price per unit of slack rises. `) +
      `Σξ = ${m.slackSum.toFixed(3)}, objective ½‖w‖² + C·Σξ = ${m.objective.toFixed(3)}.`;
  }

  const events: SimState['events'] = [{ type: 'fit', label: 'svm-soft-margin-solve', step: 0 }];
  if (m.violatedCount > 0) events.push({ type: 'violations', label: 'points-inside-margin', step: 0 });

  return {
    algorithm: algorithmOf(fit, C),
    visuals: buildVisuals(points, fit, evals),
    math,
    narration:
      `C = ${fmtC(C)}: margin = ${m.margin.toFixed(2)} (2/‖w‖), ‖w‖ = ${m.s.toFixed(3)}, ` +
      `objective = ${m.objective.toFixed(3)}, Σξ (hinge) = ${m.slackSum.toFixed(3)}, ` +
      `${m.violatedCount} points in the margin band (${m.misclassifiedCount} misclassified, ${m.insideMarginCount} inside-but-correct) — these are the bounded support vectors (αᵢ = C) — and ${m.freeSupportCount} free support vector${m.freeSupportCount === 1 ? '' : 's'} on the band (ξ = 0)`,
    explanation: {
      changed,
      why,
      formulaRef: 'svm-soft-objective',
      dependsOn: ['linear-algebra', 'convex-optimization', 'vector-geometry'],
      gateConcepts: ['SVM', 'soft-margin', 'hinge-loss', 'slack-variables', m.violatedCount > 0 ? 'misclassification' : 'separability'],
    },
    highlights: highlightOf(evals),
    metrics: m,
    events,
    timeline: first ? ['Data', 'Fit', 'Evaluate'] : ['Fit', 'Evaluate'],
  };
}

// ===== simulation: C-sweep =====

export const simulation = {
  /**
   * One snapshot per C on the log grid C_GRID below the slider's C, then the
   * slider C exactly last. The first snapshot is always C = 0.01 (the
   * slack-is-cheap reference); each step advances C to the next grid value.
   * Scrubbing the run IS the C-animation: margin band shrinks, slack lines
   * shrink, objective ½‖w‖² + C·Σξ climbs.
   */
  initialState: (p: Params): SimState => {
    const points = getPoints(p);
    return snapshotAt(points, C_GRID[0], true);
  },

  step: (p: Params, s: SimState): SimState | null => {
    const points = getPoints(p);
    const target = (p.C as number) ?? 1;
    const current = (s.algorithm.C as number) ?? 0;
    const next = C_GRID.find((g) => g > current + 1e-9);
    const nextC = next === undefined || next > target + 1e-9 ? target : next;
    if (nextC <= current + 1e-9) return null; // sweep complete
    return snapshotAt(points, nextC, false, s);
  },
};

// ===== topic module =====

export const svmSoftMarginModule: TopicModule = {
  id: 'svm-soft-margin',
  title: 'SVM: Soft Margin',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 4, coding: 2, visualization: 3, gateFrequency: 5 },
    estimatedHours: 5,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['linear-algebra', 'vector-geometry', 'svm-hard-margin', 'convex-optimization'],
    relatedTopics: ['svm-hard-margin', 'logistic-regression', 'ridge-regression', 'kernel-svm'],
    revision: { quick: '20m', standard: '1h', deep: '2h', mastery: '4h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Geometry: Data, Decision Line, Margin Band & Slack Lines' },
      { slot: 'primary', component: 'decision-boundary', title: 'Decision Regions (soft-margin classifier)' },
      { slot: 'sidebar', component: 'loss-curve', title: 'Objective ½‖w‖² + C·Σξ vs C (hinge Σξ second series)' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivation: Hinge Surrogate & Primal→Dual with Box Constraint' },
      { slot: 'primary', component: 'explain-step', title: 'Step Explanation' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'C', label: 'Soft-margin cost C', type: 'number', min: 0.01, max: 1000, step: 0.01, default: 1 },
    { id: 'nPerClass', label: 'Points per class', type: 'number', min: 2, max: 15, step: 1, default: 10 },
    { id: 'margin', label: 'Cluster separation', type: 'number', min: 0.5, max: 2.5, step: 0.1, default: 1.5 },
    { id: 'spread', label: 'Jitter spread', type: 'number', min: 0.1, max: 1.5, step: 0.1, default: 0.5 },
    { id: 'outlier', label: 'Outlier (class-0 point pushed into class 1)', type: 'toggle', default: false },
    { id: 'outlierStrength', label: 'Outlier strength (× spread)', type: 'number', min: 0.5, max: 6, step: 0.1, default: 3 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
  ],
  simulation,
  formulas: svmSoftFormulas,
  derivations: svmSoftDerivations,
  questions: svmSoftQuestions,
  comparisons: svmSoftComparisons,
  failureDemos: svmSoftFailureDemos,
  mistakes: svmSoftMistakes,
  testCases: svmSoftTestCases,
  lossMetricKey: 'objective',
  lossMetricKey2: 'hingeLoss',

  validateParams: (p) => {
    const issues: string[] = [];
    const C = p.C as number | undefined;
    if (C === undefined || !Number.isFinite(C)) issues.push('C must be a finite number');
    else if (C <= 0) issues.push('C must be > 0 (C = 0 allows every violation for free — the model degenerates to a constant)');
    else if (C < 0.01 || C > 1000) issues.push(`C = ${C} is outside the supported log-slider range [0.01, 1000]`);
    const nPerClass = p.nPerClass as number | undefined;
    if (nPerClass !== undefined && (!Number.isFinite(nPerClass) || nPerClass < 2)) {
      issues.push('nPerClass must be ≥ 2 (fewer makes the margin problem degenerate)');
    }
    const margin = p.margin as number | undefined;
    if (margin !== undefined && (!Number.isFinite(margin) || margin <= 0)) issues.push('Cluster separation (margin) must be > 0');
    const spread = p.spread as number | undefined;
    if (spread !== undefined && (!Number.isFinite(spread) || spread <= 0)) issues.push('Jitter spread must be > 0');
    const outlierStrength = p.outlierStrength as number | undefined;
    if (outlierStrength !== undefined && (!Number.isFinite(outlierStrength) || outlierStrength < 0)) {
      issues.push('outlierStrength must be ≥ 0');
    }
    return issues;
  },
};

export function register() {
  registerTopic(svmSoftMarginModule);
  // DecisionBoundary resolves this classifier via getClassifier('svm-soft-margin')
  // and merges {...params, ...snapshot.algorithm} per snapshot, so the CURRENT
  // step's hyperplane (w1, w2, b) drives the painted regions.
  registerClassifier('svm-soft-margin', (x, y, params) => {
    const w1 = params.w1 as number;
    const w2 = params.w2 as number;
    const b = params.b as number;
    return w1 * x + w2 * y + b > 0 ? 1 : 0;
  });
}
