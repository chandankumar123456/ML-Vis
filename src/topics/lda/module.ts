// src/topics/lda/module.ts
// Task 14 (Wave 3): lda — Fisher's Linear Discriminant Analysis as BOTH a
// 2-class classifier and a 1-D dimensionality reduction of 2-D data.
//
// Design decisions (deviations from the plan are documented in the report):
//  - Scatter convention (documented, self-consistent): per-class covariance
//    matrices C_c (each normalized by its own n_c — the sample covariance),
//    S_W := C_0 + C_1 (within-class scatter, normalized form), and
//    S_B := (μ₁−μ₂)(μ₁−μ₂)ᵀ (the Bishop 2-class between-class scatter).
//    With these definitions the Fisher criterion collapses to the plan's
//    test-2 formula EXACTLY (one metric, no ambiguity):
//        J(ŵ) = wᵀS_Bw / wᵀS_Ww = (μ̄₁−μ̄₂)² / (s₀² + s₁²)
//    where μ̄_c = ŵᵀμ_c is the projected class mean and s_c² = ŵᵀC_cŵ the
//    projected within-class variance. For balanced classes this is the
//    textbook raw-scatter (un-normalized) J up to the constant factor n.
//    The 2-class closed form w = S_W⁻¹(μ₁−μ₂) maximizes J (rank-1 S_B ⇒ the
//    generalized eigenproblem S_Bw = λS_Ww has a single nonzero eigenvalue
//    λ = dᵀS_W⁻¹d = J(w*), verified numerically in the tests).
//  - Step model (mandated): a 36-step ANGLE SWEEP of candidate projection
//    axes — one snapshot per direction θ_k = k·5° over [0, π), k = 0..35 —
//    followed by one final snapshot carrying the closed-form LDA solution
//    (w = S_W⁻¹(μ₁−μ₂), exact optimum, not a grid angle). Each sweep snapshot
//    projects EVERY point onto the candidate axis (guide lines + projected
//    points + direction arrow + threshold marker) and reports J(θ), the
//    within-class variances along θ, the between-class gap, and the
//    threshold-rule training error. The final snapshot compares J(θ*) with
//    the sweep's grid maximum (narration + metrics). lossMetricKey = 'jFisher'
//    plotted DIRECTLY (a quality metric: higher = better); the layer is titled
//    honestly and the final snapshot's J is the best-value marker at the end.
//  - Direction orientation: every candidate axis is oriented so ŵᵀ(μ₁−μ₂) ≥ 0
//    (class 1 projects HIGH). This makes the threshold rule
//    class = wx·x + wy·y + b > 0 ? 1 : 0 branchless for every snapshot; the
//    axis LABEL (angleDeg, in [0, π)) is what the sweep sweeps and compares.
//    Flipping the arrow leaves J, variances and error unchanged.
//  - Singular S_W (failure demo: collinear / too-few samples) THROWS from
//    initialState — the honest telemetry failure path (svm-hard-margin's
//    non-separable precedent): the run ends with failedAtStep and a reason
//    matching /singular/ instead of emitting NaN into the sandbox.
//  - Determinism: data is drawn from a seeded mulberry32 stream; the whole
//    sweep is a pure function of params (no memoization needed — 37 snapshots
//    × cheap O(n·d) stats). Same params ⇒ byte-identical snapshot arrays.
//  - Classifier contract (viewRegistry): registerClassifier('lda', fn) where
//    fn = (x, y, params) => class index. DecisionBoundary merges
//    {...params, ...snapshot.algorithm} before each grid call, so the current
//    axis/threshold (wx, wy, b) drive the boundary — scrubbing the sweep
//    rotates the decision line with the candidate axis. Before any run exists
//    the fallback re-solves the closed form from params (memoized, bounded).
import type { TopicModule, Params, SimState, VisualCommand, ParamValue, MathStep } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { registerClassifier } from '../../registry/viewRegistry';
import { ldaTestCases } from './testCases';
import { ldaFormulas } from './formulas';
import { ldaDerivations } from './derivations';
import { ldaMistakes } from './mistakes';
import { ldaQuestions } from './questions';
import { ldaComparisons } from './comparisons';
import { ldaFailureDemos } from './failures';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Sweep grid: 36 directions over [0, π) = 5° steps. Must divide π exactly so
// the sweep is uniform and the LAST grid angle is 175° (θ* = π − 5°).
export const ANGLE_COUNT = 36;
export const ANGLE_STEP_DEG = 5;

// Class palette matches the --cat1/--cat2 CSS-variable conventions used by
// the decision-boundary view and the rest of the Wave-3 cluster.
export const CLASS_COLORS = ['#2563eb', '#dc2626'];

// Visual-semantic colors (scatter canvas).
export const AXIS_COLOR = '#64748b';       // candidate projection axis
export const DIRECTION_COLOR = '#0f172a';  // oriented direction arrow
export const MEAN_COLOR = '#0f172a';       // projected class-mean markers
export const THRESHOLD_COLOR = '#f59e0b';  // decision threshold marker
export const GUIDE_ALPHA = '55';           // hex alpha for per-point guide lines

// Shared-covariance base scale: the minor axis std is fixed; the major expands
// with covShape (default 2 → σ = (1.0, 0.5) before rotation).
export const BASE_SIGMA = 0.5;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

export interface LdaPoint { x: number; y: number; cls: number; } // cls ∈ {0, 1}

/**
 * Hand-derivable toy set (test-only, `toy: true` — the plan's case-1 target).
 * The scatter math is printed in full in testCases.ts / testCases.test.ts:
 *
 *   class 0: (0,0) (2,0) (0,2) (2,2)   → μ₀ = (1, 1),  C₀ = [[1, 0],[0, 1]] = I
 *   class 1: (4,0) (4,2) (6,2) (6,4)   → μ₁ = (5, 2),  C₁ = [[1, 1],[1, 2]]
 *
 *   d  = μ₁ − μ₀ = (4, 1)
 *   S_W = C₀ + C₁ = [[2, 1],[1, 3]]       det = 5
 *   S_W⁻¹ = (1/5)·[[3, −1],[−1, 2]]
 *   w  = S_W⁻¹·d = (11/5, −2/5)           (unit: (11, −2)/(5√5))
 *   J(w*) = dᵀS_W⁻¹d = 42/5 = 8.4         (= eigen-Link λ = 8.4)
 *   s₀² = 1, s₁² = ŵᵀC₁ŵ = 17/25 = 0.68
 *   τ  = ŵᵀ(μ₀+μ₁)/2 = 6/√5 ≈ 2.6833
 *   → every point is classified correctly by the threshold rule on ŵ (hand
 *     checked: class-0 projections 0, 16/√...  all < τ; class-1 all > τ).
 */
export const TOY_POINTS: LdaPoint[] = [
  { x: 0, y: 0, cls: 0 }, { x: 2, y: 0, cls: 0 }, { x: 0, y: 2, cls: 0 }, { x: 2, y: 2, cls: 0 },
  { x: 4, y: 0, cls: 1 }, { x: 4, y: 2, cls: 1 }, { x: 6, y: 2, cls: 1 }, { x: 6, y: 4, cls: 1 },
];

export interface GeneratedData { points: LdaPoint[]; dataSeed: number; }

/** Mulberry32 — deterministic PRNG (matches every Wave-1/2/3 topic). */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller standard normal (with the spare-draw trick). */
function makeNormal(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0;
    let v = 0;
    do { u = rng(); } while (u <= 1e-12);
    v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

/**
 * Deterministic data synthesis. Two Gaussian clusters sharing ONE covariance
 * (the LDA assumption — the point of the topic): class 0 at (−sep/2, 0),
 * class 1 at (+sep/2, 0), both ~ N(μ_c, Σ) with
 *   Σ = R(φ)·diag(σA², σB²)·R(φ)ᵀ,  σA = BASE_SIGMA·covShape, σB = BASE_SIGMA.
 * Sample p = μ_c + R(φ)·diag(σA, σB)·[g₁,g₂]ᵀ with g₁,g₂ iid N(0,1) — the
 * Cholesky form of the Gaussian draw, so the EMPIRICAL per-class covariance
 * tilts with covAngleDeg and elongates with covShape deterministically.
 * Test-only overrides: `toy: true` → TOY_POINTS (hand-derivable); `points`
 * JSON string '[[x,y,cls],…]' → a hand-crafted dataset (failure demos).
 */
export function generateData(p: Params): GeneratedData {
  if (p.toy === true) return { points: TOY_POINTS.map((d) => ({ ...d })), dataSeed: (p.seed as number) ?? 42 };
  if (typeof p.points === 'string') {
    const rows: [number, number, number][] = JSON.parse(p.points);
    return { points: rows.map(([x, y, cls]) => ({ x, y, cls })), dataSeed: (p.seed as number) ?? 42 };
  }
  const nPerClass = (p.nPerClass as number) ?? 15;
  const separation = (p.separation as number) ?? 2;
  const angleDeg = (p.covAngleDeg as number) ?? 30;
  const shape = (p.covShape as number) ?? 2;
  const seed = (p.seed as number) ?? 42;
  const rng = mulberry32(seed);
  const normal = makeNormal(rng);
  const phi = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(phi), sin = Math.sin(phi);
  const sA = BASE_SIGMA * shape, sB = BASE_SIGMA;
  const pts: LdaPoint[] = [];
  for (let c = 0; c < 2; c++) {
    const cx = c === 0 ? -separation / 2 : separation / 2;
    for (let i = 0; i < nPerClass; i++) {
      const g1 = normal(), g2 = normal();
      // Cholesky factor rows: [cos·σA, −sin·σB ; sin·σA, cos·σB]
      const x = cx + cos * sA * g1 - sin * sB * g2;
      const y = 0 + sin * sA * g1 + cos * sB * g2;
      pts.push({ x, y, cls: c });
    }
  }
  return { points: pts, dataSeed: seed };
}

// ---------------------------------------------------------------------------
// LDA math core (all mutations-free, exported for hand-verified tests)
// ---------------------------------------------------------------------------

export interface LdaStats {
  n: number; n0: number; n1: number;
  mu0: [number, number]; mu1: [number, number];       // class means
  d: [number, number];                                 // μ₁ − μ₀
  C0: number[][]; C1: number[][];                      // per-class covariances
  SW: number[][]; SB: number[][];                      // C₀+C₁, d·dᵀ
  detSW: number;
  invertible: boolean;
  wRaw: [number, number];                              // S_W⁻¹·d (un-normalized)
  wUnit: [number, number];                            // oriented unit direction
  thetaAxis: number;                                   // axis label ∈ [0, π) rad
  thetaAxisDeg: number;
  jOpt: number;                                        // J(w*) = dᵀS_W⁻¹d
  threshold: number;                                   // τ = ŵᵀ(μ₀+μ₁)/2
  wx: number; wy: number; b: number;                  // affine rule, b = −τ
  withinVar0: number; withinVar1: number;             // s_c² along ŵ*
  trainError: number;                                  // error at the optimum
}

/** Per-class mean of the points carrying the given class. */
export function meanOf(points: LdaPoint[], cls: number): [number, number] {
  const ps = points.filter((d) => d.cls === cls);
  const n = Math.max(ps.length, 1);
  let sx = 0, sy = 0;
  for (const d of ps) { sx += d.x; sy += d.y; }
  return [sx / n, sy / n];
}

/** Per-class sample covariance (normalized by n_c — the S_W convention above). */
export function covOf(points: LdaPoint[], cls: number, mu: [number, number]): number[][] {
  const ps = points.filter((d) => d.cls === cls);
  const n = Math.max(ps.length, 1);
  let s11 = 0, s12 = 0, s22 = 0;
  for (const d of ps) {
    const dx = d.x - mu[0], dy = d.y - mu[1];
    s11 += dx * dx; s12 += dx * dy; s22 += dy * dy;
  }
  return [[s11 / n, s12 / n], [s12 / n, s22 / n]];
}

/** Quadratic form wᵀ·M·w for a 2×2 symmetric M and 2-vector w. */
export function quadForm(M: number[][], w: [number, number]): number {
  return w[0] * (M[0][0] * w[0] + M[0][1] * w[1]) + w[1] * (M[1][0] * w[0] + M[1][1] * w[1]);
}

/** Wrap an angle into [0, π) — the LDA axis label (unoriented direction). */
export function toAxisAngle(theta: number): number {
  let t = theta % Math.PI;
  if (t < 0) t += Math.PI;
  return t;
}

/** Threshold-rule classification error of (ŵ, τ) on the dataset. */
function trainErrorOf(points: LdaPoint[], wx: number, wy: number, tau: number): number {
  let bad = 0;
  for (const d of points) {
    const pred = wx * d.x + wy * d.y > tau ? 1 : 0;
    if (pred !== d.cls) bad++;
  }
  return points.length ? bad / points.length : 0;
}

/**
 * The closed-form 2-class LDA solution. Throws an honest error when S_W is
 * singular (det below 1e-9 — collinear classes or too few samples): LDA has
 * no solution there, and the failure demo leans on this telemetry path.
 */
export function computeLdaStats(data: { points: LdaPoint[] }): LdaStats {
  const points = data.points;
  const mu0 = meanOf(points, 0);
  const mu1 = meanOf(points, 1);
  const C0 = covOf(points, 0, mu0);
  const C1 = covOf(points, 1, mu1);
  const SW = [
    [C0[0][0] + C1[0][0], C0[0][1] + C1[0][1]],
    [C0[1][0] + C1[1][0], C0[1][1] + C1[1][1]],
  ];
  const d: [number, number] = [mu1[0] - mu0[0], mu1[1] - mu0[1]];
  const detSW = SW[0][0] * SW[1][1] - SW[0][1] * SW[1][0];
  const invertible = Math.abs(detSW) > 1e-9;
  if (!invertible) {
    throw new Error(
      'lda: within-class scatter S_W is singular (|det S_W| < 1e-9) — the classes are collinear or have too few samples; ' +
      'Fisher\'s LDA needs a full-rank within-class scatter to compute S_W⁻¹(μ₁−μ₂)',
    );
  }
  // 2×2 adjugate inverse: (1/det)·[[c, −b],[−b, a]] for [[a,b],[b,c]]
  const invSW = [
    [SW[1][1] / detSW, -SW[0][1] / detSW],
    [-SW[1][0] / detSW, SW[0][0] / detSW],
  ];
  const wRaw: [number, number] = [
    invSW[0][0] * d[0] + invSW[0][1] * d[1],
    invSW[1][0] * d[0] + invSW[1][1] * d[1],
  ];
  const norm = Math.hypot(wRaw[0], wRaw[1]);
  let wUnit: [number, number] = [wRaw[0] / norm, wRaw[1] / norm];
  // Orient the axis so class 1 projects HIGH (ŵᵀd ≥ 0): branchless threshold rule.
  if (wUnit[0] * d[0] + wUnit[1] * d[1] < 0) wUnit = [-wUnit[0], -wUnit[1]];
  const thetaAxis = toAxisAngle(Math.atan2(wRaw[1], wRaw[0]));
  const muBar0 = wUnit[0] * mu0[0] + wUnit[1] * mu0[1];
  const muBar1 = wUnit[0] * mu1[0] + wUnit[1] * mu1[1];
  const threshold = (muBar0 + muBar1) / 2;
  const withinVar0 = quadForm(C0, wUnit);
  const withinVar1 = quadForm(C1, wUnit);
  // J(w*) = (ŵᵀd)²/(s₀²+s₁²) — also equal to dᵀS_W⁻¹d (asserted in tests).
  const jOpt = (wUnit[0] * d[0] + wUnit[1] * d[1]) ** 2 / Math.max(withinVar0 + withinVar1, 1e-12);
  const n0 = points.filter((q) => q.cls === 0).length;
  const n1 = points.filter((q) => q.cls === 1).length;
  const SB = [[d[0] * d[0], d[0] * d[1]], [d[0] * d[1], d[1] * d[1]]];
  return {
    n: points.length, n0, n1,
    mu0, mu1, d, C0, C1, SW, SB, detSW, invertible, wRaw, wUnit,
    thetaAxis, thetaAxisDeg: (thetaAxis * 180) / Math.PI,
    jOpt, threshold, wx: wUnit[0], wy: wUnit[1], b: -threshold,
    withinVar0, withinVar1,
    trainError: trainErrorOf(points, wUnit[0], wUnit[1], threshold),
  };
}

// ---------------------------------------------------------------------------
// Per-angle evaluation (the sweep)
// ---------------------------------------------------------------------------

export interface AngleEval {
  theta: number;                 // rad
  angleDeg: number;              // axis label ∈ [0, π) in degrees
  wx: number; wy: number;        // oriented unit direction
  muBar0: number; muBar1: number;// projected class means
  betweenGap: number;            // |μ̄₁ − μ̄₀|
  withinVar0: number; withinVar1: number;
  jFisher: number;               // (μ̄₁−μ̄₀)²/(s₀²+s₁²) — the loss-curve metric
  threshold: number;             // τ for this direction
  b: number;                     // −τ
  trainError: number;            // threshold-rule error along this axis
}

/** Evaluate the Fisher criterion and threshold rule for one candidate axis. */
export function angleEvalAt(stats: LdaStats, data: GeneratedData, theta: number): AngleEval {
  const d0 = stats.d[0], d1 = stats.d[1];
  let wx = Math.cos(theta), wy = Math.sin(theta);
  if (wx * d0 + wy * d1 < 0) { wx = -wx; wy = -wy; } // orient class-1 side HIGH
  const muBar0 = wx * stats.mu0[0] + wy * stats.mu0[1];
  const muBar1 = wx * stats.mu1[0] + wy * stats.mu1[1];
  const withinVar0 = quadForm(stats.C0, [wx, wy]);
  const withinVar1 = quadForm(stats.C1, [wx, wy]);
  const between = muBar1 - muBar0;
  const threshold = (muBar0 + muBar1) / 2;
  const jFisher = between * between / Math.max(withinVar0 + withinVar1, 1e-12);
  return {
    theta, angleDeg: toAxisAngle(theta) * 180 / Math.PI,
    wx, wy, muBar0, muBar1, betweenGap: Math.abs(between),
    withinVar0, withinVar1, jFisher, threshold, b: -threshold,
    trainError: trainErrorOf(data.points, wx, wy, threshold),
  };
}

export interface LdaSweep {
  data: GeneratedData;
  stats: LdaStats;                 // closed form (throws when S_W singular)
  angles: number[];                // θ_k = k·π/36, k = 0..35
  evals: AngleEval[];              // per grid angle
  center: [number, number];        // overall mean (axis anchor)
}

/** Full sweep + closed form for a params tuple — the run's single source. */
export function getSweep(p: Params): LdaSweep {
  const data = generateData(p);
  const stats = computeLdaStats(data);
  const angles = Array.from({ length: ANGLE_COUNT }, (_, k) => (k * Math.PI) / ANGLE_COUNT);
  const evals = angles.map((theta) => angleEvalAt(stats, data, theta));
  const center: [number, number] = [
    (stats.mu0[0] + stats.mu1[0]) / 2,
    (stats.mu0[1] + stats.mu1[1]) / 2,
  ];
  return { data, stats, angles, evals, center };
}

// ---------------------------------------------------------------------------
// Classifier for the decision-boundary view
// ---------------------------------------------------------------------------

let solveCache: { key: string; sweep: LdaSweep } | null = null;

function sweepKey(p: Params): string {
  return JSON.stringify([
    p.toy ?? false, p.points ?? null,
    p.nPerClass ?? 15, p.separation ?? 2, p.covAngleDeg ?? 30, p.covShape ?? 2, p.seed ?? 42,
  ]);
}

function cachedSweep(p: Params): LdaSweep {
  const key = sweepKey(p);
  if (solveCache && solveCache.key === key) return solveCache.sweep;
  // Bounded by construction: any single run / boundary redraw sees ≤ a few keys.
  solveCache = { key, sweep: getSweep(p) };
  return solveCache.sweep;
}

/**
 * The registered classifier. The view merges the CURRENT snapshot's algorithm
 * state (wx, wy, b) into params, so the axis+threshold of the exact step being
 * scrubbed drive the boundary — no re-solving per grid cell. Fallback (before
 * any run exists) re-solves the closed form from params via the memo cache.
 */
export function classifyByParams(x: number, y: number, p: Params): number {
  const wx = p.wx as number | undefined;
  const wy = p.wy as number | undefined;
  const b = p.b as number | undefined;
  if (typeof wx === 'number' && typeof wy === 'number' && typeof b === 'number' && Number.isFinite(wx + wy + b)) {
    return wx * x + wy * y + b > 0 ? 1 : 0;
  }
  const s = cachedSweep(p);
  return s.stats.wx * x + s.stats.wy * y + s.stats.b > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

function metricsOf(sweep: LdaSweep, ev: AngleEval, step: number, isFinal: boolean): Record<string, number> {
  const m: Record<string, number> = {
    step, angleDeg: ev.angleDeg,
    jFisher: ev.jFisher,
    betweenGap: ev.betweenGap,
    withinVar0: ev.withinVar0, withinVar1: ev.withinVar1,
    threshold: ev.threshold,
    trainError: ev.trainError,
    isOptimal: isFinal ? 1 : 0,
    n: sweep.data.points.length,
    n0: sweep.stats.n0, n1: sweep.stats.n1,
    dataSeed: sweep.data.dataSeed,
    // classifier model (also mirrored in algorithm — convenient for tests)
    wx: ev.wx, wy: ev.wy, b: ev.b,
  };
  if (isFinal) {
    m.rawW1 = sweep.stats.wRaw[0];
    m.rawW2 = sweep.stats.wRaw[1];
    m.jOpt = sweep.stats.jOpt;
    m.thetaOptDeg = sweep.stats.thetaAxisDeg;
    m.eigenLambda = sweep.stats.d[0] * sweep.stats.wRaw[0] + sweep.stats.d[1] * sweep.stats.wRaw[1]; // dᵀS_W⁻¹d
    // grid max for the "optimal beats the sweep" narration
    let gridMax = -Infinity;
    for (const e of sweep.evals) gridMax = Math.max(gridMax, e.jFisher);
    m.gridMaxJ = gridMax;
  }
  return m;
}

// ---------------------------------------------------------------------------
// Visuals
// ---------------------------------------------------------------------------

/** Clip the ray center + t·dir (unit dir) to the data bbox (Liang–Barsky). */
function clipRay(center: [number, number], dir: [number, number],
  x0: number, x1: number, y0: number, y1: number): [number, number][] | null {
  let tMin = -Infinity, tMax = Infinity;
  for (const [lo, hi, pc, dc] of [[x0, x1, center[0], dir[0]], [y0, y1, center[1], dir[1]]] as const) {
    if (Math.abs(dc) < 1e-12) continue;
    const t1 = (lo - pc) / dc, t2 = (hi - pc) / dc;
    tMin = Math.max(tMin, Math.min(t1, t2));
    tMax = Math.min(tMax, Math.max(t1, t2));
  }
  if (tMin > tMax || !Number.isFinite(tMin) || !Number.isFinite(tMax)) return null;
  return [[center[0] + tMin * dir[0], center[1] + tMin * dir[1]],
    [center[0] + tMax * dir[0], center[1] + tMax * dir[1]]] as [number, number][];
}

function dataBounds(points: LdaPoint[]): { x0: number; x1: number; y0: number; y1: number } {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const d of points) {
    x0 = Math.min(x0, d.x); x1 = Math.max(x1, d.x);
    y0 = Math.min(y0, d.y); y1 = Math.max(y1, d.y);
  }
  return { x0, x1, y0, y1 };
}

/**
 * Projection convention (registry-rendered via scatter-plot commands):
 *  - 'axis'      : the candidate projection line through the overall data
 *                  center along ŵ (dashed-guide visual semantics, solid line);
 *  - 'dir'       : arrow along the ORIENTED direction ŵ (class-1 side high);
 *  - 'pg<i>'     : per-point guide line from the point to its projection
 *                  (class color, faint — the projection residual segment);
 *  - 'pp<i>'     : the projected point ON the axis (class color);
 *  - 'pm0'/'pm1' : class-mean projections (circle markers);
 *  - 'thr'       : the decision threshold τ on the axis (amber circle).
 * Everything is world-space; bounds fitting follows ScatterPlot's convention.
 */
function buildScatter(
  data: GeneratedData, sweep: LdaSweep, ev: AngleEval, isFinal: boolean,
): VisualCommand[] {
  const { points } = data;
  const cmd: VisualCommand[] = points.map((d, i) => ({
    type: 'point', id: `d${i}`, x: d.x, y: d.y, color: CLASS_COLORS[d.cls],
  }));
  const { x0, x1, y0, y1 } = dataBounds(points);
  const axis = clipRay(sweep.center, [ev.wx, ev.wy], x0, x1, y0, y1);
  if (axis) cmd.push({ type: 'line', id: 'axis', points: axis, color: AXIS_COLOR });
  // oriented direction arrow: from the center along ŵ, length ~0.9 world units
  cmd.push({
    type: 'arrow', id: 'dir', color: DIRECTION_COLOR,
    x1: sweep.center[0], y1: sweep.center[1],
    x2: sweep.center[0] + ev.wx * 0.9, y2: sweep.center[1] + ev.wy * 0.9,
  });
  // per-point projection: guide line + projected point on the axis
  points.forEach((d, i) => {
    const t = ev.wx * (d.x - sweep.center[0]) + ev.wy * (d.y - sweep.center[1]);
    const px = sweep.center[0] + ev.wx * t;
    const py = sweep.center[1] + ev.wy * t;
    cmd.push({ type: 'line', id: `pg${i}`, points: [[d.x, d.y], [px, py]] as [number, number][], color: CLASS_COLORS[d.cls] + GUIDE_ALPHA });
    cmd.push({ type: 'point', id: `pp${i}`, x: px, y: py, color: CLASS_COLORS[d.cls] });
  });
  // class-mean projections (markers on the axis)
  for (const c of [0, 1] as const) {
    const mu = c === 0 ? sweep.stats.mu0 : sweep.stats.mu1;
    const t = ev.wx * (mu[0] - sweep.center[0]) + ev.wy * (mu[1] - sweep.center[1]);
    cmd.push({
      type: 'circle', id: c === 0 ? 'pm0' : 'pm1',
      x: sweep.center[0] + ev.wx * t, y: sweep.center[1] + ev.wy * t, r: 0.16,
      color: MEAN_COLOR,
    });
  }
  // decision threshold marker on the axis
  const tc = sweep.center[0] + ev.wx * (ev.threshold - (ev.wx * sweep.center[0] + ev.wy * sweep.center[1]));
  const tcY = sweep.center[1] + ev.wy * (ev.threshold - (ev.wx * sweep.center[0] + ev.wy * sweep.center[1]));
  cmd.push({ type: 'circle', id: 'thr', x: tc, y: tcY, r: 0.16, color: THRESHOLD_COLOR });
  if (isFinal) {
    cmd.push({ type: 'line', id: 'optimal-axis', points: axis ?? [[0, 0], [0, 0]], color: THRESHOLD_COLOR });
  }
  return cmd;
}

/** Matrix story for matrix-animator: S_W, S_B, w, and (final) S_W⁻¹ + λ. */
function buildMatrices(sweep: LdaSweep, ev: AngleEval, isFinal: boolean): VisualCommand[] {
  const s = sweep.stats;
  const cmds: VisualCommand[] = [
    { type: 'matrix', id: 'S_W', rows: 2, cols: 2, cells: s.SW },
    { type: 'matrix', id: 'S_B', rows: 2, cols: 2, cells: s.SB },
    { type: 'matrix', id: 'ŵ', rows: 2, cols: 1, cells: [[ev.wx], [ev.wy]] },
    { type: 'matrix', id: 'J(θ)', rows: 1, cols: 1, cells: [[ev.jFisher]] },
  ];
  if (isFinal) {
    const det = s.detSW;
    const inv = [[s.SW[1][1] / det, -s.SW[0][1] / det], [-s.SW[1][0] / det, s.SW[0][0] / det]];
    cmds.push(
      { type: 'matrix', id: 'S_W⁻¹', rows: 2, cols: 2, cells: inv },
      { type: 'matrix', id: 'w = S_W⁻¹(μ₁−μ₂)', rows: 2, cols: 1, cells: [[s.wRaw[0]], [s.wRaw[1]]] },
      { type: 'matrix', id: 'λ = dᵀS_W⁻¹d', rows: 1, cols: 1, cells: [[s.jOpt]] },
    );
  }
  return cmds;
}

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

function snapshotAt(sweep: LdaSweep, ev: AngleEval, step: number, isFinal: boolean, first: boolean): SimState {
  const m = metricsOf(sweep, ev, step, isFinal);
  const timeline = first
    ? ['Data', 'Scatter', 'Project', 'Evaluate']
    : isFinal ? ['Project', 'Evaluate', 'LDA Solution'] : ['Scatter', 'Project', 'Evaluate'];
  const events: SimState['events'] = first
    ? [
        { type: 'init', label: 'lda-data-generated', step },
        // the first snapshot IS grid direction 0° — carry its candidate label
        // so the 36 grid directions produce events g1..g36 exactly
        { type: 'candidate', label: `direction-g${step}-of-${ANGLE_COUNT}`, step },
      ]
    : isFinal
      ? [{ type: 'converged', label: 'closed-form-lda-solution', step }]
      : [{ type: 'candidate', label: `direction-g${step}-of-${ANGLE_COUNT}`, step }];

  if (isFinal) {
    const s = sweep.stats;
    const narration =
      `Closed-form LDA: S_W = ${fmtMat(s.SW)} (det ${s.detSW.toFixed(4)}), S_W⁻¹(μ₁−μ₂) = (${s.wRaw[0].toFixed(4)}, ${s.wRaw[1].toFixed(4)}) ` +
      `→ unit axis θ* = ${s.thetaAxisDeg.toFixed(1)}°, J(θ*) = ${s.jOpt.toFixed(4)} ` +
      `(grid maximum was J = ${m.gridMaxJ!.toFixed(4)}). ` +
      `Projected within-class variances s₀² = ${s.withinVar0.toFixed(4)}, s₁² = ${s.withinVar1.toFixed(4)}; ` +
      `threshold τ = ${s.threshold.toFixed(4)} on the axis (b = ${s.b.toFixed(4)}); ` +
      `threshold-rule training error = ${(s.trainError * 100).toFixed(1)}%. ` +
      `Eigen-link: the single nonzero eigenvalue of S_W⁻¹S_B is λ = dᵀS_W⁻¹d = ${m.eigenLambda!.toFixed(4)} = J(θ*).`;
    const math: MathStep[] = [
      { latex: 'J(w) = \\frac{w^T S_B w}{w^T S_W w} = \\frac{(\\bar\\mu_1 - \\bar\\mu_0)^2}{s_0^2 + s_1^2}', id: 'lda-fisher-criterion' },
      { latex: 'w = S_W^{-1}(\\mu_1 - \\mu_2)', id: 'lda-solution' },
      { latex: 'S_B w = \\lambda S_W w \\;\\Rightarrow\\; S_W^{-1}S_B w = \\lambda w', id: 'lda-eigenproblem' },
      { latex: '\\tau = \\hat{w}^T \\frac{\\mu_0 + \\mu_1}{2}', id: 'lda-threshold' },
    ];
    return {
      algorithm: {
        mode: 'lda-optimal', step, isOptimal: 1,
        wx: s.wx, wy: s.wy, b: s.b,
        thetaAxisDeg: s.thetaAxisDeg, jFisher: s.jOpt,
        rawW1: s.wRaw[0], rawW2: s.wRaw[1],
        threshold: s.threshold, dataSeed: sweep.data.dataSeed,
      } as Record<string, ParamValue>,
      visuals: [...buildScatter(sweep.data, sweep, ev, true), ...buildMatrices(sweep, ev, true)],
      math,
      narration,
      explanation: {
        changed: [`axis → θ* = ${s.thetaAxisDeg.toFixed(1)}°`, `J → ${s.jOpt.toFixed(4)} (grid max ${m.gridMaxJ!.toFixed(4)})`],
        why: `The sweep ends: instead of a grid direction, the CLOSED-FORM solution w = S_W⁻¹(μ₁−μ₂) is evaluated — for rank-1 S_B the generalized eigenproblem collapses to this single vector, the exact Fisher maximum (λ = dᵀS_W⁻¹d = J).`,
        formulaRef: 'lda-solution',
        dependsOn: ['linear-algebra', 'probability', 'statistics'],
        gateConcepts: ['LDA', 'Fisher criterion', 'between-class scatter', 'within-class scatter'],
      },
      highlights: [
        { panel: 'canvas', id: 'axis', intensity: 1 },
        { panel: 'canvas', id: 'thr', intensity: 1 },
        { panel: 'equation', id: 'lda-solution', intensity: 1 },
      ],
      metrics: m,
      events,
      timeline,
    };
  }

  const narration =
    `Direction θ = ${ev.angleDeg.toFixed(0)}° (axis ${step}/${ANGLE_COUNT}): J = ${ev.jFisher.toFixed(4)}, ` +
    `between-class gap |μ̄₁−μ̄₀| = ${ev.betweenGap.toFixed(4)}, ` +
    `within-class variances s₀² = ${ev.withinVar0.toFixed(4)} / s₁² = ${ev.withinVar1.toFixed(4)}; ` +
    `threshold τ = ${ev.threshold.toFixed(3)} → training error ${(ev.trainError * 100).toFixed(1)}%.`;
  const math: MathStep[] = [
    { latex: 'z_i = \\hat{w}(\\theta)^T x_i', id: 'lda-projection' },
    { latex: 'J(\\theta) = \\frac{(\\bar\\mu_1 - \\bar\\mu_0)^2}{s_0^2 + s_1^2}', id: 'lda-fisher-criterion' },
  ];
  return {
    algorithm: {
      mode: 'lda-sweep', step, isOptimal: 0,
      wx: ev.wx, wy: ev.wy, b: ev.b,
      thetaAxisDeg: ev.angleDeg, jFisher: ev.jFisher,
      threshold: ev.threshold, dataSeed: sweep.data.dataSeed,
    } as Record<string, ParamValue>,
    visuals: [...buildScatter(sweep.data, sweep, ev, false), ...buildMatrices(sweep, ev, false)],
    math,
    narration,
    explanation: {
      changed: first
        ? ['axis initialized', `J → ${ev.jFisher.toFixed(4)}`]
        : [`axis → θ = ${ev.angleDeg.toFixed(0)}°`, `J → ${ev.jFisher.toFixed(4)}`, `error → ${(ev.trainError * 100).toFixed(1)}%`],
      why: first
        ? `The run starts with the projection axis at θ = 0°: every point is orthogonally projected onto the candidate line and the Fisher criterion J = between-class gap² / within-class variance is evaluated for this direction.`
        : `The axis advances by Δθ = 5°. For each candidate direction the points are re-projected, the threshold τ = (μ̄₀+μ̄₁)/2 is re-placed between the projected class means, and J — the separation/compactness ratio — is re-measured.`,
      formulaRef: 'lda-fisher-criterion',
      dependsOn: ['linear-algebra', 'probability', 'statistics'],
      gateConcepts: ['LDA', 'Fisher criterion', 'projection', 'dimensionality reduction'],
    },
    highlights: first ? [] : [{ panel: 'canvas', id: 'axis', intensity: 0.30 }],
    metrics: m,
    events,
    timeline,
  };
}

function fmtMat(M: number[][]): string {
  return `[[${M[0][0].toFixed(3)}, ${M[0][1].toFixed(3)}],[${M[1][0].toFixed(3)}, ${M[1][1].toFixed(3)}]]`;
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

export const simulation = {
  /**
   * Snapshot 1 = grid direction 0°. Throws (honest telemetry failure) when S_W
   * is singular — computeRun records failedAtStep, mirroring svm-hard-margin.
   */
  initialState: (p: Params): SimState => {
    const sweep = getSweep(p);
    return snapshotAt(sweep, sweep.evals[0], 1, false, true);
  },

  /**
   * Advance the angle sweep; after the last grid direction (35) emit the
   * closed-form LDA final snapshot; then null.
   */
  step: (p: Params, s: SimState): SimState | null => {
    const sweep = getSweep(p);
    const current = (s.algorithm.step as number) ?? 1;
    const next = current + 1;
    if (next <= ANGLE_COUNT) {
      return snapshotAt(sweep, sweep.evals[next - 1], next, false, false);
    }
    if (next === ANGLE_COUNT + 1) {
      // final snapshot: the closed-form solution on the OPTIMAL axis
      const sOpt = sweep.stats;
      const optEval: AngleEval = {
        theta: sOpt.thetaAxis, angleDeg: sOpt.thetaAxisDeg,
        wx: sOpt.wx, wy: sOpt.wy,
        muBar0: sOpt.wx * sOpt.mu0[0] + sOpt.wy * sOpt.mu0[1],
        muBar1: sOpt.wx * sOpt.mu1[0] + sOpt.wy * sOpt.mu1[1],
        betweenGap: Math.abs(sOpt.wx * sOpt.d[0] + sOpt.wy * sOpt.d[1]),
        withinVar0: sOpt.withinVar0, withinVar1: sOpt.withinVar1,
        jFisher: sOpt.jOpt, threshold: sOpt.threshold, b: sOpt.b,
        trainError: sOpt.trainError,
      };
      return snapshotAt(sweep, optEval, ANGLE_COUNT + 1, true, false);
    }
    return null;
  },
};

// ---------------------------------------------------------------------------
// Topic module + registration
// ---------------------------------------------------------------------------

export const ldaModule: TopicModule = {
  id: 'lda',
  title: "Fisher's LDA",
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 5, coding: 2, visualization: 3, gateFrequency: 4 },
    estimatedHours: 6,
    revisionPriority: 'P1',
    examFrequency: 'Frequent',
    prerequisites: ['linear-algebra', 'probability', 'statistics', 'pca'],
    relatedTopics: ['pca', 'logistic-regression', 'naive-bayes', 'svm-hard-margin', 'perceptron'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Data, Candidate Axis, Projection Guides & Threshold' },
      { slot: 'primary', component: 'decision-boundary', title: 'LDA Decision Boundary: Thresholded Projection Line' },
      { slot: 'primary', component: 'loss-curve', title: 'Fisher Criterion J(θ) over the Direction Sweep (higher = better; max at the LDA axis)' },
      { slot: 'sidebar', component: 'matrix-animator', title: 'S_W, S_B, ŵ & J(θ) — final step adds S_W⁻¹, w, λ' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivations: J → Eigenproblem, Closed Form, Threshold' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Data → Scatter → Project → Evaluate → LDA Solution' },
      { slot: 'sidebar', component: 'explain-step', title: 'Step Explanation' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'nPerClass', label: 'Points per class', type: 'number', min: 3, max: 40, step: 1, default: 15 },
    { id: 'separation', label: 'Cluster separation (center gap)', type: 'number', min: 0.2, max: 4, step: 0.1, default: 2 },
    { id: 'covAngleDeg', label: 'Shared-covariance angle φ (deg)', type: 'number', min: 0, max: 175, step: 5, default: 30 },
    { id: 'covShape', label: 'Shared-covariance shape σ_major/σ_minor', type: 'number', min: 1, max: 4, step: 0.25, default: 2 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
  ],
  simulation,
  formulas: ldaFormulas,
  derivations: ldaDerivations,
  questions: ldaQuestions,
  comparisons: ldaComparisons,
  failureDemos: ldaFailureDemos,
  mistakes: ldaMistakes,
  testCases: ldaTestCases,
  lossMetricKey: 'jFisher',

  validateParams: (p) => {
    const issues: string[] = [];
    const nPerClass = p.nPerClass as number | undefined;
    if (nPerClass !== undefined) {
      if (!Number.isInteger(nPerClass) || nPerClass < 3) {
        issues.push('nPerClass must be an integer ≥ 3 — each class needs at least 3 points so the per-class covariance is full-rank (≥ 2 would give a rank-1 within-class scatter, singular S_W)');
      }
      if (nPerClass > 40) issues.push('nPerClass > 40 exceeds the lightweight demo size (keep n ≤ 40 for smooth scrubbing)');
    }
    const sep = p.separation as number | undefined;
    if (sep !== undefined && !Number.isFinite(sep)) issues.push('separation must be a finite number');
    if (sep !== undefined && sep <= 0) issues.push('separation must be positive (the signed gap between the cluster centers)');
    const ang = p.covAngleDeg as number | undefined;
    if (ang !== undefined && !(ang >= 0 && ang < 180)) issues.push('covAngleDeg must be in [0, 180) — the shared-covariance rotation is defined mod 180°');
    const shape = p.covShape as number | undefined;
    if (shape !== undefined && !(shape >= 1 && shape <= 4)) issues.push('covShape (σ_major/σ_minor) must be in [1, 4]');
    const seed = p.seed as number | undefined;
    if (seed !== undefined && (!Number.isInteger(seed) || seed < 0 || seed > 9999)) issues.push('seed must be an integer in [0, 9999]');
    if (typeof p.points === 'string') {
      try {
        const rows = JSON.parse(p.points) as unknown;
        if (!Array.isArray(rows) || rows.length === 0 || !rows.every((r) => Array.isArray(r) && r.length === 3 && r.every(Number.isFinite))) {
          issues.push('points must be a JSON array of [x, y, cls] triples');
        } else {
          const classes = new Set((rows as [number, number, number][]).map((r) => r[2]));
          if (classes.size !== 2 || ![...classes].every((c) => c === 0 || c === 1)) {
            issues.push('points must contain exactly the two classes 0 and 1');
          }
          const count = (c: number) => (rows as [number, number, number][]).filter((r) => r[2] === c).length;
          if (count(0) < 3 || count(1) < 3) {
            issues.push('points needs ≥ 3 samples per class — fewer leaves the within-class scatter rank-deficient (singular S_W)');
          }
        }
      } catch {
        issues.push('points must be a valid JSON array of [x, y, cls] triples');
      }
    }
    if (p.toy !== undefined && p.toy !== true) issues.push('toy must be true (test-only hand-derivable dataset)');
    return issues;
  },
};

export function register() {
  registerTopic(ldaModule);
  // DecisionBoundary resolves this via getClassifier('lda') to paint class
  // regions + the threshold line (2500 calls per snapshot — reading wx/wy/b
  // off the merged params keeps it a score+sign per cell, no re-solving).
  registerClassifier(ldaModule.id, classifyByParams);
}