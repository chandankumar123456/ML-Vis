// src/topics/pca/module.ts
// Task 16 (Wave 4): pca — Principal Component Analysis as an UNSUPERVISED
// 2-D rotation of a correlated Gaussian cloud into its variance-maximizing
// directions, solved with the EXACT 2×2 closed-form eigen-decomposition
// (characteristic polynomial / rotation-angle formula — no iterative solver).
//
// Design decisions (deviations from the plan are documented in the report):
//  - Data: a single 2-D Gaussian cloud N(μ, Σ) with a NONZERO fixed mean
//    μ = (2.5, 1.5). The offset exists so "forgetting to center" is a REAL,
//    measurable mistake: the raw (uncentered) covariance (1/n)Σ xᵢxᵢᵀ is then
//    dominated by the μμᵀ term and its eigenvectors point at the mean instead
//    of at the data's variance structure (test case 4 measures the contrast).
//    The population covariance is
//      Σ = R(φ)·[[1, ρ],[ρ, 1]]·R(φ)ᵀ + σ²·I
//    (correlation ρ ∈ (−1,1), rotation φ, isotropic noise σ²) — drawn exactly
//    via its Cholesky factor L = R(φ)·chol([[1,ρ],[ρ,1]]) plus σ·N(0,1) per
//    axis, all from a seeded mulberry32 stream. Adding σ²I SHIFTS the
//    eigenvalues (λ₁ = 1+ρ+σ², λ₂ = 1−ρ+σ²) but never rotates the
//    eigenvectors, so PC1 stays at the rotation angle φ.
//  - Eigen-solve: the EXACT 2×2 symmetric closed form. For Σ = [[a,b],[b,c]]:
//      Δ = (a−c)² + 4b²,  λ₁,₂ = (a+c ± √Δ)/2,
//      θ = ½·atan2(2b, a−c)  (major-axis angle),
//      v₁ = (cos θ, sin θ),  v₂ = (−sin θ, cos θ).
//    v₂ is the exact 90° rotation of v₁, so v₁·v₂ = 0 BY CONSTRUCTION (to
//    machine precision — asserted in the tests) and the decomposition is
//    verified against Σv = λv at 1e-9. Mirrors lda's adjugate closed-form
//    style: no iteration anywhere.
//  - Step model (mandated "sweep + final exact snapshot", the lda precedent):
//    36 candidate axes θ_k = k·5° over [0°, 180°) — one snapshot per direction
//    — followed by ONE final snapshot carrying the closed-form PCA solution.
//    Each sweep snapshot projects every point onto the candidate axis
//    (eigenviewer 'axis' + 'projection' commands) and reports the Rayleigh
//    quotient u(θ)ᵀΣu(θ) — the variance along that direction — plus the
//    running best so far. The variance curve peaks at the PC1 direction; the
//    final snapshot evaluates the exact eigen-solution (λ₁, λ₂, ratio,
//    reconstruction k=1 vs k=2) and lands ON that peak. lossMetricKey =
//    'axisVariance' (higher = better — PCA maximizes variance; the layer title
//    states the convention explicitly, the lda 'jFisher' precedent).
//  - Reconstruction honesty: reconErrK1 = λ₂ and reconErrK2 = 0 are computed
//    from the empirical centered covariance and its eigen-decomposition, so
//    the identity "error = sum of dropped eigenvalues" holds to ~1e-12 (the
//    sample satisfies it exactly, not just the population).
//  - Degenerate data: validateParams warns when noise = 0 and |corr| ≥ 0.98
//    (population λ₂ = 1−|ρ| → 0 — PC2 carries no information); a test-only
//    `points` override with zero variance (all points identical) makes the
//    eigen-solve THROW from getSweep → computeRun records the honest telemetry
//    failure (the svm/lda precedent) instead of emitting NaN.
//  - Unsupervised: no labels, no classifier registration. The `points`
//    override (JSON '[[x,y],…]') is test-only and is what the failure demos
//    use for hand-crafted datasets (outliers, unscaled features).
import type { TopicModule, Params, SimState, VisualCommand, ParamValue } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { pcaTestCases } from './testCases';
import { pcaFormulas } from './formulas';
import { pcaDerivations } from './derivations';
import { pcaMistakes } from './mistakes';
import { pcaQuestions } from './questions';
import { pcaComparisons } from './comparisons';
import { pcaFailureDemos } from './failures';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Sweep grid: 36 directions over [0, π) = 5° steps (lda's convention) — must
// divide π exactly so the sweep is uniform and the LAST grid angle is 175°.
export const ANGLE_COUNT = 36;
export const ANGLE_STEP_DEG = 5;

// Fixed NONZERO data mean. Deliberate: the uncentered-vs-centered contrast
// (test case 4) needs the raw covariance to differ from the centered one —
// with μ = 0 they would coincide and "forgetting to center" would be
// invisible. ‖μ‖² = 8.5 vs the population spread ≈ 2.05 makes the raw PC1
// point at the mean instead of at the variance structure (measured on the
// rotDeg-80 config: raw PC1 at 33.30° vs the mean direction 34.06°).
export const DATA_OFFSET: [number, number] = [2.5, 1.5];

// Visual-semantic colors (data cloud + axes).
export const POINT_COLOR = '#2563eb';    // single unlabeled cloud (PCA is unsupervised)
export const AXIS_COLOR = '#64748b';     // candidate sweep axis
export const PC1_COLOR = '#dc2626';      // first principal component (final)
export const PC2_COLOR = '#16a34a';      // second principal component (final)

// ---------------------------------------------------------------------------
// Deterministic PRNG + data synthesis
// ---------------------------------------------------------------------------

export interface PcaPoint { x: number; y: number; }

/** Mulberry32 — deterministic PRNG (matches every Wave-1/2/3/4 topic). */
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
 * Deterministic data synthesis — a single correlated Gaussian cloud:
 *   xᵢ ~ N(DATA_OFFSET, Σ),  Σ = R(φ)·[[1,ρ],[ρ,1]]·R(φ)ᵀ + σ²I.
 * Drawn exactly via the Cholesky factor L = R(φ)·chol([[1,ρ],[ρ,1]]) with
 * σ·N(0,1) added per axis (the isotropic noise). Test-only overrides:
 * `points` (JSON '[[x,y],…]') → a hand-crafted dataset (failure demos).
 */
export function generateData(p: Params): PcaPoint[] {
  if (typeof p.points === 'string') {
    const rows: [number, number][] = JSON.parse(p.points);
    return rows.map(([x, y]) => ({ x, y }));
  }
  const n = (p.n as number) ?? 40;
  const corr = (p.corr as number) ?? 0.7;
  const rotDeg = (p.rotDeg as number) ?? 30;
  const noise = (p.noise as number) ?? 0.15;
  const seed = (p.seed as number) ?? 42;
  const rng = mulberry32(seed);
  const normal = makeNormal(rng);
  const phi = (rotDeg * Math.PI) / 180;
  const cos = Math.cos(phi), sin = Math.sin(phi);
  const s = Math.sqrt(Math.max(0, 1 - corr * corr));
  // L = R(φ)·chol([[1,ρ],[ρ,1]]) = [[cos−ρ·sin, −s·sin],[sin+ρ·cos, s·cos]]
  const l11 = cos - corr * sin, l12 = -s * sin;
  const l21 = sin + corr * cos, l22 = s * cos;
  const pts: PcaPoint[] = [];
  for (let i = 0; i < n; i++) {
    const g1 = normal(), g2 = normal();
    const x = l11 * g1 + l12 * g2 + (noise > 0 ? noise * normal() : 0);
    const y = l21 * g1 + l22 * g2 + (noise > 0 ? noise * normal() : 0);
    pts.push({ x: x + DATA_OFFSET[0], y: y + DATA_OFFSET[1] });
  }
  return pts;
}

// ---------------------------------------------------------------------------
// PCA math core (all mutation-free, exported for hand-verified tests)
// ---------------------------------------------------------------------------

/** Arithmetic mean of the points; [0, 0] for an empty list. */
export function meanOf(points: PcaPoint[]): [number, number] {
  const n = Math.max(points.length, 1);
  let sx = 0, sy = 0;
  for (const d of points) { sx += d.x; sy += d.y; }
  return [sx / n, sy / n];
}

/**
 * CENTERED covariance Σ = (1/n)·XᵀX with X the mean-subtracted design matrix
 * — the PCA covariance. Sample version: (1/n)·Σᵢ (xᵢ−μ)(xᵢ−μ)ᵀ.
 */
export function centeredCov(points: PcaPoint[], mu: [number, number]): number[][] {
  const n = Math.max(points.length, 1);
  let s11 = 0, s12 = 0, s22 = 0;
  for (const d of points) {
    const dx = d.x - mu[0], dy = d.y - mu[1];
    s11 += dx * dx; s12 += dx * dy; s22 += dy * dy;
  }
  return [[s11 / n, s12 / n], [s12 / n, s22 / n]];
}

/**
 * RAW (UNcentered) covariance (1/n)·Σᵢ xᵢxᵢᵀ — the "forgot to center" mistake.
 * With μ ≠ 0 this is Σ_centered + μμᵀ, whose dominant eigenvector points at
 * the MEAN, not at the data's variance structure (measured in test case 4).
 */
export function rawCov(points: PcaPoint[]): number[][] {
  const n = Math.max(points.length, 1);
  let s11 = 0, s12 = 0, s22 = 0;
  for (const d of points) {
    s11 += d.x * d.x; s12 += d.x * d.y; s22 += d.y * d.y;
  }
  return [[s11 / n, s12 / n], [s12 / n, s22 / n]];
}

export interface Eigen2x2 {
  lambda1: number;               // largest eigenvalue (variance along v₁)
  lambda2: number;               // smallest eigenvalue (variance along v₂)
  v1: [number, number];          // PC1 unit vector
  v2: [number, number];          // PC2 unit vector (exact 90° rotation of v₁)
  angleDeg: number;              // PC1 axis label ∈ [0, 180)
  totalVariance: number;         // trace Σ = λ₁ + λ₂
}

/**
 * EXACT 2×2 symmetric eigen-decomposition via the rotation-angle closed form:
 *   λ₁,₂ = (a+c ± √((a−c)² + 4b²))/2,   θ = ½·atan2(2b, a−c),   v₁ = (cosθ, sinθ).
 * v₂ = R(90°)v₁ so the PCs are orthogonal BY CONSTRUCTION. Throws an honest
 * error when the data has zero variance (the covariance is the zero matrix) —
 * computeRun then records the failure via telemetry.
 */
export function eigen2x2(M: number[][]): Eigen2x2 {
  const a = M[0][0], b = M[0][1], c = M[1][1];
  const t = a + c;                                   // trace
  const delta = Math.max(0, (a - c) * (a - c) + 4 * b * b); // (a−c)² + 4b² ≥ 0
  const root = Math.sqrt(delta);
  const lambda1 = (t + root) / 2;
  const lambda2 = (t - root) / 2;
  if (lambda1 + lambda2 < 1e-12) {
    throw new Error(
      'pca: zero-variance data — the covariance matrix is the zero matrix (all points identical); ' +
      'PCA needs some spread to find a direction',
    );
  }
  let angle = 0.5 * Math.atan2(2 * b, a - c);        // major-axis angle
  if (angle < 0) angle += Math.PI;                   // line angle in [0, π)
  return {
    lambda1, lambda2,
    v1: [Math.cos(angle), Math.sin(angle)],
    v2: [-Math.sin(angle), Math.cos(angle)],         // 90° rotation ⇒ v₁·v₂ = 0 exactly
    angleDeg: (angle * 180) / Math.PI,
    totalVariance: t,
  };
}

/** Rayleigh quotient uᵀΣu — the empirical variance of projections onto unit u. */
export function quadForm(M: number[][], u: [number, number]): number {
  return u[0] * (M[0][0] * u[0] + M[0][1] * u[1]) + u[1] * (M[1][0] * u[0] + M[1][1] * u[1]);
}

/** Empirical variance of the projections of `points` onto the unit direction u. */
export function varianceAlong(points: PcaPoint[], mu: [number, number], u: [number, number]): number {
  const n = Math.max(points.length, 1);
  let sum = 0, sum2 = 0;
  for (const d of points) {
    const t = u[0] * (d.x - mu[0]) + u[1] * (d.y - mu[1]);
    sum += t; sum2 += t * t;
  }
  return Math.max(0, sum2 / n - (sum / n) ** 2);
}

/**
 * Mean squared reconstruction error after keeping `k` principal components
 * (1 or 2): (1/n)·Σᵢ ‖xᵢ − (μ + Σ_{j≤k} (vⱼ·(xᵢ−μ))vⱼ)‖². Mathematically
 * equals the sum of the DROPPED eigenvalues (λ₂ for k=1, 0 for k=2) — asserted
 * in the tests to ~1e-12.
 */
export function reconError(points: PcaPoint[], mu: [number, number], eig: Eigen2x2, k: number): number {
  const n = Math.max(points.length, 1);
  const vs: [number, number][] = k >= 1 ? [eig.v1] : [];
  if (k >= 2) vs.push(eig.v2);
  let err = 0;
  for (const d of points) {
    const dx = d.x - mu[0], dy = d.y - mu[1];
    let rx = dx, ry = dy;
    for (const v of vs) {
      const t = v[0] * dx + v[1] * dy;
      rx -= t * v[0]; ry -= t * v[1];
    }
    err += rx * rx + ry * ry;
  }
  return err / n;
}

// ---------------------------------------------------------------------------
// The sweep (candidate axes) + closed form — the run's single source
// ---------------------------------------------------------------------------

export interface AxisEval {
  angleDeg: number;            // axis label ∈ [0, 180) in degrees
  ux: number; uy: number;      // unit direction u(θ)
  variance: number;            // uᵀΣu — the Rayleigh quotient (loss-curve metric)
  explainedFraction: number;   // variance / totalVariance
}

export interface PcaSweep {
  data: PcaPoint[];
  dataSeed: number;
  mu: [number, number];
  Sigma: number[][];           // centered covariance (1/n)XᵀX
  eig: Eigen2x2;               // exact closed-form decomposition (throws when degenerate)
  angles: number[];            // θ_k = k·π/36, k = 0..35 (radians)
  evals: AxisEval[];           // per grid angle
}

export function getSweep(p: Params): PcaSweep {
  const data = generateData(p);
  const dataSeed = (p.seed as number) ?? 42;
  const mu = meanOf(data);
  const Sigma = centeredCov(data, mu);
  const eig = eigen2x2(Sigma);
  const angles = Array.from({ length: ANGLE_COUNT }, (_, k) => (k * Math.PI) / ANGLE_COUNT);
  const evals = angles.map((theta) => {
    const ux = Math.cos(theta), uy = Math.sin(theta);
    const variance = quadForm(Sigma, [ux, uy]);
    return {
      angleDeg: (theta * 180) / Math.PI,
      ux, uy,
      variance,
      explainedFraction: variance / Math.max(eig.totalVariance, 1e-12),
    };
  });
  return { data, dataSeed, mu, Sigma, eig, angles, evals };
}

// Bounded memoization (the svm precedent): initialState/step stay O(1) after
// the first evaluation of a params key.
const SWEEP_CACHE = new Map<string, PcaSweep>();
const SWEEP_CACHE_MAX = 16;

function sweepKey(p: Params): string {
  return JSON.stringify([p.n, p.corr, p.rotDeg, p.noise, p.seed, p.points ?? null]);
}

export function cachedSweep(p: Params): PcaSweep {
  const key = sweepKey(p);
  let sw = SWEEP_CACHE.get(key);
  if (!sw) {
    sw = getSweep(p);
    if (SWEEP_CACHE.size >= SWEEP_CACHE_MAX) SWEEP_CACHE.clear();
    SWEEP_CACHE.set(key, sw);
  }
  return sw;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

function sweepMetrics(sweep: PcaSweep, ev: AxisEval, step: number): Record<string, number> {
  // running best over the directions seen SO FAR (steps 1..step) — honest
  // "the best candidate so far" narration for the sweep.
  let bestVariance = -Infinity;
  let bestAngleDeg = ev.angleDeg;
  for (let k = 0; k < step && k < sweep.evals.length; k++) {
    const e = sweep.evals[k];
    if (e.variance > bestVariance) { bestVariance = e.variance; bestAngleDeg = e.angleDeg; }
  }
  return {
    step, angleDeg: ev.angleDeg,
    axisVariance: ev.variance,
    explainedFraction: ev.explainedFraction,
    bestVariance, bestAngleDeg,
    totalVariance: sweep.eig.totalVariance,
    n: sweep.data.length,
    dataSeed: sweep.dataSeed,
    isOptimal: 0,
  };
}

function finalMetrics(sweep: PcaSweep): Record<string, number> {
  const { eig } = sweep;
  const e1 = eig.lambda1 / Math.max(eig.totalVariance, 1e-12);
  const e2 = eig.lambda2 / Math.max(eig.totalVariance, 1e-12);
  return {
    step: ANGLE_COUNT + 1,
    angleDeg: eig.angleDeg,
    axisVariance: eig.lambda1,                 // the sweep's peak — lands ON the loss-curve max
    lambda1: eig.lambda1,
    lambda2: eig.lambda2,
    explainedRatio1: e1,
    explainedRatio2: e2,
    reconErrK1: eig.lambda2,                   // dropped eigenvalue λ₂ — asserted in tests
    reconErrK2: 0,                             // both PCs reconstruct exactly
    totalVariance: eig.totalVariance,
    n: sweep.data.length,
    dataSeed: sweep.dataSeed,
    isOptimal: 1,
  };
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

function dataBounds(points: PcaPoint[]): { x0: number; x1: number; y0: number; y1: number } {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const d of points) {
    x0 = Math.min(x0, d.x); x1 = Math.max(x1, d.x);
    y0 = Math.min(y0, d.y); y1 = Math.max(y1, d.y);
  }
  return { x0, x1, y0, y1 };
}

/**
 * Scatter-plot commands (registry-rendered): the data cloud, the candidate
 * axis line through the data mean, a direction arrow, and (final snapshot
 * only) the two PC axes. Per-point projection guides/lines live in the
 * eigenviewer commands (buildEigenviewer) — no duplication.
 */
function buildScatter(sweep: PcaSweep, axis: { ux: number; uy: number; color: string }, isFinal: boolean): VisualCommand[] {
  const { data, mu } = sweep;
  const cmd: VisualCommand[] = data.map((d, i) => ({
    type: 'point', id: `d${i}`, x: d.x, y: d.y, color: POINT_COLOR,
  }));
  const { x0, x1, y0, y1 } = dataBounds(data);
  const line = clipRay(mu, [axis.ux, axis.uy], x0, x1, y0, y1);
  if (line) cmd.push({ type: 'line', id: 'axis', points: line, color: axis.color });
  cmd.push({
    type: 'arrow', id: 'dir', color: axis.color,
    x1: mu[0], y1: mu[1],
    x2: mu[0] + axis.ux * 1.1, y2: mu[1] + axis.uy * 1.1,
  });
  if (isFinal) {
    const e = sweep.eig;
    const p1 = clipRay(mu, e.v1, x0, x1, y0, y1);
    const p2 = clipRay(mu, e.v2, x0, x1, y0, y1);
    if (p1) cmd.push({ type: 'line', id: 'pc1', points: p1, color: PC1_COLOR });
    if (p2) cmd.push({ type: 'line', id: 'pc2', points: p2, color: PC2_COLOR });
    cmd.push({ type: 'arrow', id: 'pc1-arrow', color: PC1_COLOR, x1: mu[0], y1: mu[1], x2: mu[0] + e.v1[0] * 1.1, y2: mu[1] + e.v1[1] * 1.1 });
    cmd.push({ type: 'arrow', id: 'pc2-arrow', color: PC2_COLOR, x1: mu[0], y1: mu[1], x2: mu[0] + e.v2[0] * 1.1, y2: mu[1] + e.v2[1] * 1.1 });
  }
  return cmd;
}

/**
 * Eigenviewer commands: the data cloud ('point'), the candidate axis
 * ('axis' — angle in radians through the data centroid), and per-point
 * orthogonal projections ('projection': from = data point, onto = projection
 * on the axis, residual = perpendicular distance). The view renders the
 * variance-explained bars itself from the point commands.
 */
function buildEigenviewer(sweep: PcaSweep, u: [number, number], color: string): VisualCommand[] {
  const { data, mu } = sweep;
  const cmd: VisualCommand[] = data.map((d, i) => ({
    type: 'point', id: `d${i}`, x: d.x, y: d.y, color: POINT_COLOR,
  }));
  const angle = Math.atan2(u[1], u[0]);
  cmd.push({ type: 'axis', id: 'axis', angle, color });
  data.forEach((d, i) => {
    const t = u[0] * (d.x - mu[0]) + u[1] * (d.y - mu[1]);
    const px = mu[0] + u[0] * t;
    const py = mu[1] + u[1] * t;
    cmd.push({
      type: 'projection', id: `proj${i}`,
      point: [d.x, d.y], onto: [px, py],
      residual: Math.hypot(d.x - px, d.y - py),
    });
  });
  return cmd;
}

/** Matrix story for matrix-animator: Σ, the candidate axis, uᵀΣu; final adds λ, V, reconstruction. */
function buildMatrices(sweep: PcaSweep, ev: AxisEval | null, isFinal: boolean): VisualCommand[] {
  const { Sigma } = sweep;
  const cmds: VisualCommand[] = [
    { type: 'matrix', id: 'Σ = (1/n)XᵀX', rows: 2, cols: 2, cells: Sigma },
  ];
  if (ev) {
    cmds.push(
      { type: 'matrix', id: 'u(θ)', rows: 2, cols: 1, cells: [[ev.ux], [ev.uy]] },
      { type: 'matrix', id: 'uᵀΣu', rows: 1, cols: 1, cells: [[ev.variance]] },
    );
  }
  if (isFinal) {
    const e = sweep.eig;
    cmds.push(
      { type: 'matrix', id: 'λ', rows: 2, cols: 1, cells: [[e.lambda1], [e.lambda2]] },
      { type: 'matrix', id: 'V = [v₁ v₂]', rows: 2, cols: 2, cells: [[e.v1[0], e.v2[0]], [e.v1[1], e.v2[1]]] },
      { type: 'matrix', id: 'recon err k=1', rows: 1, cols: 1, cells: [[e.lambda2]] },
    );
  }
  return cmds;
}

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

function fmtMat(M: number[][]): string {
  return `[[${M[0][0].toFixed(3)}, ${M[0][1].toFixed(3)}],[${M[1][0].toFixed(3)}, ${M[1][1].toFixed(3)}]]`;
}

function sweepSnapshot(sweep: PcaSweep, ev: AxisEval, step: number, first: boolean): SimState {
  const m = sweepMetrics(sweep, ev, step);
  const u: [number, number] = [ev.ux, ev.uy];
  const events: SimState['events'] = first
    ? [
        { type: 'init', label: 'pca-seeded-correlated-gaussian', step },
        { type: 'candidate', label: `direction-g${step}-of-${ANGLE_COUNT}`, step },
      ]
    : [{ type: 'candidate', label: `direction-g${step}-of-${ANGLE_COUNT}`, step }];
  const narration =
    `Candidate axis θ = ${ev.angleDeg.toFixed(0)}° (step ${step}/${ANGLE_COUNT}): ` +
    `variance along the axis uᵀΣu = ${ev.variance.toFixed(4)} (${(ev.explainedFraction * 100).toFixed(1)}% of the total ${m.totalVariance.toFixed(4)}). ` +
    `Best direction so far: θ = ${m.bestAngleDeg.toFixed(0)}° with variance ${m.bestVariance.toFixed(4)} — ` +
    `keep rotating to find the direction of MAXIMUM variance (the first principal component).`;
  return {
    algorithm: {
      mode: 'pca-sweep', step, angleDeg: ev.angleDeg,
      ux: ev.ux, uy: ev.uy, dataSeed: sweep.dataSeed,
    } as Record<string, ParamValue>,
    visuals: [
      ...buildEigenviewer(sweep, u, AXIS_COLOR),
      ...buildScatter(sweep, { ux: ev.ux, uy: ev.uy, color: AXIS_COLOR }, false),
      ...buildMatrices(sweep, ev, false),
    ],
    math: [
      { latex: '\\Sigma = \\frac{1}{n}X_c^T X_c', id: 'pca-covariance' },
      { latex: 'R(u) = u^T \\Sigma u = \\operatorname{Var}(\\text{projections onto } u)', id: 'pca-rayleigh' },
    ],
    narration,
    explanation: {
      changed: first
        ? ['data centered', `Σ = ${fmtMat(sweep.Sigma)}`, `axis initialized at θ = 0°`, `uᵀΣu → ${ev.variance.toFixed(4)}`]
        : [`axis → θ = ${ev.angleDeg.toFixed(0)}°`, `uᵀΣu → ${ev.variance.toFixed(4)}`, `best so far → ${m.bestAngleDeg.toFixed(0)}° (${m.bestVariance.toFixed(4)})`],
      why: first
        ? `The run starts by centering the cloud at its mean μ and forming the empirical covariance Σ = (1/n)XᵀX — the matrix that encodes how the data spreads. The candidate axis starts at θ = 0° (the x-direction); every point is orthogonally projected onto it and the variance of those projections uᵀΣu is measured — the Rayleigh quotient. PCA's whole question: WHICH direction maximizes this variance?`
        : `The axis advances by Δθ = 5°. Each candidate direction re-projects every point and re-measures uᵀΣu. The variance curve over the sweep peaks at the direction the cloud actually stretches along — the first principal component. Rotating toward that peak is the sweep's goal.`,
      formulaRef: 'pca-rayleigh',
      dependsOn: ['linear-algebra', 'probability', 'statistics'],
      gateConcepts: ['PCA', 'principal component', 'covariance', 'projection', 'variance'],
    },
    highlights: first ? [] : [{ panel: 'canvas', id: 'axis', intensity: 0.30 }],
    metrics: m,
    events,
    timeline: first ? ['Data', 'Center', 'Covariance', 'Project', 'Evaluate'] : ['Project', 'Evaluate'],
  };
}

function finalSnapshot(sweep: PcaSweep): SimState {
  const m = finalMetrics(sweep);
  const e = sweep.eig;
  const narration =
    `Closed-form PCA: centered covariance Σ = ${fmtMat(sweep.Sigma)} — characteristic polynomial (a−λ)(c−λ)−b² = 0 gives ` +
    `λ₁ = ${e.lambda1.toFixed(4)}, λ₂ = ${e.lambda2.toFixed(4)}; the rotation-angle formula θ = ½·atan2(2b, a−c) gives ` +
    `PC1 = (${e.v1[0].toFixed(4)}, ${e.v1[1].toFixed(4)}) at θ₁ = ${e.angleDeg.toFixed(2)}° (exact, no iteration — verified Σv = λv to 1e-9). ` +
    `PC1 explains ${(m.explainedRatio1 * 100).toFixed(1)}% of the variance (PC2: ${(m.explainedRatio2 * 100).toFixed(1)}%). ` +
    `Reconstruction: keeping PC1 only loses the dropped eigenvalue λ₂ = ${e.lambda2.toFixed(4)} per sample on average ` +
    `(recon error k=1 = ${m.reconErrK1.toFixed(4)}); keeping both PCs reconstructs exactly (error 0).`;
  return {
    algorithm: {
      mode: 'pca-optimal', step: ANGLE_COUNT + 1, isOptimal: 1,
      angleDeg: e.angleDeg, v1x: e.v1[0], v1y: e.v1[1], v2x: e.v2[0], v2y: e.v2[1],
      lambda1: e.lambda1, lambda2: e.lambda2, dataSeed: sweep.dataSeed,
    } as Record<string, ParamValue>,
    visuals: [
      ...buildEigenviewer(sweep, e.v1, PC1_COLOR),
      ...buildScatter(sweep, { ux: e.v1[0], uy: e.v1[1], color: PC1_COLOR }, true),
      ...buildMatrices(sweep, null, true),
    ],
    math: [
      { latex: '\\Sigma v = \\lambda v', id: 'pca-eigenequation' },
      { latex: 'z_i = v_1 \\cdot (x_i - \\mu)', id: 'pca-projection' },
      { latex: '\\hat{x}_i = \\mu + \\sum_{j \\le k} z_{ij} v_j, \\quad \\text{error} = \\frac{1}{n}\\sum_i \\|x_i - \\hat{x}_i\\|^2 = \\sum_{j > k} \\lambda_j', id: 'pca-reconstruction' },
      { latex: '\\frac{\\lambda_k}{\\sum_j \\lambda_j}', id: 'pca-explained' },
    ],
    narration,
    explanation: {
      changed: [
        `axis → PC1 at θ₁ = ${e.angleDeg.toFixed(2)}° (exact, not a grid angle)`,
        `λ₁ = ${e.lambda1.toFixed(4)}, λ₂ = ${e.lambda2.toFixed(4)}`,
        `explained variance → ${(m.explainedRatio1 * 100).toFixed(1)}% / ${(m.explainedRatio2 * 100).toFixed(1)}%`,
        `recon error k=1 → ${m.reconErrK1.toFixed(4)} (= λ₂), k=2 → 0`,
      ],
      why: `The sweep ends: instead of a grid direction, the EXACT closed-form eigen-solution of the centered covariance is evaluated. ` +
        `λ₁ is the maximum of the Rayleigh quotient uᵀΣu — the sweep's curve peaks exactly here. ` +
        `Projecting onto PC1 and back loses only the variance along PC2, which is exactly λ₂; keeping both PCs spans the plane, so the reconstruction is lossless.`,
      formulaRef: 'pca-reconstruction',
      dependsOn: ['linear-algebra', 'probability', 'statistics'],
      gateConcepts: ['PCA', 'eigenvalue', 'eigenvector', 'explained variance', 'reconstruction'],
    },
    highlights: [
      { panel: 'canvas', id: 'axis', intensity: 1 },
      { panel: 'matrix', id: 'λ:0,0', intensity: 1 },
      { panel: 'matrix', id: 'λ:1,0', intensity: 1 },
      { panel: 'equation', id: 'pca-reconstruction', intensity: 1 },
    ],
    metrics: m,
    events: [
      { type: 'converged', label: 'exact-2x2-pca-solution', step: ANGLE_COUNT + 1 },
    ],
    timeline: ['Project', 'Reconstruct', 'PCA Solution'],
  };
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

export const simulation = {
  /**
   * Snapshot 1 = grid direction 0°. Throws (honest telemetry failure) when the
   * data has zero variance (the `points` override degenerate case) — computeRun
   * records failedAtStep, mirroring lda's singular-S_W path.
   */
  initialState: (p: Params): SimState => {
    const sweep = cachedSweep(p);
    return sweepSnapshot(sweep, sweep.evals[0], 1, true);
  },

  /** Advance the angle sweep; after the last grid direction emit the closed-form final; then null. */
  step: (p: Params, s: SimState): SimState | null => {
    const sweep = cachedSweep(p);
    const current = (s.algorithm.step as number) ?? 1;
    const next = current + 1;
    if (next <= ANGLE_COUNT) {
      return sweepSnapshot(sweep, sweep.evals[next - 1], next, false);
    }
    if (next === ANGLE_COUNT + 1) {
      return finalSnapshot(sweep);
    }
    return null;
  },
};

// ---------------------------------------------------------------------------
// Topic module + registration
// ---------------------------------------------------------------------------

export const pcaModule: TopicModule = {
  id: 'pca',
  title: 'Principal Component Analysis',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 4, mathematical: 5, coding: 3, visualization: 4, gateFrequency: 5 },
    estimatedHours: 7,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['linear-algebra', 'probability', 'statistics'],
    relatedTopics: ['pca-svd', 'lda', 'svm-hard-margin', 'linear-algebra'],
    revision: { quick: '20m', standard: '1h', deep: '2h', mastery: '4h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'eigenviewer', title: 'Data Cloud, Rotating Axis, Projections & Variance Bars (PC1 = direction of max variance)' },
      { slot: 'primary', component: 'scatter-plot', title: 'Data & PCA Axes — final step shows both PCs (orthogonal)' },
      { slot: 'primary', component: 'loss-curve', title: 'Variance along the Candidate Axis uᵀΣu over the Rotation Sweep — higher = better, peak = PC1' },
      { slot: 'sidebar', component: 'matrix-animator', title: 'Σ, u(θ), uᵀΣu — final step adds λ, V and the reconstruction error' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivations: wᵀΣw → Lagrange → Σw = λw → Reconstruction Error' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Data → Center → Covariance → Project → Evaluate → PCA Solution' },
      { slot: 'sidebar', component: 'explain-step', title: 'Step Explanation' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'n', label: 'Points n', type: 'number', min: 3, max: 80, step: 1, default: 40 },
    { id: 'corr', label: 'Correlation ρ', type: 'number', min: -0.95, max: 0.95, step: 0.05, default: 0.7 },
    { id: 'rotDeg', label: 'Correlation angle φ (deg)', type: 'number', min: 0, max: 170, step: 5, default: 30 },
    { id: 'noise', label: 'Isotropic noise σ', type: 'number', min: 0, max: 1.5, step: 0.05, default: 0.15 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
  ],
  simulation,
  formulas: pcaFormulas,
  derivations: pcaDerivations,
  questions: pcaQuestions,
  comparisons: pcaComparisons,
  failureDemos: pcaFailureDemos,
  mistakes: pcaMistakes,
  testCases: pcaTestCases,
  lossMetricKey: 'axisVariance',

  validateParams: (p) => {
    const issues: string[] = [];
    const n = p.n as number | undefined;
    if (n !== undefined) {
      if (!Number.isInteger(n) || n < 3) {
        issues.push('n must be an integer ≥ 3 — PCA needs at least 3 non-collinear points so the sample covariance is full-rank (2 points give a rank-1 covariance with a meaningless second PC)');
      }
      if (n > 80) issues.push('n > 80 exceeds the lightweight demo size (keep n ≤ 80 for smooth scrubbing)');
    }
    const corr = p.corr as number | undefined;
    if (corr !== undefined && !Number.isFinite(corr)) issues.push('corr must be a finite number');
    if (corr !== undefined && !(corr > -1 && corr < 1)) {
      issues.push('corr must be in (−1, 1) — |ρ| = 1 makes the base covariance singular (perfect correlation → λ₂ = 0, a degenerate PC2)');
    }
    const rot = p.rotDeg as number | undefined;
    if (rot !== undefined && !(rot >= 0 && rot < 180)) issues.push('rotDeg must be in [0, 180) — the correlation rotation is defined mod 180°');
    const noise = p.noise as number | undefined;
    if (noise !== undefined && !(noise >= 0)) issues.push('noise must be ≥ 0 (0 is allowed — it exposes the near-degenerate λ₂ ≈ 0 case below)');
    if (noise !== undefined && noise > 1.5) issues.push('noise > 1.5 overwhelms the correlation structure (σ²I dominates Σ and the PC direction becomes meaningless)');
    // Degenerate-data warning (honest, not a hard reject): noise = 0 with
    // |corr| near 1 gives population λ₂ = 1−|ρ| → 0 — PC2 carries no variance.
    if (noise === 0 && corr !== undefined && Math.abs(corr) >= 0.98) {
      issues.push('WARNING: noise = 0 with |corr| ≥ 0.98 gives a nearly degenerate covariance (λ₂ ≈ 0) — PC2 carries no information; the eigen-decomposition is still exact but the second PC is meaningless');
    }
    const seed = p.seed as number | undefined;
    if (seed !== undefined && (!Number.isInteger(seed) || seed < 0 || seed > 9999)) issues.push('seed must be an integer in [0, 9999]');
    if (typeof p.points === 'string') {
      try {
        const rows = JSON.parse(p.points) as unknown;
        if (!Array.isArray(rows) || rows.length < 3 || !rows.every((r) => Array.isArray(r) && r.length === 2 && r.every(Number.isFinite))) {
          issues.push('points must be a JSON array of ≥ 3 [x, y] pairs');
        } else {
          // zero-variance guard: all points identical → Σ = 0 → the eigen-solve
          // throws; flagging it here keeps the failure path explicit.
          const pts = (rows as [number, number][]).map(([x, y]) => ({ x, y }));
          const mu0 = pts.reduce((a, d) => a + d.x, 0) / pts.length;
          const mu1 = pts.reduce((a, d) => a + d.y, 0) / pts.length;
          let v = 0;
          for (const d of pts) v += (d.x - mu0) ** 2 + (d.y - mu1) ** 2;
          if (v < 1e-12) issues.push('points have zero variance (all points identical) — PCA needs some spread');
        }
      } catch {
        issues.push('points must be a valid JSON array of [x, y] pairs');
      }
    }
    return issues;
  },
};

export function register() {
  registerTopic(pcaModule);
  // PCA is unsupervised — no classifier to register (contrast with lda/svm).
}
