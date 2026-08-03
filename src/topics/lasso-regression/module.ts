// src/topics/lasso-regression/module.ts
import type { TopicModule, Params, SimState, VisualCommand, ParamValue } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { lassoTestCases } from './testCases';
import { lassoFormulas } from './formulas';
import { lassoDerivations } from './derivations';
import { lassoMistakes } from './mistakes';
import { lassoQuestions } from './questions';
import { lassoComparisons } from './comparisons';
import { lassoFailureDemos } from './failures';
import { mean } from '../../lib/math/linAlg';

export interface LassoData { xs: number[][]; ys: number[]; d: number; }

// Truth model used to synthesize data: y = wTrue·x + bTrue (deterministic per feature count).
// The trailing SMALL coefficients (0.4, 0.2, 0.1) are what lasso zeroes exactly at
// moderate λ — standardized magnitudes ≈ [8.7, 5.8, 4.3, 1.2, 0.6, 0.3].
export const TRUE_W = [3, -2, 1.5, 0.4, 0.2, 0.1];
export const TRUE_B = 1;

// Fixed held-out test point (NOT drawn from the RNG) — lets every run report an honest
// out-of-sample prediction in metrics.testPred / metrics.testTrue.
const TEST_X = [1.5, -0.5, 2.0, 0.8, 1.2, -0.7];

const CONV_TOL = 1e-9;      // relative objective-change threshold for convergence
const MIN_SWEEPS = 2;       // never claim convergence before two full sweeps

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

/**
 * Deterministic data synthesis. Each feature column is an independent draw from
 * U[−5,5] (seeded), so columns are near-orthogonal — the well-conditioned regime
 * where coordinate descent converges fast and soft-threshold boundaries are clean
 * (mirrors the MLR data generator). `p.correlated` is a test-only switch (not in the
 * UI schema): when true, features 2..d are x₁ plus tiny noise, corr ≈ 0.9999 — the
 * lasso failure mode where selection becomes arbitrary.
 */
export function generateData(p: Params): LassoData {
  const n = (p.n as number) ?? 25;
  const d = (p.nFeatures as number) ?? 5;
  const noise = (p.noise as number) ?? 0.5;
  const correlated = p.correlated === true;
  const rng = mulberry32((p.seed as number) ?? 42);
  const xs: number[][] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < d; j++) {
      if (correlated && j > 0) row.push(row[0] + (rng() - 0.5) * 0.1 * j); // xⱼ ≈ x₁ (corr ≈ 0.9999)
      else row.push(-5 + rng() * 10); // i.i.d. U[−5,5] per feature
    }
    let y = TRUE_B;
    for (let j = 0; j < d; j++) y += TRUE_W[j] * row[j];
    y += (rng() - 0.5) * 2 * noise;
    xs.push(row);
    ys.push(y);
  }
  return { xs, ys, d };
}

export interface StdScale { mu: number[]; sigma: number[]; }

/**
 * Per-feature z-score statistics (population variance). Coordinate descent runs in
 * standardized space so the penalty treats every feature equally and Σᵢzᵢⱼ² = n;
 * θ is converted back to original coordinates for metrics/display. The bias is NOT
 * standardized and NOT penalized.
 */
export function standardize(data: LassoData): StdScale {
  const mu: number[] = [];
  const sigma: number[] = [];
  for (let j = 0; j < data.d; j++) {
    const col = data.xs.map((r) => r[j]);
    const m = mean(col);
    mu.push(m);
    sigma.push(Math.sqrt(Math.max(mean(col.map((v) => (v - m) ** 2)), 1e-9)));
  }
  return { mu, sigma };
}

function toStandard(theta: number[], sc: StdScale): number[] {
  const d = sc.mu.length;
  const ws = theta.slice(0, d).map((w, j) => w * sc.sigma[j]);
  // ŷ = Σ wⱼxⱼ + b with xⱼ = σⱼzⱼ + μⱼ ⇒ b̃ = b + Σ wⱼμⱼ (UN-standardized wⱼ · μⱼ).
  // Using w̃ⱼ·μⱼ here breaks the round trip and leaks bias per step — MLR Wave-1 QA bug.
  const bTilde = theta[d] + theta.slice(0, d).reduce((a, w, j) => a + w * sc.mu[j], 0);
  return [...ws, bTilde];
}

function fromStandard(thetaTilde: number[], sc: StdScale): number[] {
  const d = sc.mu.length;
  const ws = thetaTilde.slice(0, d).map((wj, j) => wj / sc.sigma[j]);
  // Inverse of b̃ = b + Σ wⱼμⱼ (wⱼ = w̃ⱼ/σⱼ).
  const b = thetaTilde[d] - ws.reduce((a, w, j) => a + w * sc.mu[j], 0);
  return [...ws, b];
}

/**
 * Soft-threshold operator S(z, λ) = sign(z)·max(|z| − λ, 0).
 * The defining lasso mechanism: when |z| ≤ λ the result is EXACTLY 0 — this is how
 * coordinate descent performs feature selection (a coefficient can never "land near
 * zero"; it is either shifted by λ or clamped to the zero value).
 */
export function softThreshold(z: number, lambda: number): number {
  if (Math.abs(z) <= lambda) return 0;
  return Math.sign(z) * (Math.abs(z) - lambda);
}

/**
 * Objective in standardized space: J = (1/2n)·Σ(y − ŷ)² + λ·Σ_j |θ̃_j|.
 * The bias θ̃_d is NOT penalized. Tracked per step so the UI and tests can watch the
 * monotone decrease of coordinate descent.
 */
export function objectiveStd(thetaTilde: number[], data: LassoData, sc: StdScale, lambda: number): number {
  const { xs, ys, d } = data;
  const n = xs.length;
  let sq = 0;
  for (let i = 0; i < n; i++) {
    let pred = thetaTilde[d];
    for (let j = 0; j < d; j++) pred += thetaTilde[j] * ((xs[i][j] - sc.mu[j]) / sc.sigma[j]);
    const r = pred - ys[i];
    sq += r * r;
  }
  let l1 = 0;
  for (let j = 0; j < d; j++) l1 += Math.abs(thetaTilde[j]);
  return sq / (2 * n) + lambda * l1;
}

export interface CdStep { thetaTilde: number[]; z: number; isBias: boolean; }

/**
 * ONE coordinate update on the standardized design matrix (O(n·d)):
 * - feature j < d:  θ̃_j ← S(ρ_j/n, λ), ρ_j = Σᵢ zᵢⱼ·rᵢ^(−j) (partial residual, all
 *   other coordinates fixed). The returned z = ρ_j/n is the OLS coordinate solution
 *   used for narration ("soft-threshold visualized via narration").
 * - coordinate d (bias): θ̃_d ← (1/n)·Σᵢ residual — the exact minimizer along the
 *   bias direction, with NO penalty.
 * Each call updates exactly ONE coefficient — the timeline/step model: one step =
 * one coordinate update; a full sweep = d + 1 steps (d features + bias).
 */
export function cdStepStd(p: Params, data: LassoData, thetaTilde: number[], sc: StdScale, coordIndex: number): CdStep {
  const { xs, ys, d } = data;
  const n = xs.length;
  const lambda = (p.lambda as number) ?? 0.5;
  const next = thetaTilde.slice();
  if (coordIndex >= d) {
    // Exact minimizer along the bias direction (unpenalized): b̃ ← (1/n)·Σ(y − Σⱼθ̃ⱼzᵢⱼ).
    // The current bias must NOT enter the prediction — including it turns the update
    // into b̃ ← ȳ − b̃, an oscillation that corrupts the intercept (MLR-bias-bug class).
    let sum = 0;
    for (let i = 0; i < n; i++) {
      let pred = 0;
      for (let j = 0; j < d; j++) pred += thetaTilde[j] * ((xs[i][j] - sc.mu[j]) / sc.sigma[j]);
      sum += ys[i] - pred;
    }
    next[d] = sum / n;
    return { thetaTilde: next, z: NaN, isBias: true };
  }
  const j = coordIndex;
  let rho = 0;
  for (let i = 0; i < n; i++) {
    let pred = thetaTilde[d];
    for (let k = 0; k < d; k++) {
      if (k === j) continue;
      pred += thetaTilde[k] * ((xs[i][k] - sc.mu[k]) / sc.sigma[k]);
    }
    const z = (xs[i][j] - sc.mu[j]) / sc.sigma[j];
    rho += z * (ys[i] - pred);
  }
  const z = rho / n;
  next[j] = softThreshold(z, lambda);
  return { thetaTilde: next, z, isBias: false };
}

export function predict(theta: number[], row: number[]): number {
  let y = theta[row.length];
  for (let j = 0; j < row.length; j++) y += theta[j] * row[j];
  return y;
}

function mseOf(theta: number[], data: LassoData): number {
  return mean(data.xs.map((row, i) => (predict(theta, row) - data.ys[i]) ** 2));
}

function r2Of(theta: number[], data: LassoData): number {
  const ybar = mean(data.ys);
  const ssTot = data.ys.reduce((a, y) => a + (y - ybar) ** 2, 0);
  const ssRes = data.xs.reduce((a, row, i) => a + (predict(theta, row) - data.ys[i]) ** 2, 0);
  if (ssTot < 1e-12) return 1;
  return 1 - ssRes / ssTot;
}

function testPoint(d: number): number[] { return TEST_X.slice(0, d); }
function trueY(row: number[]): number {
  let y = TRUE_B;
  for (let j = 0; j < row.length; j++) y += TRUE_W[j] * row[j];
  return y;
}

function nNonzeroOf(theta: number[], d: number): number {
  let count = 0;
  for (let j = 0; j < d; j++) if (Math.abs(theta[j]) > 1e-12) count++;
  return count;
}

function metricsOf(theta: number[], data: LassoData, thetaTilde: number[], sc: StdScale, lambda: number, step: number, sweep: number): Record<string, number> {
  const m: Record<string, number> = {};
  for (let j = 0; j < data.d; j++) m[`w${j + 1}`] = theta[j];
  m.b = theta[data.d];
  m.mse = mseOf(theta, data);
  m.r2 = r2Of(theta, data);
  const tp = testPoint(data.d);
  m.testPred = predict(theta, tp);
  m.testTrue = trueY(tp);
  m.objective = objectiveStd(thetaTilde, data, sc, lambda);
  m.nNonzero = nNonzeroOf(theta, data.d);
  m.step = step;
  m.sweep = sweep;
  return m;
}

function thetaFromAlgorithm(a: Record<string, ParamValue>, d: number): number[] {
  const ws: number[] = [];
  for (let j = 0; j < d; j++) ws.push(a[`w${j + 1}`] as number);
  ws.push(a.b as number);
  return ws;
}

function algorithmOf(theta: number[], p: Params, step: number, sweep: number, objective: number, prevSweepObj: number, converged: boolean, d: number): Record<string, ParamValue> {
  const a: Record<string, ParamValue> = {
    mode: 'coordinate-descent',
    lambda: (p.lambda as number) ?? 0.5,
    step,
    sweep,
    objective,
    prevSweepObj,
  };
  for (let j = 0; j < d; j++) a[`w${j + 1}`] = theta[j];
  a.b = theta[d];
  if (converged) a.converged = true;
  return a;
}

/** Predicted-vs-actual diagnostic + residuals to the identity line (multi-feature visual). */
function buildVisuals(data: LassoData, theta: number[]): VisualCommand[] {
  const { xs, ys } = data;
  const preds = xs.map((row) => predict(theta, row));
  const lo = Math.min(...ys, ...preds);
  const hi = Math.max(...ys, ...preds);
  const pad = (hi - lo) * 0.1 + 0.5;
  const res = xs.map((_row, i) => ({ type: 'line', id: `res${i}`, points: [[preds[i], ys[i]], [preds[i], preds[i]]] as [number, number][], color: '#f59e0b' }));
  const pts = xs.map((_row, i) => ({ type: 'point', id: `d${i}`, x: preds[i], y: ys[i], color: '#64748b' }));
  const identity = { type: 'line', id: 'identity-line', points: [[lo - pad, lo - pad], [hi + pad, hi + pad]] as [number, number][], color: '#94a3b8' };
  return [...res, ...pts, identity];
}

/**
 * Matrix payload for the matrix-animator — emitted ONLY on the converged snapshot
 * (the "solution" snapshot, mirroring MLR's single-shot NE-mode matrices). Z and θ are
 * constant across the run except at convergence, so per-step matrix payloads would
 * bloat memory/scrub latency for zero pedagogical gain.
 */
function buildMatrices(data: LassoData, sc: StdScale, thetaTilde: number[]): VisualCommand[] {
  const Z = data.xs.map((row) => row.map((x, j) => (x - sc.mu[j]) / sc.sigma[j]));
  return [
    { type: 'matrix', id: 'Z (standardized)', rows: Z.length, cols: Z[0].length, cells: Z },
    { type: 'matrix', id: 'θ̃ (standardized)', rows: thetaTilde.length, cols: 1, cells: thetaTilde.map((v) => [v]) },
  ];
}

function fmtLambda(lambda: number): string {
  if (lambda > 0 && lambda < 0.01) return lambda.toExponential(1);
  return lambda.toFixed(2);
}

function coordLabel(coordIndex: number, d: number): string {
  return coordIndex >= d ? 'b (bias)' : `w${coordIndex + 1}`;
}

export const simulation = {
  initialState: (p: Params): SimState => {
    const data = generateData(p);
    const sc = standardize(data);
    const lambda = (p.lambda as number) ?? 0.5;
    const thetaTilde = new Array<number>(data.d + 1).fill(0); // zero start in standardized space
    const theta = fromStandard(thetaTilde, sc);
    const objective = objectiveStd(thetaTilde, data, sc, lambda);
    const m = metricsOf(theta, data, thetaTilde, sc, lambda, 0, 0);
    return {
      algorithm: algorithmOf(theta, p, 0, 0, objective, objective, false, data.d),
      visuals: buildVisuals(data, theta),
      math: [{ latex: `J(\\tilde{\\theta}) = \\frac{1}{2n}\\sum (y_i - \\hat{y}_i)^2 + \\lambda \\sum_j |\\tilde{w}_j| = ${objective.toFixed(3)}` }],
      narration: `λ = ${fmtLambda(lambda)} — coordinate descent start (all weights 0, b = 0). Features are z-scored; each step soft-thresholds ONE coordinate: θ̃ⱼ ← S(ρⱼ/n, λ). Objective J = ${objective.toFixed(3)}.`,
      explanation: {
        changed: [],
        why: 'Zero initialization: every coefficient starts at 0; coordinate descent then fills in the ones whose correlation exceeds λ',
        formulaRef: 'lasso-objective',
        dependsOn: ['convex-optimization', 'subgradient'],
        gateConcepts: ['lasso', 'coordinate descent', 'soft-threshold'],
      },
      highlights: [],
      metrics: m,
      events: [{ type: 'fit', label: 'cd-start', step: 0 }],
      timeline: ['Data', 'Fit', 'Evaluate'],
    };
  },

  step: (p: Params, s: SimState): SimState | null => {
    // Termination (both clean — null keeps failedAtStep undefined):
    // 1. convergence flag was set by the previous step;
    // 2. sweep budget reached (a full sweep just completed — sweep only increments
    //    at cycle boundaries, so this always lands on a whole-sweep boundary).
    if (s.algorithm.converged === true) return null;
    if (((s.algorithm.sweep as number) ?? 0) >= ((p.sweeps as number) ?? 300)) return null;
    const data = generateData(p);
    const sc = standardize(data);
    const d = data.d;
    const lambda = (p.lambda as number) ?? 0.5;
    const stepCount = (s.algorithm.step as number) ?? 0;
    const coordIndex = stepCount % (d + 1); // cycle w1…wd, then bias
    const theta = thetaFromAlgorithm(s.algorithm, d);
    const thetaTilde = toStandard(theta, sc);
    const res = cdStepStd(p, data, thetaTilde, sc, coordIndex);
    const nextStep = stepCount + 1;
    const sweep = Math.floor(nextStep / (d + 1));
    const thetaNext = fromStandard(res.thetaTilde, sc);
    const objective = objectiveStd(res.thetaTilde, data, sc, lambda);

    // Convergence: check at sweep boundaries (bias just updated). Monotone objective
    // decrease is guaranteed by coordinate descent; tiny float wobble is absorbed by
    // the relative tolerance.
    const prevSweepObj = (s.algorithm.prevSweepObj as number) ?? Number.POSITIVE_INFINITY;
    const converged =
      res.isBias &&
      sweep >= MIN_SWEEPS &&
      Math.abs(objective - prevSweepObj) <= CONV_TOL * Math.max(1, Math.abs(objective));
    const nextPrev = res.isBias ? objective : prevSweepObj;

    const m = metricsOf(thetaNext, data, res.thetaTilde, sc, lambda, nextStep, sweep);
    const label = coordLabel(coordIndex, d);
    let mathLatex: string;
    let narration: string;
    if (res.isBias) {
      mathLatex = `\\tilde{b} \\leftarrow \\frac{1}{n}\\sum_i r_i = ${res.thetaTilde[d].toFixed(3)}`;
      narration = `Step ${nextStep} — coordinate ${label}: intercept = mean residual = ${res.thetaTilde[d].toFixed(3)}. Objective = ${objective.toFixed(4)}, nNonzero = ${m.nNonzero}.`;
    } else {
      const newVal = res.thetaTilde[coordIndex];
      const j = coordIndex + 1;
      mathLatex = `\\tilde{w}_${j} \\leftarrow S(\\rho_${j}/n, \\lambda) = S(${res.z.toFixed(3)}, ${fmtLambda(lambda)}) = ${newVal.toFixed(3)}`;
      narration =
        Math.abs(res.z) <= lambda
          ? `Step ${nextStep} — coordinate ${label}: correlation z = ${res.z.toFixed(3)}. |z| = ${Math.abs(res.z).toFixed(3)} ≤ λ = ${fmtLambda(lambda)} → coefficient set to EXACTLY 0 (feature dropped). Objective = ${objective.toFixed(4)}, nNonzero = ${m.nNonzero}.`
          : `Step ${nextStep} — coordinate ${label}: correlation z = ρ/n = ${res.z.toFixed(3)}. Soft-threshold S(z, λ) = sign(z)·max(|z| − λ, 0) = ${newVal.toFixed(3)}. Objective = ${objective.toFixed(4)}, nNonzero = ${m.nNonzero}.`;
    }

    const snapshot: SimState = {
      algorithm: algorithmOf(thetaNext, p, nextStep, sweep, objective, nextPrev, converged, d),
      visuals: [
        ...buildVisuals(data, thetaNext),
        ...(converged ? buildMatrices(data, sc, res.thetaTilde) : []),
      ],
      math: [{ latex: mathLatex }],
      narration,
      explanation: {
        changed: [label, `objective = ${objective.toFixed(4)}`],
        why: `Coordinate ${label} updated via ${res.isBias ? 'unpenalized mean-residual update' : `soft-threshold S(ρ/n, λ) with λ = ${fmtLambda(lambda)}`}`,
        formulaRef: res.isBias ? 'lasso-objective' : 'cd-update',
        dependsOn: ['convex-optimization', 'subgradient'],
        gateConcepts: ['lasso', 'coordinate descent', 'soft-threshold', 'feature selection'],
      },
      highlights: [],
      metrics: m,
      events: [...s.events, ...(converged ? [{ type: 'converged' as const, label: 'converged' as const, step: nextStep }] : [])],
      timeline: converged ? [...s.timeline, 'Convergence'] : [...s.timeline, 'Fit'],
    };
    return snapshot;
  },
};

export const lassoModule: TopicModule = {
  id: 'lasso-regression',
  title: 'Lasso Regression (L1)',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 4, coding: 2, visualization: 2, gateFrequency: 5 },
    estimatedHours: 6,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['linear-algebra', 'calculus', 'multiple-linear-regression'],
    relatedTopics: ['ridge-regression', 'multiple-linear-regression', 'gradient-descent', 'polynomial-regression'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Geometry: Fit vs Data (Predicted vs Actual)' },
      { slot: 'primary', component: 'loss-curve', title: 'Objective over Coordinate Steps' },
      { slot: 'sidebar', component: 'matrix-animator', title: 'Standardized Design Matrix Z + Solution θ̃' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivation: Soft-Threshold Update' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Coordinate Sweeps' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'n', label: 'Number of samples', type: 'number', min: 5, max: 100, step: 1, default: 25 },
    { id: 'nFeatures', label: 'Number of features (d)', type: 'number', min: 3, max: 6, step: 1, default: 5 },
    { id: 'noise', label: 'Noise σ', type: 'number', min: 0, max: 3, step: 0.05, default: 0.5 },
    { id: 'lambda', label: 'Regularization λ', type: 'number', min: 0, max: 10, step: 0.25, default: 0.5 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
    { id: 'sweeps', label: 'Max coordinate sweeps', type: 'number', min: 10, max: 500, step: 10, default: 200 },
  ],
  simulation,
  formulas: lassoFormulas,
  derivations: lassoDerivations,
  questions: lassoQuestions,
  comparisons: lassoComparisons,
  failureDemos: lassoFailureDemos,
  mistakes: lassoMistakes,
  testCases: lassoTestCases,
  lossMetricKey: 'objective',

  validateParams: (p) => {
    const issues: string[] = [];
    const d = (p.nFeatures as number) ?? 5;
    const n = (p.n as number) ?? 25;
    if (n < d + 1) issues.push(`Underdetermined: n = ${n} < d + 1 = ${d + 1} — fewer samples than parameters makes the coefficient path meaningless`);
    const lambda = p.lambda as number | undefined;
    if (lambda !== undefined && lambda < 0) issues.push('λ (lambda) must be non-negative');
    const sweeps = p.sweeps as number | undefined;
    if (sweeps !== undefined && sweeps < 1) issues.push('Max coordinate sweeps must be at least 1');
    return issues;
  },
};

export function register() {
  registerTopic(lassoModule);
}
