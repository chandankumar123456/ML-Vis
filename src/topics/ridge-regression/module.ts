// src/topics/ridge-regression/module.ts
import type { TopicModule, Params, SimState, VisualCommand, ParamValue, MathStep } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { ridgeTestCases } from './testCases';
import { ridgeFormulas } from './formulas';
import { ridgeDerivations } from './derivations';
import { ridgeMistakes } from './mistakes';
import { ridgeQuestions } from './questions';
import { ridgeComparisons } from './comparisons';
import { ridgeFailureDemos } from './failures';
import { matMul, transpose, mean } from '../../lib/math/linAlg';

export interface RidgeData { xs: number[][]; ys: number[]; d: number; }

// Truth model used to synthesize data: y = wTrue·x + bTrue (shared with MLR, so the
// λ=0 ridge fit is bit-identical to the multiple-linear-regression normal equation).
export const TRUE_W = [3, -2, 1.5];
export const TRUE_B = 1;

// Fixed held-out test point (NOT drawn from the RNG) — an honest out-of-sample
// prediction reported in metrics.testPred / metrics.testTrue.
const TEST_X = [1.5, -0.5, 2.0];

// λ grid step — must match the UI slider step (0.5) so the run's LAST snapshot is
// always exactly the slider's λ.
const LAMBDA_STEP = 0.5;

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
 * Deterministic data synthesis (mirrors multiple-linear-regression so λ=0 ridge
 * equals its normal equation on the same params). Features are i.i.d. U[−5,5] →
 * XᵀX well-conditioned. `collinear` (test-only, not in the UI schema): x_j = j·x₁
 * for j ≥ 2 — EXACT linear dependence when `collinearJitter` = 0 (XᵀX exactly
 * singular → OLS fails cleanly), or near-collinear when jitter > 0 (XᵀX
 * invertible but ill-conditioned → OLS coefficients bounded-but-huge, the "OLS
 * explodes" demo).
 */
export function generateData(p: Params): RidgeData {
  const n = (p.n as number) ?? 25;
  const d = (p.nFeatures as number) ?? 2;
  const noise = (p.noise as number) ?? 0.5;
  const collinear = p.collinear === true;
  const jitter = (p.collinearJitter as number) ?? 0;
  const rng = mulberry32((p.seed as number) ?? 42);
  const xs: number[][] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < d; j++) {
      if (collinear && j > 0) row.push(row[0] * (j + 1) + (rng() - 0.5) * 2 * jitter);
      else row.push(-5 + rng() * 10);
    }
    let y = TRUE_B;
    for (let j = 0; j < d; j++) y += TRUE_W[j] * row[j];
    y += (rng() - 0.5) * 2 * noise;
    xs.push(row);
    ys.push(y);
  }
  return { xs, ys, d };
}

/** Design matrix with bias column LAST: X = [x₁ … x_d 1] (n × (d+1)). */
export function designMatrix(data: RidgeData): number[][] {
  return data.xs.map((row) => [...row, 1]);
}

/** Gauss–Jordan inversion with partial pivoting. Returns null when singular (rank-deficient). */
export function matInverse(A: number[][]): number[][] | null {
  const n = A.length;
  if (n === 0 || A.some((r) => r.length !== n)) return null;
  const aug = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(aug[r][col]) > Math.abs(aug[piv][col])) piv = r;
    if (Math.abs(aug[piv][col]) < 1e-12) return null; // singular
    [aug[col], aug[piv]] = [aug[piv], aug[col]];
    const d = aug[col][col];
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = aug[r][col];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) aug[r][j] -= f * aug[col][j];
    }
  }
  return aug.map((row) => row.slice(n));
}

/**
 * Closed form θ = (XᵀX + λI)⁻¹Xᵀy. The penalty applies to the FULL θ (bias
 * included) — the standard GATE closed-form treatment (compare the MLR formula
 * comparison: ridge objective is min ‖y − Xθ‖² + λ‖θ‖²). Adding λ to every
 * diagonal entry of XᵀX raises every eigenvalue by λ, so the Gram matrix is
 * invertible for λ > 0 even when XᵀX is singular (collinear / underdetermined).
 * Returns null when λ = 0 AND XᵀX is singular (exact collinearity) — the OLS
 * failure path, surfaced as a clean non-finite run.
 */
export function fitRidge(p: Params, data: RidgeData): number[] | null {
  const X = designMatrix(data);
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  const Xty = matMul(Xt, data.ys.map((y) => [y])).map((r) => r[0]);
  const lambda = (p.lambda as number) ?? 0;
  const G = XtX.map((row, r) => row.map((v, c) => v + (r === c ? lambda : 0)));
  const inv = matInverse(G);
  if (!inv) return null;
  return matMul(inv, Xty.map((v) => [v])).map((r) => r[0]);
}

export function predict(theta: number[], row: number[]): number {
  let y = theta[row.length];
  for (let j = 0; j < row.length; j++) y += theta[j] * row[j];
  return y;
}

function mseOn(theta: number[], data: RidgeData, idx: number[]): number {
  return mean(idx.map((i) => (predict(theta, data.xs[i]) - data.ys[i]) ** 2));
}

function r2Of(theta: number[], data: RidgeData): number {
  const ybar = mean(data.ys);
  const ssTot = data.ys.reduce((a, y) => a + (y - ybar) ** 2, 0);
  const ssRes = data.xs.reduce((a, row, i) => a + (predict(theta, row) - data.ys[i]) ** 2, 0);
  if (ssTot < 1e-12) return 1;
  return 1 - ssRes / ssTot;
}

function normOf(theta: number[]): number {
  return Math.sqrt(theta.reduce((a, v) => a + v * v, 0));
}

function testPoint(d: number): number[] { return TEST_X.slice(0, d); }
function trueY(row: number[]): number {
  let y = TRUE_B;
  for (let j = 0; j < row.length; j++) y += TRUE_W[j] * row[j];
  return y;
}

/**
 * Metrics: per-feature weights + bias, full-data mse/r², a deterministic held-out
 * split (first 70% train, rest test — pure index split, no RNG) for the
 * bias–variance story, and the L2 norms that track the shrinkage path.
 */
function metricsOf(theta: number[], data: RidgeData, lambda: number): Record<string, number> {
  const n = data.xs.length;
  const nTrain = Math.floor(n * 0.7);
  const tr = Array.from({ length: nTrain }, (_, i) => i);
  const te = Array.from({ length: n - nTrain }, (_, i) => i + nTrain);
  const m: Record<string, number> = {};
  for (let j = 0; j < data.d; j++) m[`w${j + 1}`] = theta[j];
  m.b = theta[data.d];
  m.lambda = lambda;
  m.mse = mseOn(theta, data, tr.concat(te));
  m.r2 = r2Of(theta, data);
  m.trainMse = mseOn(theta, data, tr);
  m.testMse = mseOn(theta, data, te);
  m.normTheta = normOf(theta);
  m.normW = normOf(theta.slice(0, data.d));
  const tp = testPoint(data.d);
  m.testPred = predict(theta, tp);
  m.testTrue = trueY(tp);
  return m;
}

function algorithmOf(theta: number[], lambda: number, d: number): Record<string, ParamValue> {
  const a: Record<string, ParamValue> = { lambda, mode: 'ridge-closed-form' };
  for (let j = 0; j < d; j++) a[`w${j + 1}`] = theta[j];
  a.b = theta[d];
  return a;
}

function buildVisuals(data: RidgeData, theta: number[]): VisualCommand[] {
  const { xs, ys, d } = data;
  if (d === 1) {
    // Geometry: scatter of (x₁, y) + fitted line (mirrors SLR / MLR d=1).
    const pts = xs.map((row, i) => ({ type: 'point', id: `d${i}`, x: row[0], y: ys[i], color: '#64748b' }));
    const line = { type: 'line', id: 'fit-line', points: [[-5.2, theta[0] * -5.2 + theta[1]], [5.2, theta[0] * 5.2 + theta[1]]] as [number, number][], color: '#3b82f6' };
    return [...pts, line];
  }
  // d ≥ 2: predicted-vs-actual diagnostic + residual lines to the identity line.
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
 * Matrix story for matrix-animator: X → XᵀX → XᵀX + λI → Xᵀy → θ. Scrub the λ
 * sweep and the DIAGONAL of the regularized Gram visibly grows by λ each step —
 * the "adding λ to the diagonal" animation. Matrices are tiny (≤ (d+1)² cells
 * each, ≤ 21 snapshots), so every snapshot carries the full story.
 */
function buildMatrices(data: RidgeData, theta: number[], lambda: number): VisualCommand[] {
  const X = designMatrix(data);
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  const Xty = matMul(Xt, data.ys.map((y) => [y]));
  const G = XtX.map((row, r) => row.map((v, c) => v + (r === c ? lambda : 0)));
  return [
    { type: 'matrix', id: 'X', rows: X.length, cols: X[0].length, cells: X },
    { type: 'matrix', id: 'XᵀX', rows: XtX.length, cols: XtX[0].length, cells: XtX },
    { type: 'matrix', id: 'XᵀX + λI', rows: G.length, cols: G[0].length, cells: G },
    { type: 'matrix', id: 'Xᵀy', rows: Xty.length, cols: 1, cells: Xty },
    { type: 'matrix', id: 'θ', rows: theta.length, cols: 1, cells: theta.map((v) => [v]) },
  ];
}

function wStr(theta: number[], d: number): string {
  return Array.from({ length: d }, (_, j) => `w${j + 1} = ${theta[j].toFixed(3)}`).join(', ');
}

function snapshotAt(p: Params, data: RidgeData, lambda: number, first: boolean): SimState {
  const theta = fitRidge({ ...p, lambda }, data);
  // λ=0 with singular XᵀX → null → NaN fill → sandbox reports a clean failure.
  const th = theta ?? Array.from({ length: data.d + 1 }, () => NaN);
  const m = metricsOf(th, data, lambda);
  const norm = m.normTheta;
  const lamStr = lambda === 0 ? 'λ = 0 (OLS baseline)' : `λ = ${lambda}`;
  const math: MathStep[] = [
    { latex: 'J(\\theta) = \\frac{1}{n}\\|y - X\\theta\\|_2^2 + \\lambda\\|\\theta\\|_2^2', id: 'ridge-objective' },
    { latex: '\\theta = (X^T X + \\lambda I)^{-1} X^T y', id: 'ridge-closed-form' },
  ];
  if (lambda === 0) {
    math.push({ latex: '\\lambda = 0 \\Rightarrow \\theta = (X^T X)^{-1} X^T y', caption: 'ridge at λ=0 is exactly OLS' });
  } else {
    math.push({ latex: `X^T X + ${lambda} I`, caption: 'λ added to every diagonal entry' });
  }
  return {
    algorithm: algorithmOf(th, lambda, data.d),
    visuals: [
      ...buildVisuals(data, th),
      ...buildMatrices(data, th, lambda),
    ],
    math,
    narration: `${lamStr}: ${wStr(th, data.d)}, b = ${th[data.d].toFixed(3)} — ‖θ‖ = ${norm.toFixed(4)}, train MSE = ${m.trainMse.toFixed(4)}, test MSE = ${m.testMse.toFixed(4)}`,
    explanation: {
      changed: first ? [] : [`λ → ${lambda}`, `‖θ‖ = ${norm.toFixed(4)}`],
      why: first
        ? 'Closed-form ridge solution at λ = 0 — the penalty term vanishes, so this is the OLS normal equation'
        : `Ridge closed form with λ = ${lambda}: +λI on the diagonal of XᵀX shrinks every coefficient`,
      formulaRef: 'ridge-closed-form',
      dependsOn: ['linear-algebra', 'matrix-inverse'],
      gateConcepts: ['ridge', lambda === 0 ? 'OLS' : 'shrinkage'],
    },
    highlights: [],
    metrics: m,
    events: [{ type: 'fit', label: 'ridge-solve', step: 0 }],
    timeline: first ? ['Data', 'Fit', 'Evaluate'] : ['Fit', 'Evaluate'],
  };
}

export const simulation = {
  /**
   * One snapshot per λ on the grid [0, 0.5, …, params.lambda]. The first
   * snapshot is the λ = 0 OLS reference; each step advances λ by 0.5 up to the
   * slider's value. Scrubbing the run IS the shrinkage path; the loss-curve
   * plots train/test error vs λ. Single-shot per λ (closed form — no epochs).
   */
  initialState: (p: Params): SimState => {
    const data = generateData(p);
    return snapshotAt(p, data, 0, true);
  },

  step: (p: Params, s: SimState): SimState | null => {
    const data = generateData(p);
    const target = (p.lambda as number) ?? 1;
    const current = (s.algorithm.lambda as number) ?? 0;
    const next = current + LAMBDA_STEP;
    if (next > target + 1e-9) return null; // sweep complete
    return snapshotAt(p, data, next, false);
  },
};

export const ridgeModule: TopicModule = {
  id: 'ridge-regression',
  title: 'Ridge Regression',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 4, coding: 2, visualization: 3, gateFrequency: 4 },
    estimatedHours: 5,
    revisionPriority: 'P1',
    examFrequency: 'Frequent',
    prerequisites: ['linear-algebra', 'calculus', 'multiple-linear-regression'],
    relatedTopics: ['multiple-linear-regression', 'lasso-regression', 'polynomial-regression', 'bias-variance'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Geometry: Fit vs Data (1 feature) / Predicted vs Actual' },
      { slot: 'primary', component: 'loss-curve', title: 'Test MSE vs λ (bias–variance trace)' },
      { slot: 'sidebar', component: 'matrix-animator', title: 'Design Matrices: X, XᵀX, XᵀX + λI, Xᵀy, θ' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivation: Ridge Closed Form' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Fit Stages' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'lambda', label: 'Ridge penalty λ', type: 'number', min: 0, max: 10, step: 0.5, default: 1 },
    { id: 'n', label: 'Number of samples', type: 'number', min: 5, max: 100, step: 1, default: 25 },
    { id: 'nFeatures', label: 'Number of features (d)', type: 'number', min: 1, max: 3, step: 1, default: 2 },
    { id: 'noise', label: 'Noise σ', type: 'number', min: 0, max: 3, step: 0.05, default: 0.5 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
  ],
  simulation,
  formulas: ridgeFormulas,
  derivations: ridgeDerivations,
  questions: ridgeQuestions,
  comparisons: ridgeComparisons,
  failureDemos: ridgeFailureDemos,
  mistakes: ridgeMistakes,
  testCases: ridgeTestCases,
  lossMetricKey: 'testMse',

  validateParams: (p) => {
    const issues: string[] = [];
    const d = (p.nFeatures as number) ?? 2;
    const n = (p.n as number) ?? 25;
    const lambda = (p.lambda as number) ?? 1;
    if (!Number.isFinite(lambda)) issues.push('λ must be a finite number');
    if (lambda < 0) issues.push('λ must be non-negative (negative λ is anti-ridge / not a shrinkage penalty)');
    if (n < d + 1 && lambda <= 0) {
      issues.push(`Underdetermined: n = ${n} < d + 1 = ${d + 1} — XᵀX is singular at λ = 0 (set λ > 0: ridge restores invertibility)`);
    }
    if (lambda <= 0 && collinearFeatures(p)) {
      issues.push('Features are collinear — at λ = 0, XᵀX is singular and OLS coefficients explode; set λ > 0 (ridge) to stabilize');
    }
    return issues;
  },
};

/** Data-driven collinearity probe: pairwise |correlation| ≈ 1 or a zero-variance column. */
function collinearFeatures(p: Params): boolean {
  const data = generateData(p);
  const { xs, d } = data;
  for (let j = 0; j < d; j++) {
    const col = xs.map((r) => r[j]);
    const m = mean(col);
    const v = mean(col.map((x) => (x - m) ** 2));
    if (v < 1e-12) return true;
    for (let k = j + 1; k < d; k++) {
      const col2 = xs.map((r) => r[k]);
      const m2 = mean(col2);
      const v2 = mean(col2.map((x) => (x - m2) ** 2));
      if (v2 < 1e-12) return true;
      const cov = mean(col.map((x, i) => (x - m) * (col2[i] - m2)));
      if (Math.abs(cov / Math.sqrt(v * v2)) > 0.9999) return true;
    }
  }
  return false;
}

export function register() {
  registerTopic(ridgeModule);
}
