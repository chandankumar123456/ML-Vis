// src/topics/multiple-linear-regression/module.ts
import type { TopicModule, Params, SimState, VisualCommand, ParamValue } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { mlrTestCases } from './testCases';
import { mlrFormulas } from './formulas';
import { mlrDerivations } from './derivations';
import { mlrMistakes } from './mistakes';
import { mlrQuestions } from './questions';
import { mlrComparisons } from './comparisons';
import { mlrFailureDemos } from './failures';
import { matMul, transpose, mean } from '../../lib/math/linAlg';

export interface MlrData { xs: number[][]; ys: number[]; d: number; }

// Truth model used to synthesize data: y = wTrue·x + bTrue (deterministic per feature count).
export const TRUE_W = [3, -2, 1.5];
export const TRUE_B = 1;

// Fixed held-out test point (NOT drawn from the RNG) — lets every run report an honest
// out-of-sample prediction in metrics.testPred / metrics.testTrue.
const TEST_X = [1.5, -0.5, 2.0];

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
 * U[−5,5] (seeded), so columns are near-orthogonal and XᵀX is well-conditioned —
 * critical for both the normal equation (stable inversion) and GD (fast, uniform
 * convergence). A shared stratification base would make x₁ ≈ x₂ (corr ≈ 0.999) and
 * turn XᵀX nearly singular — a genuine data-generation trap (Wave 1 QA).
 * `p.collinear` is a test-only switch (not exposed in the UI param schema): when true,
 * x_j = j·x₁ for j ≥ 2 → perfect multicollinearity → XᵀX singular.
 */
export function generateData(p: Params): MlrData {
  const n = (p.n as number) ?? 25;
  const d = (p.nFeatures as number) ?? 2;
  const noise = (p.noise as number) ?? 0.5;
  const collinear = p.collinear === true;
  const rng = mulberry32((p.seed as number) ?? 42);
  const xs: number[][] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < d; j++) {
      if (collinear && j > 0) row.push(row[0] * (j + 1)); // x₂ = 2x₁, x₃ = 3x₁
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

/** Design matrix with bias column LAST: X = [x₁ … x_d 1] (n × (d+1)). */
export function designMatrix(data: MlrData): number[][] {
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

/** Closed-form θ = (XᵀX)⁻¹Xᵀy. Returns null when XᵀX is singular (collinear / underdetermined). */
export function fitNormalEquation(_p: Params, data: MlrData): number[] | null {
  const X = designMatrix(data);
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  const Xty = matMul(Xt, data.ys.map((y) => [y])).map((r) => r[0]);
  const inv = matInverse(XtX);
  if (!inv) return null;
  return matMul(inv, Xty.map((v) => [v])).map((r) => r[0]);
}

export interface StdScale { mu: number[]; sigma: number[]; }

/**
 * Per-feature z-score statistics. GD runs in standardized space so convergence is
 * fast and lr is insensitive to feature scale; θ is converted back to original
 * coordinates for metrics/display. The bias column (all 1s) is NOT standardized.
 */
export function standardize(data: MlrData): StdScale {
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
  // Using w̃ⱼ·μⱼ here breaks the round trip and leaks bias per epoch — Wave-1 QA bug.
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

/** ONE full-batch gradient descent epoch on the standardized design matrix: O(n·d). */
export function gdEpochStd(p: Params, data: MlrData, thetaTilde: number[], sc: StdScale): number[] {
  const { xs, ys, d } = data;
  const n = xs.length;
  const lr = (p.learningRate as number) ?? 0.01;
  const grad = new Array<number>(d + 1).fill(0);
  for (let i = 0; i < n; i++) {
    let pred = thetaTilde[d];
    for (let j = 0; j < d; j++) pred += thetaTilde[j] * ((xs[i][j] - sc.mu[j]) / sc.sigma[j]);
    const err = pred - ys[i];
    for (let j = 0; j < d; j++) grad[j] += ((2 / n) * err * (xs[i][j] - sc.mu[j])) / sc.sigma[j];
    grad[d] += (2 / n) * err;
  }
  return thetaTilde.map((v, j) => v - lr * grad[j]);
}

export function predict(theta: number[], row: number[]): number {
  let y = theta[row.length];
  for (let j = 0; j < row.length; j++) y += theta[j] * row[j];
  return y;
}

function mseOf(theta: number[], data: MlrData): number {
  return mean(data.xs.map((row, i) => (predict(theta, row) - data.ys[i]) ** 2));
}

function r2Of(theta: number[], data: MlrData): number {
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

function metricsOf(theta: number[], data: MlrData): Record<string, number> {
  const m: Record<string, number> = {};
  for (let j = 0; j < data.d; j++) m[`w${j + 1}`] = theta[j];
  m.b = theta[data.d];
  m.mse = mseOf(theta, data);
  m.r2 = r2Of(theta, data);
  const tp = testPoint(data.d);
  m.testPred = predict(theta, tp);
  m.testTrue = trueY(tp);
  return m;
}

function algorithmOf(theta: number[], mode: 'normal-equation' | 'gradient-descent', epoch: number, d: number): Record<string, ParamValue> {
  const a: Record<string, ParamValue> = { mode, epoch };
  for (let j = 0; j < d; j++) a[`w${j + 1}`] = theta[j];
  a.b = theta[d];
  return a;
}

function thetaFromAlgorithm(a: Record<string, ParamValue>, d: number): number[] {
  const ws: number[] = [];
  for (let j = 0; j < d; j++) ws.push(a[`w${j + 1}`] as number);
  ws.push(a.b as number);
  return ws;
}

function buildVisuals(data: MlrData, theta: number[]): VisualCommand[] {
  const { xs, ys, d } = data;
  if (d === 1) {
    // Geometry: scatter of (x₁, y) + fitted line (mirrors SLR).
    const pts = xs.map((row, i) => ({ type: 'point', id: `d${i}`, x: row[0], y: ys[i], color: '#64748b' }));
    const line = { type: 'line', id: 'fit-line', points: [[-5.2, theta[0] * -5.2 + theta[1]], [5.2, theta[0] * 5.2 + theta[1]]] as [number, number][], color: '#3b82f6' };
    return [...pts, line];
  }
  // d ≥ 2: predicted-vs-actual diagnostic + residual lines to the identity line.
  // Points hugging ŷ = y mean a good fit — the honest multi-feature visual.
  const preds = xs.map((row) => predict(theta, row));
  const lo = Math.min(...ys, ...preds);
  const hi = Math.max(...ys, ...preds);
  const pad = (hi - lo) * 0.1 + 0.5;
  const res = xs.map((_row, i) => ({ type: 'line', id: `res${i}`, points: [[preds[i], ys[i]], [preds[i], preds[i]]] as [number, number][], color: '#f59e0b' }));
  const pts = xs.map((_row, i) => ({ type: 'point', id: `d${i}`, x: preds[i], y: ys[i], color: '#64748b' }));
  const identity = { type: 'line', id: 'identity-line', points: [[lo - pad, lo - pad], [hi + pad, hi + pad]] as [number, number][], color: '#94a3b8' };
  return [...res, ...pts, identity];
}

// Design-matrix story for matrix-animator: X → XᵀX → Xᵀy → θ, with dimension
// compatibility visible (n×(d+1) · (d+1)×n → (d+1)×(d+1), etc.). Emitted in
// normal-equation mode only — GD mode has thousands of snapshots and matrix payloads
// per step would blow up memory/scrub latency (see plan drift notes).
function buildMatrices(data: MlrData, theta: number[]): VisualCommand[] {
  const X = designMatrix(data);
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  const Xty = matMul(Xt, data.ys.map((y) => [y]));
  return [
    { type: 'matrix', id: 'X', rows: X.length, cols: X[0].length, cells: X },
    { type: 'matrix', id: 'XᵀX', rows: XtX.length, cols: XtX[0].length, cells: XtX },
    { type: 'matrix', id: 'Xᵀy', rows: Xty.length, cols: 1, cells: Xty },
    { type: 'matrix', id: 'θ', rows: theta.length, cols: 1, cells: theta.map((v) => [v]) },
  ];
}

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

export const simulation = {
  initialState: (p: Params): SimState => {
    const data = generateData(p);
    const useNormal = p.useNormalEquation === true;
    let theta: number[];
    let epoch = 0;
    if (useNormal) {
      // Closed form θ = (XᵀX)⁻¹Xᵀy; singular XᵀX → NaN → sandbox reports a clean failure.
      theta = fitNormalEquation(p, data) ?? Array.from({ length: data.d + 1 }, () => NaN);
    } else {
      const sc = standardize(data);
      const z = new Array<number>(data.d + 1).fill(0); // zero init in standardized space
      theta = fromStandard(gdEpochStd(p, data, z, sc), sc); // one epoch
      epoch = 1;
    }
    const m = metricsOf(theta, data);
    const mode = useNormal ? 'normal-equation' : 'gradient-descent';
    const wStr = Array.from({ length: data.d }, (_, j) => `w${j + 1} = ${theta[j].toFixed(3)}`).join(', ');
    return {
      algorithm: algorithmOf(theta, mode, epoch, data.d),
      visuals: [
        ...buildVisuals(data, theta),
        ...(useNormal ? buildMatrices(data, theta) : []),
      ],
      math: [{ latex: `\\theta = (X^T X)^{-1} X^T y`, id: 'normal-equation' }],
      narration: useNormal
        ? `Normal equation solved directly: ${wStr}, b = ${theta[data.d].toFixed(3)} — MSE = ${m.mse.toFixed(4)}, R² = ${m.r2.toFixed(3)}`
        : `Gradient descent epoch 1 (zero init, standardized features): ${wStr}, b = ${theta[data.d].toFixed(3)} — MSE = ${m.mse.toFixed(4)}`,
      explanation: {
        changed: [],
        why: useNormal ? 'Closed-form solution from XᵀX and Xᵀy' : 'One epoch of gradient descent on MSE (features z-scored for fast convergence)',
        formulaRef: useNormal ? 'normal-equation' : 'mse',
        dependsOn: ['linear-algebra', 'projection'],
        gateConcepts: ['OLS', 'normal equation', 'least squares'],
      },
      highlights: [],
      metrics: m,
      events: [{ type: 'fit', label: useNormal ? 'normal-equation' : 'gd-epoch', step: 0 }],
      timeline: useNormal ? ['Data', 'Design Matrix', 'Solve', 'Evaluate'] : ['Data', 'Fit', 'Evaluate'],
    };
  },

  step: (p: Params, s: SimState): SimState | null => {
    if (s.algorithm.mode !== 'gradient-descent') return null; // normal equation is single-shot
    const data = generateData(p);
    const epochs = (p.epochs as number) ?? 2000;
    const currentEpoch = ((s.algorithm.epoch as number) ?? 1) + 1;
    if (currentEpoch > epochs) return null;
    const sc = standardize(data);
    const theta = fromStandard(
      gdEpochStd(p, data, toStandard(thetaFromAlgorithm(s.algorithm, data.d), sc), sc),
      sc,
    );
    const m = metricsOf(theta, data);
    const wStr = Array.from({ length: data.d }, (_, j) => `w${j + 1} = ${theta[j].toFixed(3)}`).join(', ');
    return {
      algorithm: algorithmOf(theta, 'gradient-descent', currentEpoch, data.d),
      visuals: buildVisuals(data, theta),
      math: [{ latex: `\\theta \\leftarrow \\theta - \\eta \\cdot \\frac{2}{n} X^T (X\\theta - y)` }],
      narration: `Epoch ${currentEpoch}: ${wStr}, b = ${theta[data.d].toFixed(3)} — MSE = ${m.mse.toFixed(4)}, R² = ${m.r2.toFixed(3)}`,
      explanation: {
        changed: [`epoch → ${currentEpoch}`, `MSE = ${m.mse.toFixed(4)}`],
        why: `Gradient descent step with η = ${p.learningRate} (standardized features)`,
        formulaRef: 'mse',
        dependsOn: ['gradient-descent'],
        gateConcepts: ['GD', 'MSE'],
      },
      highlights: [],
      metrics: m,
      events: [...s.events],
      timeline: ['Fit', 'Evaluate'],
    };
  },
};

export const mlrModule: TopicModule = {
  id: 'multiple-linear-regression',
  title: 'Multiple Linear Regression',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 2, mathematical: 4, coding: 2, visualization: 2, gateFrequency: 5 },
    estimatedHours: 6,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['linear-algebra', 'calculus', 'simple-linear-regression'],
    relatedTopics: ['simple-linear-regression', 'ridge-regression', 'polynomial-regression', 'gradient-descent'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Geometry: Fit vs Data (1 feature) / Predicted vs Actual' },
      { slot: 'primary', component: 'loss-curve', title: 'MSE over Epochs (GD mode)' },
      { slot: 'sidebar', component: 'matrix-animator', title: 'Design Matrices: X, XᵀX, Xᵀy, θ' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivation: Normal Equation' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Fit Stages' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'n', label: 'Number of samples', type: 'number', min: 5, max: 100, step: 1, default: 25 },
    { id: 'nFeatures', label: 'Number of features (d)', type: 'number', min: 1, max: 3, step: 1, default: 2 },
    { id: 'noise', label: 'Noise σ', type: 'number', min: 0, max: 3, step: 0.05, default: 0.5 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
    { id: 'useNormalEquation', label: 'Use normal equation', type: 'toggle', default: true },
    { id: 'learningRate', label: 'Learning rate η', type: 'number', min: 0.001, max: 0.1, step: 0.001, default: 0.01 },
    { id: 'epochs', label: 'Epochs (GD)', type: 'number', min: 50, max: 5000, step: 50, default: 500 },
  ],
  simulation,
  formulas: mlrFormulas,
  derivations: mlrDerivations,
  questions: mlrQuestions,
  comparisons: mlrComparisons,
  failureDemos: mlrFailureDemos,
  mistakes: mlrMistakes,
  testCases: mlrTestCases,
  lossMetricKey: 'mse',

  validateParams: (p) => {
    const issues: string[] = [];
    const d = (p.nFeatures as number) ?? 2;
    const n = (p.n as number) ?? 25;
    if (n < d + 1) issues.push(`Underdetermined: n = ${n} < d + 1 = ${d + 1} — XᵀX is rank-deficient (infinite solutions)`);
    const lr = p.learningRate as number | undefined;
    if (lr !== undefined && lr <= 0) issues.push('Learning rate must be positive');
    if (collinearFeatures(p)) issues.push('Features are collinear — XᵀX is singular (no unique least-squares solution)');
    return issues;
  },
};

export function register() {
  registerTopic(mlrModule);
}
