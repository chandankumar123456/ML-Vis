// src/topics/polynomial-regression/module.ts
import type { TopicModule, Params, SimState, VisualCommand, ParamValue } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { polyTestCases } from './testCases';
import { polyFormulas } from './formulas';
import { polyDerivations } from './derivations';
import { polyMistakes } from './mistakes';
import { polyQuestions } from './questions';
import { polyComparisons } from './comparisons';
import { polyFailureDemos } from './failures';
import { matMul, transpose, mean } from '../../lib/math/linAlg';

/**
 * Polynomial regression: linear regression on the basis-expanded feature
 * φ(u) = [1, u, u², …, u^d]. The input x ∈ [−XMAX, XMAX] is normalized to
 * u = x/XMAX ∈ [−1, 1] FIRST — this is the standard conditioning fix for
 * power bases: without it, Φ entries grow like x^d (3¹⁵ ≈ 1.4×10⁷) and the
 * Vandermonde Gram matrix becomes numerically singular well below degree 30.
 * The model class is unchanged (degree-d in u ⇔ degree-d in x); only the
 * coefficient basis is rescaled. Coefficients are reported in this u-basis.
 */
export const XMAX = 3;

/** Truth model used to synthesize data: y = w₂·u² + w₁·u + w₀ (u = x/XMAX). */
export const TRUTH = { w0: 1, w1: 0.5, w2: 1 };

export type FitOn = 'train' | 'all';

export interface PolyData {
  xs: number[];   // raw input x ∈ [−XMAX, XMAX] (for scatter/curve display)
  us: number[];   // normalized u = x/XMAX
  ys: number[];
  nTrain: number; // first nTrain samples are the training split; the rest is held-out test
}

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
 * Deterministic data synthesis. Inputs u are i.i.d. U[−1,1] (seeded); targets
 * come from the quadratic truth y = w₂u² + w₁u + w₀ plus uniform noise.
 * A single seeded stream of nTrain + nTest points is drawn — the first nTrain
 * are the training split, the rest the held-out test split (same distribution).
 */
export function generateData(p: Params): PolyData {
  const nTrain = (p.nTrain as number) ?? 30;
  const nTest = (p.nTest as number) ?? 20;
  const noise = (p.noise as number) ?? 0.5;
  const rng = mulberry32((p.seed as number) ?? 42);
  const xs: number[] = [];
  const us: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < nTrain + nTest; i++) {
    const u = -1 + rng() * 2;
    const x = u * XMAX;
    const y = TRUTH.w0 + TRUTH.w1 * u + TRUTH.w2 * u * u + (rng() - 0.5) * 2 * noise;
    xs.push(x);
    us.push(u);
    ys.push(y);
  }
  return { xs, us, ys, nTrain };
}

/**
 * Vandermonde design matrix Φ (n × (d+1)) with the bias column FIRST,
 * following the polynomial convention φ(u) = [1, u, u², …, u^d]:
 * row i = [1, uᵢ, uᵢ², …, uᵢ^d]. (MLR/SLR put the bias column LAST; the
 * polynomial basis is conventionally written with the constant first, so
 * w₀ is the intercept here.)
 */
export function vandermonde(us: number[], degree: number): number[][] {
  return us.map((u) => {
    const row = [1];
    let p = 1;
    for (let j = 1; j <= degree; j++) {
      p *= u;
      row.push(p);
    }
    return row;
  });
}

/**
 * Gauss–Jordan inversion with partial pivoting. Returns null when rank-deficient.
 * Pivot threshold is 1e-14 (MLR's canonical matInverse uses 1e-12): the
 * degree-15 Vandermonde Gram has condition number ~10¹² and intermediate
 * pivots dip below 1e-12 for unlucky data draws — the tighter threshold keeps
 * the full degree-15 slider range solvable (verified 50/50 across seeds at
 * n=30, d=15) while degree 30 still fails cleanly (pivots < 1e-14, the
 * numerical-instability failure story).
 */
export function matInverse(A: number[][]): number[][] | null {
  const n = A.length;
  if (n === 0 || A.some((r) => r.length !== n)) return null;
  const aug = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(aug[r][col]) > Math.abs(aug[piv][col])) piv = r;
    if (Math.abs(aug[piv][col]) < 1e-14) return null; // numerically singular
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

export interface PolyFit { theta: number[]; cond: number; }

/** 1-norm condition number estimate κ = ‖G‖₁·‖G⁻¹‖₁ from the already-computed inverse. */
function condOf(G: number[][], inv: number[][]): number {
  const n1 = (M: number[][]) => Math.max(...M.map((r) => r.reduce((a, b) => a + Math.abs(b), 0)));
  return n1(G) * n1(inv);
}

/**
 * Closed-form θ = (ΦᵀΦ)⁻¹Φᵀy on the u-basis Vandermonde of the fit subset
 * (train split, or all data when fitOn = 'all'). Returns null when ΦᵀΦ is
 * numerically singular (degree + 1 > nTrain, or the degenerate high-degree
 * regime) — the caller turns that into a clean non-finite failure.
 */
export function fitPoly(_p: Params, data: PolyData, degree: number, fitOn: FitOn): PolyFit | null {
  const subsetUs = fitOn === 'train' ? data.us.slice(0, data.nTrain) : data.us;
  const subsetYs = fitOn === 'train' ? data.ys.slice(0, data.nTrain) : data.ys;
  const Phi = vandermonde(subsetUs, degree);
  const Pht = transpose(Phi);
  const G = matMul(Pht, Phi);
  const gy = matMul(Pht, subsetYs.map((y) => [y])).map((r) => r[0]);
  const inv = matInverse(G);
  if (!inv) return null;
  const theta = matMul(inv, gy.map((v) => [v])).map((r) => r[0]);
  return { theta, cond: condOf(G, inv) };
}

export function predictPoly(theta: number[], u: number, degree: number): number {
  let y = theta[0];
  let p = 1;
  for (let j = 1; j <= degree; j++) {
    p *= u;
    y += theta[j] * p;
  }
  return y;
}

/** Slice the dataset for a given subset: 'train' | 'test' | 'all'. */
function subsetOf(data: PolyData, subset: FitOn | 'test'): { us: number[]; ys: number[] } {
  if (subset === 'train') return { us: data.us.slice(0, data.nTrain), ys: data.ys.slice(0, data.nTrain) };
  if (subset === 'test') return { us: data.us.slice(data.nTrain), ys: data.ys.slice(data.nTrain) };
  return { us: data.us, ys: data.ys }; // 'all'
}

export function mseOf(theta: number[], data: PolyData, degree: number, subset: FitOn | 'test'): number {
  const { us, ys } = subsetOf(data, subset);
  return mean(us.map((u, i) => (predictPoly(theta, u, degree) - ys[i]) ** 2));
}

function r2Of(theta: number[], data: PolyData, degree: number, subset: FitOn | 'test'): number {
  const { us, ys } = subsetOf(data, subset);
  const ybar = mean(ys);
  const ssTot = ys.reduce((a, y) => a + (y - ybar) ** 2, 0);
  const ssRes = us.reduce((a, u, i) => a + (predictPoly(theta, u, degree) - ys[i]) ** 2, 0);
  if (ssTot < 1e-12) return 1;
  return 1 - ssRes / ssTot;
}

function metricsOf(theta: number[], degree: number, data: PolyData, fitOn: FitOn, cond: number): Record<string, number> {
  const m: Record<string, number> = {};
  for (let j = 0; j <= degree; j++) m[`w${j}`] = theta[j];
  const fitSubset: FitOn = fitOn === 'train' ? 'train' : 'all';
  // fitOn='all': the whole dataset is the training set, so test error is in-sample.
  m.trainMse = mseOf(theta, data, degree, fitSubset);
  m.testMse = mseOf(theta, data, degree, fitOn === 'train' ? 'test' : 'all');
  m.r2 = r2Of(theta, data, degree, fitSubset);
  m.maxAbsW = Math.max(...theta.slice(1).map(Math.abs));
  m.cond = cond;
  m.nParams = degree + 1;
  return m;
}

function algorithmOf(theta: number[], degree: number, fitOn: FitOn): Record<string, ParamValue> {
  const a: Record<string, ParamValue> = { mode: 'normal-equation', degree, fitOn };
  for (let j = 0; j <= degree; j++) a[`w${j}`] = theta[j];
  return a;
}

/** Train scatter + residuals, test scatter, and the fitted polynomial curve. */
function buildVisuals(data: PolyData, theta: number[], degree: number): VisualCommand[] {
  const { xs, us, ys, nTrain } = data;
  const res: VisualCommand[] = [];
  const trainPts: VisualCommand[] = [];
  for (let i = 0; i < nTrain; i++) {
    const yHat = predictPoly(theta, us[i], degree);
    res.push({ type: 'line', id: `res${i}`, points: [[xs[i], ys[i]], [xs[i], yHat]] as [number, number][], color: '#cbd5e1' });
    trainPts.push({ type: 'point', id: `tr${i}`, x: xs[i], y: ys[i], color: '#2563eb' });
  }
  const testPts: VisualCommand[] = [];
  for (let i = nTrain; i < us.length; i++) {
    testPts.push({ type: 'point', id: `te${i}`, x: xs[i], y: ys[i], color: '#f97316' });
  }
  const curvePts: [number, number][] = [];
  const SAMPLES = 121;
  for (let i = 0; i <= SAMPLES; i++) {
    const x = -XMAX + (2 * XMAX * i) / SAMPLES;
    curvePts.push([x, predictPoly(theta, x / XMAX, degree)]);
  }
  return [
    ...res,
    ...trainPts,
    ...testPts,
    { type: 'line', id: 'fit-curve', points: curvePts, color: '#3b82f6' },
  ];
}

/**
 * Vandermonde matrix story for matrix-animator: Φ → ΦᵀΦ → Φᵀy → θ.
 * The solve is single-shot (normal equation only — no GD mode), so the full
 * matrix story is always cheap to emit (mirrors the Task 1 drift note).
 */
function buildMatrices(data: PolyData, theta: number[], degree: number, fitOn: FitOn): VisualCommand[] {
  const subsetUs = fitOn === 'train' ? data.us.slice(0, data.nTrain) : data.us;
  const subsetYs = fitOn === 'train' ? data.ys.slice(0, data.nTrain) : data.ys;
  const Phi = vandermonde(subsetUs, degree);
  const Pht = transpose(Phi);
  const G = matMul(Pht, Phi);
  const gy = matMul(Pht, subsetYs.map((y) => [y]));
  return [
    { type: 'matrix', id: 'Φ', rows: Phi.length, cols: Phi[0].length, cells: Phi },
    { type: 'matrix', id: 'ΦᵀΦ', rows: G.length, cols: G[0].length, cells: G },
    { type: 'matrix', id: 'Φᵀy', rows: gy.length, cols: 1, cells: gy },
    { type: 'matrix', id: 'θ', rows: theta.length, cols: 1, cells: theta.map((v) => [v]) },
  ];
}

function fitComment(degree: number, trainMse: number, testMse: number): string {
  if (trainMse > 0 && testMse > 5 * trainMse) {
    return 'The test error is far above the training error — the curve has memorized noise (variance / overfitting).';
  }
  if (degree === 1) {
    return 'A line cannot capture the quadratic curvature — the fit is biased (underfitting).';
  }
  return 'Training and test error are comparable — the model generalizes.';
}

export const simulation = {
  initialState: (p: Params): SimState => {
    const data = generateData(p);
    const degree = (p.degree as number) ?? 5;
    const fitOn: FitOn = (p.fitOn as string) === 'all' ? 'all' : 'train';
    const fit = fitPoly(p, data, degree, fitOn);
    // Singular ΦᵀΦ (underdetermined / degenerate high degree) → NaN θ → the
    // sandbox reports a clean non-finite failure.
    const theta = fit ? fit.theta : Array.from({ length: degree + 1 }, () => NaN);
    const cond = fit ? fit.cond : NaN;
    const m = metricsOf(theta, degree, data, fitOn, cond);
    const wStr = Array.from({ length: degree + 1 }, (_, j) => `w${j} = ${theta[j].toFixed(3)}`).join(', ');
    return {
      algorithm: algorithmOf(theta, degree, fitOn),
      visuals: [
        ...buildVisuals(data, theta, degree),
        ...buildMatrices(data, theta, degree, fitOn),
      ],
      math: [
        { latex: `\\phi(u) = [1,\\ u,\\ u^2,\\ \\dots,\\ u^{${degree}}],\\quad u = x/${XMAX}`, id: 'basis' },
        { latex: `\\theta = (\\Phi^T \\Phi)^{-1} \\Phi^T y`, id: 'normal-equation' },
        { latex: `\\hat{y}(u) = ${wStr}` },
      ],
      narration: fitOn === 'all'
        ? `Degree ${degree} polynomial fit on ALL ${data.us.length} points (u = x/${XMAX}): ${wStr} — in-sample MSE = ${m.trainMse.toFixed(4)}. ${fitComment(degree, m.trainMse, m.testMse)}`
        : `Degree ${degree} polynomial fit on ${data.nTrain} training points (u = x/${XMAX}): ${wStr} — train MSE = ${m.trainMse.toFixed(4)}, held-out test MSE = ${m.testMse.toFixed(4)}. ${fitComment(degree, m.trainMse, m.testMse)}`,
      explanation: {
        changed: [],
        why: `Closed-form normal equation on the degree-${degree} polynomial basis φ(u) = [1, u, …, u^${degree}]`,
        formulaRef: 'normal-equation',
        dependsOn: ['linear-algebra', 'simple-linear-regression'],
        gateConcepts: ['polynomial regression', 'normal equation', 'overfitting', 'bias-variance'],
      },
      highlights: [],
      metrics: m,
      events: [{ type: 'fit', label: 'normal-equation', step: 0 }],
      timeline: ['Data', 'Basis Expansion', 'Solve', 'Evaluate'],
    };
  },

  step: (): SimState | null => null, // single-shot normal equation
};

export const polyModule: TopicModule = {
  id: 'polynomial-regression',
  title: 'Polynomial Regression',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 3, coding: 2, visualization: 3, gateFrequency: 4 },
    estimatedHours: 6,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['linear-algebra', 'simple-linear-regression', 'multiple-linear-regression'],
    relatedTopics: ['simple-linear-regression', 'multiple-linear-regression', 'ridge-regression', 'lasso-regression', 'bias-variance'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Geometry: Data vs Fitted Polynomial Curve' },
      { slot: 'primary', component: 'loss-curve', title: 'Train MSE (single-shot fit)' },
      { slot: 'sidebar', component: 'matrix-animator', title: 'Vandermonde Matrices: Φ, ΦᵀΦ, Φᵀy, θ' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivation: Normal Equation on Φ' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Fit Stages' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'degree', label: 'Degree (d)', type: 'number', min: 1, max: 15, step: 1, default: 5 },
    { id: 'noise', label: 'Noise σ', type: 'number', min: 0, max: 2, step: 0.05, default: 0.5 },
    { id: 'nTrain', label: 'Training samples', type: 'number', min: 5, max: 100, step: 1, default: 30 },
    { id: 'nTest', label: 'Test samples', type: 'number', min: 5, max: 100, step: 1, default: 20 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
    { id: 'fitOn', label: 'Fit on', type: 'select', options: [
      { value: 'train', label: 'Train only (held-out test)' },
      { value: 'all', label: 'All data' },
    ], default: 'train' },
  ],
  simulation,
  formulas: polyFormulas,
  derivations: polyDerivations,
  questions: polyQuestions,
  comparisons: polyComparisons,
  failureDemos: polyFailureDemos,
  mistakes: polyMistakes,
  testCases: polyTestCases,
  lossMetricKey: 'trainMse',
  lossMetricKey2: 'testMse',

  validateParams: (p) => {
    const issues: string[] = [];
    const degree = (p.degree as number) ?? 5;
    const nTrain = (p.nTrain as number) ?? 30;
    const noise = (p.noise as number) ?? 0.5;
    if (noise < 0) issues.push('Noise must be non-negative');
    if (degree + 1 > nTrain) {
      issues.push(`Underdetermined: degree + 1 = ${degree + 1} > nTrain = ${nTrain} — ΦᵀΦ is rank-deficient (infinite solutions)`);
    } else if (degree >= 20) {
      issues.push(`Degree ${degree} ≥ 20: the power basis is numerically degenerate — the normal equation will fail; use a lower degree or ridge regularization`);
    } else {
      // Data-driven conditioning probe on the ACTUAL generated data.
      const data = generateData(p);
      const fit = fitPoly(p, data, degree, (p.fitOn as string) === 'all' ? 'all' : 'train');
      if (fit && fit.cond > 1e10) {
        issues.push(`High-degree polynomial basis is ill-conditioned (condition number ≈ ${fit.cond.toExponential(1)}) — fitted coefficients are numerically unstable; prefer a lower degree or ridge regularization`);
      }
    }
    return issues;
  },
};

export function register() {
  registerTopic(polyModule);
}
