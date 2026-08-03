// src/topics/softmax-regression/module.ts
// Multiclass logistic (softmax) regression on 3 Gaussian clusters in 2D.
//
// Parameterization (documented — consistent everywhere incl. the classifier):
//   W is (K × d) — row k is the weight vector w_k of class k.
//   b is (K,)    — per-class bias b_k (NOT a shared scalar, NOT a bias column).
//   logits z_k(x) = w_k · x + b_k,  ŷ = softmax(z).
//   K = 3 fixed (matches the 3-cluster simulation; the matrix-animator W rows
//   are classes; the classifier returns 0..2). d = 2 (scatter/decision-boundary
//   are 2D). The K = 2 → sigmoid reduction is covered as a formula + question.
//
// GD runs in feature-standardized space (shared per-feature z-score across all
// classes, bias NOT penalized) exactly like multiple-linear-regression: the
// per-class round trip is W̃_kj = W_kj·σ_j and b̃_k = b_k + Σ_j W_kj·μ_j (the
// UN-standardized W in the bias term — the Wave-1 corrected pair; using w̃·μ
// here leaks bias per epoch and breaks the round trip). One step = one full
// batch epoch over the whole dataset.

import type { TopicModule, Params, SimState, VisualCommand, ParamValue } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import * as viewRegistry from '../../registry/viewRegistry';
import { softmaxTestCases } from './testCases';
import { softmaxFormulas } from './formulas';
import { softmaxDerivations } from './derivations';
import { softmaxMistakes } from './mistakes';
import { softmaxQuestions } from './questions';
import { softmaxComparisons } from './comparisons';
import { softmaxFailureDemos } from './failures';
import { mean } from '../../lib/math/linAlg';

export const SOFTMAX_K = 3; // fixed class count (documented)
export const SOFTMAX_D = 2; // fixed feature count (2D scatter / decision boundary)
const CLASS_COLORS = ['#3b82f6', '#22c55e', '#f59e0b']; // blue, green, orange

export interface SoftmaxData { xs: number[][]; ys: number[]; n: number; d: number; K: number; }

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

/** Cluster centers: equilateral triangle scaled by `margin` (side length 2·margin). */
export function clusterCenters(margin: number): [number, number][] {
  const r3 = Math.sqrt(3);
  return [
    [-margin, -margin / r3],
    [margin, -margin / r3],
    [0, (2 * margin) / r3],
  ];
}

/** Box–Muller standard normal from the seeded RNG (2 uniforms → 1 Gaussian). */
function gauss(rng: () => number): number {
  const u = Math.max(rng(), 1e-12); // avoid log(0)
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Deterministic 3-class data: each class is a Gaussian cluster (σ = 0.5) around
 * a vertex of an equilateral triangle of side 2·margin. With margin = 3 the
 * clusters are well separated → linearly separable → GD reaches accuracy 1.
 * With margin ≈ 1 the clusters overlap heavily (honest non-separable regime).
 */
export function generateData(p: Params): SoftmaxData {
  const nPerClass = (p.nPerClass as number) ?? 20;
  const margin = (p.margin as number) ?? 3;
  const rng = mulberry32((p.seed as number) ?? 42);
  const centers = clusterCenters(margin);
  const xs: number[][] = [];
  const ys: number[] = [];
  for (let k = 0; k < SOFTMAX_K; k++) {
    for (let i = 0; i < nPerClass; i++) {
      const row = centers[k].map((c) => c + 0.5 * gauss(rng)); // σ = 0.5 per cluster
      xs.push(row);
      ys.push(k);
    }
  }
  return { xs, ys, n: xs.length, d: SOFTMAX_D, K: SOFTMAX_K };
}

export interface StdScale { mu: number[]; sigma: number[]; }

/**
 * Per-feature z-score statistics over ALL points (classes pooled) — same μ, σ
 * for every class, so the shared standardization is invertible per class.
 */
export function standardize(data: SoftmaxData): StdScale {
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

/**
 * Numerically STABLE softmax (log-sum-exp with max-shift, REQUIRED — the
 * "trap topic"): ŷ_k = e^{z_k − m} / Σ_j e^{z_j − m}, m = max_j z_j. Identical
 * to the textbook form algebraically (shift invariance), but exp never sees an
 * argument > 0, so huge logits cannot overflow to Infinity/NaN.
 */
export function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const ex = z.map((v) => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / s);
}

export type WMatrix = number[][]; // K × d
export type BiasVec = number[];   // K

export function logitsOf(W: WMatrix, b: BiasVec, row: number[]): number[] {
  return W.map((wRow, k) => wRow.reduce((a, w, j) => a + w * row[j], 0) + b[k]);
}

/** Class probability vector for one point (ORIGINAL feature space). */
export function softmaxProbs(W: WMatrix, b: BiasVec, row: number[]): number[] {
  return softmax(logitsOf(W, b, row));
}

/** Prediction: argmax_k of the logits (equivalently of the probabilities). */
export function predictClass(W: WMatrix, b: BiasVec, row: number[]): number {
  const z = logitsOf(W, b, row);
  let best = 0;
  for (let k = 1; k < z.length; k++) if (z[k] > z[best]) best = k;
  return best;
}

/**
 * Mean categorical cross-entropy with stable log: predictions clipped to
 * [1e-12, 1] so a perfectly confident wrong prediction costs −log(1e-12) ≈ 27.6
 * instead of +∞ (the "log(0) → −∞" failure). L = −(1/n)Σ_i log ŷ_{i,y_i}.
 */
export function categoricalCE(data: SoftmaxData, W: WMatrix, b: BiasVec): number {
  let s = 0;
  for (let i = 0; i < data.n; i++) {
    const p = softmaxProbs(W, b, data.xs[i]);
    const pk = Math.min(1, Math.max(p[data.ys[i]], 1e-12));
    s += -Math.log(pk);
  }
  return s / data.n;
}

/**
 * Analytic gradient of the MEAN categorical CE:
 *   ∂L/∂w_k = (1/n) Σ_i (ŷ_ik − 1{y_i=k}) · x_i
 *   ∂L/∂b_k = (1/n) Σ_i (ŷ_ik − 1{y_i=k})
 * (derived in derivations.ts — the ŷ − indicator result). No RNG; used by the
 * numeric gradient check in testCases.test.ts and by the GD epoch below.
 */
export function categoricalCEGrad(data: SoftmaxData, W: WMatrix, b: BiasVec): { dW: WMatrix; db: BiasVec } {
  const dW = W.map((r) => r.map(() => 0));
  const db = b.map(() => 0);
  for (let i = 0; i < data.n; i++) {
    const yh = softmaxProbs(W, b, data.xs[i]);
    for (let k = 0; k < data.K; k++) {
      const err = yh[k] - (data.ys[i] === k ? 1 : 0); // ŷ_ik − 1{y_i=k}
      for (let j = 0; j < data.d; j++) dW[k][j] += (err * data.xs[i][j]) / data.n;
      db[k] += err / data.n;
    }
  }
  return { dW, db };
}

/** toStandard (per class): W̃_kj = W_kj·σ_j, b̃_k = b_k + Σ_j W_kj·μ_j. */
export function toStandard(W: WMatrix, b: BiasVec, sc: StdScale): { Wt: WMatrix; bt: BiasVec } {
  const Wt = W.map((row) => row.map((w, j) => w * sc.sigma[j]));
  const bt = b.map((bk, k) => bk + W[k].reduce((a, w, j) => a + w * sc.mu[j], 0));
  return { Wt, bt };
}

/** fromStandard (per class): inverse of toStandard (exact round trip). */
export function fromStandard(Wt: WMatrix, bt: BiasVec, sc: StdScale): { W: WMatrix; b: BiasVec } {
  const W = Wt.map((row) => row.map((w, j) => w / sc.sigma[j]));
  const b = bt.map((btk, k) => btk - W[k].reduce((a, w, j) => a + w * sc.mu[j], 0));
  return { W, b };
}

/**
 * ONE full-batch gradient descent epoch on the standardized features:
 *   W ← W − η·∇W,  b ← b − η·∇b   (mean CE gradient; bias NOT penalized).
 */
export function gdEpochStd(p: Params, data: SoftmaxData, Wt: WMatrix, bt: BiasVec, sc: StdScale): { Wt: WMatrix; bt: BiasVec } {
  const lr = (p.learningRate as number) ?? 0.1;
  const dW = Wt.map((r) => r.map(() => 0));
  const db = bt.map(() => 0);
  for (let i = 0; i < data.n; i++) {
    const xstd = data.xs[i].map((v, j) => (v - sc.mu[j]) / sc.sigma[j]);
    const z = Wt.map((row, k) => row.reduce((a, w, j) => a + w * xstd[j], 0) + bt[k]);
    const yh = softmax(z);
    for (let k = 0; k < data.K; k++) {
      const err = yh[k] - (data.ys[i] === k ? 1 : 0);
      for (let j = 0; j < data.d; j++) dW[k][j] += (err * xstd[j]) / data.n;
      db[k] += err / data.n;
    }
  }
  return {
    Wt: Wt.map((row, k) => row.map((v, j) => v - lr * dW[k][j])),
    bt: bt.map((v, k) => v - lr * db[k]),
  };
}

/**
 * Train from zero init for `epochs` full-batch epochs (standardized space),
 * return the ORIGINAL-space W, b. Pure + deterministic given params. Used by
 * the classifier (memoized) so decision-region pixels match the run's LAST
 * snapshot exactly — the documented consistency contract.
 */
export function trainFinal(p: Params): { W: WMatrix; b: BiasVec } {
  const data = generateData(p);
  const sc = standardize(data);
  const epochs = (p.epochs as number) ?? 300;
  let Wt = Array.from({ length: data.K }, () => Array<number>(data.d).fill(0));
  let bt = Array<number>(data.K).fill(0);
  for (let e = 0; e < epochs; e++) {
    const out = gdEpochStd(p, data, Wt, bt, sc);
    Wt = out.Wt; bt = out.bt;
  }
  return fromStandard(Wt, bt, sc);
}

// Classifier cache keyed by the training-affecting params fingerprint — per-pixel
// decision-boundary evaluation calls classifyPoint O(50×50) times; without this
// the first render would retrain the model 2500×.
const classifierCache = new Map<string, { W: WMatrix; b: BiasVec }>();

/**
 * classIdx = argmax_k (w_k·(x,y) + b_k).
 *
 * Consistency contract (documented): the DecisionBoundary visualizer merges the
 * snapshot's algorithm state (w11..w32, b1..b3 — ORIGINAL-space weights, the
 * same ones metricsOf uses for accuracy) into params before calling the
 * classifier, so when those keys are present we classify with EXACTLY the
 * scrubbed step's weights — the boundary is bit-consistent with the snapshot
 * and animates per epoch. Without a snapshot context (empty boundary / params
 * only), we fall back to re-deriving the FINAL trained weights deterministically
 * from params (seeded data + deterministic GD → identical to the run's last
 * snapshot), memoized per fingerprint so per-pixel evaluation never retrains.
 */
export function classifyPoint(x: number, y: number, p: Params): number {
  if (typeof p.w11 === 'number' && typeof p.b3 === 'number') {
    const W = [
      [p.w11 as number, p.w12 as number],
      [p.w21 as number, p.w22 as number],
      [p.w31 as number, p.w32 as number],
    ];
    const b = [p.b1 as number, p.b2 as number, p.b3 as number];
    return predictClass(W, b, [x, y]);
  }
  const key = `${p.seed}|${p.nPerClass}|${p.margin}|${p.learningRate}|${p.epochs}`;
  let model = classifierCache.get(key);
  if (!model) {
    model = trainFinal(p);
    classifierCache.set(key, model);
  }
  return predictClass(model.W, model.b, [x, y]);
}

function metricsOf(W: WMatrix, b: BiasVec, data: SoftmaxData): Record<string, number> {
  let correct = 0;
  for (let i = 0; i < data.n; i++) {
    if (predictClass(W, b, data.xs[i]) === data.ys[i]) correct++;
  }
  return {
    ce: categoricalCE(data, W, b),
    accuracy: correct / data.n,
    misclassCount: data.n - correct,
  };
}

function algorithmOf(W: WMatrix, b: BiasVec, epoch: number): Record<string, ParamValue> {
  const a: Record<string, ParamValue> = { mode: 'softmax-gd', epoch };
  for (let k = 0; k < W.length; k++) {
    for (let j = 0; j < W[0].length; j++) a[`w${k + 1}${j + 1}`] = W[k][j];
    a[`b${k + 1}`] = b[k];
  }
  return a;
}

function weightsFromAlgorithm(a: Record<string, ParamValue>): { W: WMatrix; b: BiasVec } {
  const W = Array.from({ length: SOFTMAX_K }, () => Array<number>(SOFTMAX_D).fill(0));
  const b = Array<number>(SOFTMAX_K).fill(0);
  for (let k = 0; k < SOFTMAX_K; k++) {
    for (let j = 0; j < SOFTMAX_D; j++) W[k][j] = a[`w${k + 1}${j + 1}`] as number;
    b[k] = a[`b${k + 1}`] as number;
  }
  return { W, b };
}

function buildVisuals(data: SoftmaxData, W: WMatrix, b: BiasVec): VisualCommand[] {
  const pts = data.xs.map((row, i) => ({
    type: 'point', id: `d${i}`, x: row[0], y: row[1], color: CLASS_COLORS[data.ys[i]],
  }));
  const Wm = { type: 'matrix', id: 'W (K×d)', rows: W.length, cols: W[0].length, cells: W };
  const bm = { type: 'matrix', id: 'b (K,)', rows: b.length, cols: 1, cells: b.map((v) => [v]) };
  return [...pts, Wm, bm];
}

function snapshotAt(p: Params, data: SoftmaxData, W: WMatrix, b: BiasVec, epoch: number, first: boolean, prevEvents: SimState['events']): SimState {
  const m = metricsOf(W, b, data);
  const accPct = (m.accuracy * 100).toFixed(1);
  const math = [
    { latex: 'z_k = w_k \\cdot x + b_k', id: 'softmax-logits' },
    { latex: '\\hat{y}_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}', id: 'softmax-def' },
    { latex: 'L = -\\frac{1}{n}\\sum_i \\log \\hat{y}_{i,y_i}', id: 'categorical-ce' },
  ];
  const events = [...prevEvents];
  if (m.accuracy === 1 && !events.some((e) => e.label === 'converged')) {
    events.push({ type: 'converged', label: 'converged', step: epoch });
  }
  const why = first
    ? 'Zero init in standardized space: every logit is 0, so ŷ = (1/3, 1/3, 1/3) — pure guessing with CE = ln 3 ≈ 1.099. One GD epoch then moves each class weight toward its cluster.'
    : `Full-batch GD step: W ← W − η·(Ŷ − Y)ᵀX/n with η = ${p.learningRate}. The Ŷ − Y term pushes probability mass toward the true class for every point at once.`;
  return {
    algorithm: algorithmOf(W, b, epoch),
    visuals: buildVisuals(data, W, b),
    math,
    narration: `Epoch ${epoch}: CE = ${m.ce.toFixed(4)}${first ? ' (ln 3, uniform guess)' : ''}, accuracy = ${accPct}% (${m.misclassCount}/${data.n} misclassified) — ${m.accuracy === 1 ? 'every point classified correctly; weights keep sharpening confidence' : m.ce < Math.log(SOFTMAX_K) / 2 ? 'separating: probability mass is concentrating on true classes' : 'early training: logits still near-uniform'}`,
    explanation: {
      changed: first ? [] : [`epoch → ${epoch}`, `CE = ${m.ce.toFixed(4)}`, `accuracy = ${accPct}%`],
      why,
      formulaRef: 'softmax-gradient',
      dependsOn: ['gradient-descent', 'cross-entropy'],
      gateConcepts: ['softmax', 'categorical cross-entropy', 'multiclass logistic'],
    },
    highlights: [],
    metrics: m,
    events,
    timeline: first ? ['Data', 'Fit', 'Evaluate'] : ['Fit', 'Evaluate'],
  };
}

export const simulation = {
  /**
   * Epoch-based run: initialState = epoch 1 (zero init, one GD epoch); each
   * step = one more epoch up to params.epochs. loss-curve plots CE with
   * accuracy as the secondary series; the per-epoch misclassCount metric is
   * reported every snapshot.
   */
  initialState: (p: Params): SimState => {
    const data = generateData(p);
    const sc = standardize(data);
    const z = { Wt: Array.from({ length: data.K }, () => Array<number>(data.d).fill(0)), bt: Array<number>(data.K).fill(0) };
    const out = gdEpochStd(p, data, z.Wt, z.bt, sc);
    const { W, b } = fromStandard(out.Wt, out.bt, sc);
    return snapshotAt(p, data, W, b, 1, true, []);
  },

  step: (p: Params, s: SimState): SimState | null => {
    const epochs = (p.epochs as number) ?? 300;
    const next = ((s.algorithm.epoch as number) ?? 1) + 1;
    if (next > epochs) return null; // epoch budget reached
    const data = generateData(p);
    const sc = standardize(data);
    const { W, b } = weightsFromAlgorithm(s.algorithm);
    const { Wt, bt } = toStandard(W, b, sc);
    const out = gdEpochStd(p, data, Wt, bt, sc);
    const ret = fromStandard(out.Wt, out.bt, sc);
    return snapshotAt(p, data, ret.W, ret.b, next, false, s.events);
  },
};

export const softmaxModule: TopicModule = {
  id: 'softmax-regression',
  title: 'Softmax Regression',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 4, coding: 2, visualization: 3, gateFrequency: 4 },
    estimatedHours: 5,
    revisionPriority: 'P1',
    examFrequency: 'Frequent',
    prerequisites: ['linear-algebra', 'calculus', 'logistic-regression', 'cross-entropy-loss'],
    relatedTopics: ['logistic-regression', 'cross-entropy-loss', 'neural-networks', 'perceptron'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: '3-Class Data: Gaussian Clusters (per-class colors)' },
      { slot: 'primary', component: 'decision-boundary', title: 'Decision Regions (trained softmax model)' },
      { slot: 'sidebar', component: 'loss-curve', title: 'Categorical CE over Epochs (accuracy secondary)' },
    ],
    core: [
      { slot: 'sidebar', component: 'matrix-animator', title: 'Weight Matrix W (rows = classes) + bias b' },
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivation: Softmax Gradient' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'nPerClass', label: 'Points per class', type: 'number', min: 2, max: 50, step: 1, default: 20 },
    { id: 'margin', label: 'Cluster separation (triangle side/2)', type: 'number', min: 0.5, max: 6, step: 0.25, default: 3 },
    { id: 'learningRate', label: 'Learning rate η', type: 'number', min: 0.01, max: 1, step: 0.01, default: 0.1 },
    { id: 'epochs', label: 'Epochs (GD)', type: 'number', min: 10, max: 1000, step: 10, default: 300 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
  ],
  simulation,
  formulas: softmaxFormulas,
  derivations: softmaxDerivations,
  questions: softmaxQuestions,
  comparisons: softmaxComparisons,
  failureDemos: softmaxFailureDemos,
  mistakes: softmaxMistakes,
  testCases: softmaxTestCases,
  lossMetricKey: 'ce',
  lossMetricKey2: 'accuracy',

  validateParams: (p) => {
    const issues: string[] = [];
    const nPerClass = (p.nPerClass as number) ?? 20;
    const lr = (p.learningRate as number) ?? 0.1;
    const epochs = (p.epochs as number) ?? 300;
    const margin = (p.margin as number) ?? 3;
    if (!Number.isFinite(nPerClass) || nPerClass < 2) {
      issues.push('Points per class must be at least 2 (with fewer, a class cannot be estimated)');
    }
    if (!Number.isFinite(lr) || lr <= 0) issues.push('Learning rate must be positive');
    if (!Number.isFinite(epochs) || epochs < 1) issues.push('Epochs must be at least 1');
    if (!Number.isFinite(margin) || margin <= 0) issues.push('Cluster separation must be positive');
    return issues;
  },
};

export function register() {
  registerTopic(softmaxModule);
  // Wave-2 registry contract (viewRegistry.registerClassifier) is added by the
  // Task 5 agent — CALL it when present so the decision-boundary view can
  // resolve this topic's 3-class regions; defensively no-ops until then.
  const reg = (viewRegistry as unknown as { registerClassifier?: (id: string, fn: (x: number, y: number, params: Params) => number) => void }).registerClassifier;
  if (typeof reg === 'function') reg('softmax-regression', classifyPoint);
}
