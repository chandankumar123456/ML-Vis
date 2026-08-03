// src/topics/logistic-regression/module.ts
// Task 6 (Wave 2): logistic-regression — 2D binary classification via gradient
// descent on cross-entropy, with a linear decision boundary at p = 0.5.
//
// Design decisions (deviations from the plan are documented in the report):
//  - Step model: ONE full-batch GD epoch per step (not a λ-style sweep). The
//    initial snapshot is epoch 0 (the init weights) so the whole training
//    trajectory — including the "all probabilities 0.5" start — is scrubbable.
//  - Features are z-scored for training (MLR's toStandard/fromStandard pattern).
//    The affine reparametrization leaves the log-odds z = w·x + b EXACTLY
//    unchanged (σ(z̃) = σ(z)), so the sigmoid sees the same scores; only the
//    conditioning improves. The bias is NOT penalized (no penalty exists here).
//  - Cross-entropy is computed in a numerically stable softplus form:
//    CEᵢ = y·ln(1+e^−z) + (1−y)·ln(1+e^z), with softplus(z) = z + ln(1+e^−z) for
//    z > 0 — no overflow even when the model saturates on separable data.
//  - The decision-boundary view resolves the classifier through
//    viewRegistry.registerClassifier('logistic-regression', classifyByParams).
//    Per the landed DecisionBoundary contract the classifier returns a CLASS
//    INDEX (0/1) and receives the CURRENT snapshot's algorithm state merged
//    into params (the view spreads snapshot.algorithm over params before each
//    grid call). classifyByParams therefore reads the current epoch's weights
//    (w1, w2, b in original coordinates) straight off params — the boundary
//    tracks the exact step being scrubbed, with no re-training. Before a run
//    exists (no snapshot) it falls back to the deterministically re-trained
//    final model (memoized). The probability HEAT (blue→red = p) lives in the
//    scatter-plot point colors, not in this classifier.
import type { TopicModule, Params, SimState, VisualCommand, ParamValue, MathStep } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { registerClassifier } from '../../registry/viewRegistry';
import { logisticTestCases } from './testCases';
import { logisticFormulas } from './formulas';
import { logisticDerivations } from './derivations';
import { logisticMistakes } from './mistakes';
import { logisticQuestions } from './questions';
import { logisticComparisons } from './comparisons';
import { logisticFailureDemos } from './failures';
import { mean } from '../../lib/math/linAlg';

export interface LogisticData { xs: number[][]; ys: number[]; }

/** Mulberry32 — deterministic PRNG so runs are reproducible (matches all Wave-1 topics). */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller Gaussian from two uniform draws (deterministic given the PRNG). */
function gaussian(rng: () => number): number {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Deterministic data synthesis. Two clusters: class 0 centred at (−margin, 0),
 * class 1 at (+margin, 0), each feature spread by `noise` (Gaussian σ). The
 * boundary the model should find is the near-vertical line x₁ ≈ 0.
 * `nClass1` (test-only, imbalance failure demo) overrides the class-1 count;
 * `nonLinear` (test-only, non-linear failure demo) replaces class 1 with an
 * ANNULUS around class 0's central blob — a ring no linear boundary can carve.
 */
export function generateData(p: Params): LogisticData {
  const nPerClass = (p.nPerClass as number) ?? 20;
  const margin = (p.margin as number) ?? 2;
  const noise = (p.noise as number) ?? 1;
  const nClass1 = (p.nClass1 as number) ?? nPerClass;
  const nonLinear = p.nonLinear === true;
  const rng = mulberry32((p.seed as number) ?? 42);
  const xs: number[][] = [];
  const ys: number[] = [];
  const push = (x1: number, x2: number, y: number) => { xs.push([x1, x2]); ys.push(y); };
  if (nonLinear) {
    for (let i = 0; i < nPerClass; i++) push(gaussian(rng) * 0.5, gaussian(rng) * 0.5, 0);
    for (let i = 0; i < nClass1; i++) {
      const ang = rng() * 2 * Math.PI;
      const rad = 2 + rng() * 0.8; // ring radius ∈ [2, 2.8]
      push(rad * Math.cos(ang), rad * Math.sin(ang), 1);
    }
  } else {
    for (let i = 0; i < nPerClass; i++) push(-margin + gaussian(rng) * noise, gaussian(rng) * noise, 0);
    for (let i = 0; i < nClass1; i++) push(margin + gaussian(rng) * noise, gaussian(rng) * noise, 1);
  }
  return { xs, ys };
}

/**
 * Numerically stable sigmoid. For z ≥ 0: 1/(1+e^−z); for z < 0: e^z/(1+e^z) —
 * the exponential never appears in its overflow direction, so σ is well defined
 * for every finite z, σ(0) = 0.5 exactly, and σ → {0,1} without ever hitting
 * the endpoints (which the saturated-sigmoid failure demo leans on).
 */
export function sigmoid(z: number): number {
  if (z >= 0) { const e = Math.exp(-z); return 1 / (1 + e); }
  const e = Math.exp(z);
  return e / (1 + e);
}

/** softplus(z) = ln(1+e^z), computed without overflow for large z. */
export function softplus(z: number): number {
  if (z > 0) return z + Math.log(1 + Math.exp(-z));
  return Math.log(1 + Math.exp(z));
}

/**
 * Stable per-point cross-entropy. With ŷ = σ(z):
 *   −ln ŷ = ln(1+e^−z) = softplus(−z),  −ln(1−ŷ) = ln(1+e^z) = softplus(z),
 * so CEᵢ = y·softplus(−z) + (1−y)·softplus(z). Exact for all z (no log(0),
 * no exp overflow) — documented in formulas.ts and derivations.ts.
 */
export function cePoint(z: number, y: number): number {
  return y * softplus(-z) + (1 - y) * softplus(z);
}

/** Log-odds z = w·x + b (bias LAST in theta, matching the design-matrix convention). */
export function logOdds(theta: number[], row: number[]): number {
  let z = theta[row.length];
  for (let j = 0; j < row.length; j++) z += theta[j] * row[j];
  return z;
}

export function predictProb(theta: number[], row: number[]): number {
  return sigmoid(logOdds(theta, row));
}

/** Default 0.5-threshold decision rule (the threshold is a choice — see mistakes.ts). */
export function predictClass(theta: number[], row: number[]): number {
  return predictProb(theta, row) > 0.5 ? 1 : 0;
}

/** Mean cross-entropy over the dataset. */
export function ceLoss(theta: number[], data: LogisticData): number {
  return mean(data.xs.map((row, i) => cePoint(logOdds(theta, row), data.ys[i])));
}

/**
 * The famous result ∂L/∂w = (1/n)Σ(ŷ−y)x, ∂L/∂b = (1/n)Σ(ŷ−y) — derived in
 * derivations.ts. In ORIGINAL coordinates (independent of the standardization
 * used for training, which is an exact reparametrization of the same scores).
 */
export function ceGradient(theta: number[], data: LogisticData): number[] {
  const d = data.xs[0].length;
  const n = data.xs.length;
  const grad = new Array<number>(d + 1).fill(0);
  for (let i = 0; i < n; i++) {
    const r = predictProb(theta, data.xs[i]) - data.ys[i];
    for (let j = 0; j < d; j++) grad[j] += (r * data.xs[i][j]) / n;
    grad[d] += r / n;
  }
  return grad;
}

export interface StdScale { mu: number[]; sigma: number[]; }

/**
 * Per-feature z-score statistics (identical convention to MLR/ridge: population
 * std with a 1e-9 floor). Training runs in standardized space for conditioning;
 * θ is converted back to original coordinates for display, metrics and the
 * decision-boundary classifier.
 */
export function standardize(data: LogisticData): StdScale {
  const d = data.xs[0].length;
  const mu: number[] = [];
  const sigma: number[] = [];
  for (let j = 0; j < d; j++) {
    const col = data.xs.map((r) => r[j]);
    const m = mean(col);
    mu.push(m);
    sigma.push(Math.sqrt(Math.max(mean(col.map((v) => (v - m) ** 2)), 1e-9)));
  }
  return { mu, sigma };
}

// Corrected MLR round-trip pair (w̃ⱼ = wⱼσⱼ, b̃ = b + Σ wⱼμⱼ) — the algebra that
// keeps z = Σ wⱼxⱼ + b identical in both coordinate systems.
function toStandard(theta: number[], sc: StdScale): number[] {
  const d = sc.mu.length;
  const ws = theta.slice(0, d).map((w, j) => w * sc.sigma[j]);
  const bTilde = theta[d] + theta.slice(0, d).reduce((a, w, j) => a + w * sc.mu[j], 0);
  return [...ws, bTilde];
}

function fromStandard(thetaTilde: number[], sc: StdScale): number[] {
  const d = sc.mu.length;
  const ws = thetaTilde.slice(0, d).map((wj, j) => wj / sc.sigma[j]);
  const b = thetaTilde[d] - ws.reduce((a, w, j) => a + w * sc.mu[j], 0);
  return [...ws, b];
}

/**
 * Initial weights in STANDARD space (training coordinates). 'zero' → all log-odds
 * 0 → every p = 0.5 (the maximum-entropy start, CE = ln 2 on balanced data).
 * 'random' → small seeded draws (±initScale, default 0.1) on a PRNG seeded
 * independently of the data draw (seed + 1), so the init is reproducible without
 * perturbing the dataset.
 */
export function initWeights(p: Params): number[] {
  const init = (p.init as string) ?? 'random';
  const scale = (p.initScale as number) ?? 0.1;
  if (init === 'zero') return [0, 0, 0];
  const rng = mulberry32(((p.seed as number) ?? 42) + 1);
  return Array.from({ length: 3 }, () => (rng() - 0.5) * 2 * scale);
}

/**
 * ONE full-batch gradient-descent epoch on the standardized design matrix:
 * θ̃ ← θ̃ − lr·(1/n)Σ(ŷ−y)x̃, b̃ ← b̃ − lr·(1/n)Σ(ŷ−y). O(n·d) per epoch.
 */
export function gdEpochStd(p: Params, data: LogisticData, thetaTilde: number[], sc: StdScale): number[] {
  const { xs, ys } = data;
  const n = xs.length;
  const d = xs[0].length;
  const lr = (p.lr as number) ?? 0.3;
  const grad = new Array<number>(d + 1).fill(0);
  for (let i = 0; i < n; i++) {
    let z = thetaTilde[d];
    for (let j = 0; j < d; j++) z += thetaTilde[j] * ((xs[i][j] - sc.mu[j]) / sc.sigma[j]);
    const r = sigmoid(z) - ys[i];
    for (let j = 0; j < d; j++) grad[j] += (r * (xs[i][j] - sc.mu[j])) / (sc.sigma[j] * n);
    grad[d] += r / n;
  }
  return thetaTilde.map((v, j) => v - lr * grad[j]);
}

/**
 * Full training run: init → `epochs` GD epochs in standard space → θ in original
 * coordinates. Deterministic given params. This is what the decision-boundary
 * classifier re-runs (memoized) to reconstruct the FINAL trained model.
 */
export function train(p: Params): { theta: number[]; data: LogisticData } {
  const data = generateData(p);
  const sc = standardize(data);
  const epochs = (p.epochs as number) ?? 200;
  let thetaTilde = initWeights(p);
  for (let e = 0; e < epochs; e++) thetaTilde = gdEpochStd(p, data, thetaTilde, sc);
  return { theta: fromStandard(thetaTilde, sc), data };
}

// ---- classifier for the decision-boundary view ----------------------------------
// Contract (landed by the Wave-2 decision-boundary task in parallel):
//   registerClassifier(id, (x, y, params) => number)  /  getClassifier(id)
// The view samples the classifier on a 50×50 lattice and maps the output to a
// palette entry via floor(cls) (class index 0/1), and scans for a SIGN FLIP to
// fit the boundary line. It also merges the current snapshot's algorithm state
// into params before every call — so the CURRENT step's weights arrive directly.
// classifyByParams returns the predicted CLASS INDEX of the point:
//   - primary: z = w1·x + w2·y + b with the snapshot's weights (params.w1/w2/b)
//     → the boundary reflects the exact epoch being scrubbed (cheap: a score
//     and a sign per grid cell, no re-training);
//   - fallback (no snapshot yet): the final trained model, reconstructed
//     deterministically from params and memoized so the grid retrains ≤ once.

function paramsKey(p: Params): string {
  return JSON.stringify([
    p.nPerClass, p.margin, p.noise, p.lr, p.epochs, p.init, p.seed,
    p.nClass1, p.nonLinear, p.initScale,
  ]);
}

let trainCache: { key: string; theta: number[] } | null = null;

export function classifyByParams(x: number, y: number, p: Params): number {
  const w1 = p.w1 as number | undefined;
  const w2 = p.w2 as number | undefined;
  const b = p.b as number | undefined;
  if (typeof w1 === 'number' && typeof w2 === 'number' && typeof b === 'number' && Number.isFinite(w1 + w2 + b)) {
    return w1 * x + w2 * y + b > 0 ? 1 : 0;
  }
  const key = paramsKey(p);
  if (!trainCache || trainCache.key !== key) {
    trainCache = { key, theta: train(p).theta };
  }
  const [t1, t2, t3] = trainCache.theta;
  return t1 * x + t2 * y + t3 > 0 ? 1 : 0;
}

// ---- metrics ---------------------------------------------------------------------

function metricsOf(theta: number[], data: LogisticData, epoch: number, sc: StdScale, p: Params): Record<string, number> {
  const d = data.xs[0].length;
  const n = data.xs.length;
  const ps = data.xs.map((row) => predictProb(theta, row));
  const preds = ps.map((pp) => (pp > 0.5 ? 1 : 0));
  const m: Record<string, number> = {};
  for (let j = 0; j < d; j++) m[`w${j + 1}`] = theta[j];
  m.b = theta[d];
  m.epoch = epoch;
  m.ce = ceLoss(theta, data);
  m.ceInit = ceLoss(fromStandard(initWeights(p), sc), data); // loss at epoch 0 (monotone reference)
  let c0Tot = 0, c0Ok = 0, c1Tot = 0, c1Ok = 0;
  for (let i = 0; i < n; i++) {
    if (data.ys[i] === 0) { c0Tot++; if (preds[i] === 0) c0Ok++; }
    else { c1Tot++; if (preds[i] === 1) c1Ok++; }
  }
  m.accuracy = (c0Ok + c1Ok) / n;
  m.accClass0 = c0Tot ? c0Ok / c0Tot : 1;
  m.accClass1 = c1Tot ? c1Ok / c1Tot : 1;
  m.meanP = mean(ps);
  m.posFrac = mean(data.ys);
  return m;
}

// ---- visuals ---------------------------------------------------------------------

/** Blue (p = 0) → red (p = 1) linear RGB blend — the probability heat coloring. */
function heatColor(p: number): string {
  const r = Math.round(37 + (220 - 37) * p);
  const g = Math.round(99 + (38 - 99) * p);
  const b = Math.round(235 + (38 - 235) * p);
  return `rgb(${r}, ${g}, ${b})`;
}

/** Clip the boundary line w·x + b = 0 to the data bounding box (Liang–Barsky style). */
function boundarySegment(theta: number[], data: LogisticData): VisualCommand | null {
  const [w1, w2, b] = theta;
  const normSq = w1 * w1 + w2 * w2;
  if (normSq < 1e-12) return null; // zero weights → no line (every p = 0.5)
  const xs0 = data.xs.map((r) => r[0]);
  const xs1 = data.xs.map((r) => r[1]);
  const loX = Math.min(...xs0), hiX = Math.max(...xs0);
  const loY = Math.min(...xs1), hiY = Math.max(...xs1);
  // parametric form: p(t) = p0 + t·d, with p0 the closest point of the line to
  // the origin and d a direction along it.
  const p0x = -b * w1 / normSq, p0y = -b * w2 / normSq;
  const dx = -w2, dy = w1;
  let tMin = -Infinity, tMax = Infinity;
  for (const [lo, hi, c, d] of [[loX, hiX, p0x, dx], [loY, hiY, p0y, dy]] as const) {
    if (Math.abs(d) < 1e-12) continue;
    const t1 = (lo - c) / d, t2 = (hi - c) / d;
    tMin = Math.max(tMin, Math.min(t1, t2));
    tMax = Math.min(tMax, Math.max(t1, t2));
  }
  if (tMin > tMax || !Number.isFinite(tMin) || !Number.isFinite(tMax)) return null;
  return {
    type: 'line', id: 'boundary',
    points: [[p0x + tMin * dx, p0y + tMin * dy], [p0x + tMax * dx, p0y + tMax * dy]] as [number, number][],
    color: '#f59e0b',
  };
}

function buildVisuals(data: LogisticData, theta: number[]): VisualCommand[] {
  const pts = data.xs.map((row, i) => ({
    type: 'point', id: `p${i}`, x: row[0], y: row[1],
    color: heatColor(predictProb(theta, row)),
  }));
  const line = boundarySegment(theta, data);
  return line ? [...pts, line] : pts;
}

/**
 * Matrix story for matrix-animator: w (3×1, bias last), a representative sample
 * x (2×1), the dot product z = w·x + b (1×1) and the sigmoid output p (1×1).
 * Small (8 cells) so every epoch snapshot can carry them — scrubbing animates
 * w and p as training progresses. The full X matrix (n×3) is omitted: it is
 * constant across epochs and would bloat each snapshot (MLR GD precedent).
 */
function buildMatrices(data: LogisticData, theta: number[]): VisualCommand[] {
  const x = data.xs[0];
  const z = logOdds(theta, x);
  return [
    { type: 'matrix', id: 'w', rows: 3, cols: 1, cells: [[theta[0]], [theta[1]], [theta[2]]] },
    { type: 'matrix', id: 'x', rows: 2, cols: 1, cells: [[x[0]], [x[1]]] },
    { type: 'matrix', id: 'z = w·x + b', rows: 1, cols: 1, cells: [[z]] },
    { type: 'matrix', id: 'p = σ(z)', rows: 1, cols: 1, cells: [[sigmoid(z)]] },
  ];
}

// ---- simulation ------------------------------------------------------------------

function snapshotAt(
  p: Params, data: LogisticData, sc: StdScale, theta: number[], epoch: number,
  events: SimState['events'], first: boolean,
): SimState {
  const m = metricsOf(theta, data, epoch, sc, p);
  const wStr = `w = (${theta[0].toFixed(3)}, ${theta[1].toFixed(3)}), b = ${theta[2].toFixed(3)}`;
  const math: MathStep[] = [
    { latex: 'z = w \\cdot x + b', id: 'log-odds' },
    { latex: '\\hat{y} = \\sigma(z) = \\frac{1}{1 + e^{-z}}', id: 'sigmoid' },
    { latex: 'L = \\frac{1}{n}\\sum_i \\left[ y_i \\ln(1 + e^{-z_i}) + (1 - y_i) \\ln(1 + e^{z_i}) \\right]', id: 'cross-entropy' },
  ];
  const initLabel = (p.init as string) ?? 'random';
  return {
    algorithm: {
      mode: 'logistic-gd', epoch, lr: (p.lr as number) ?? 0.3, init: initLabel,
      w1: theta[0], w2: theta[1], b: theta[2],
    },
    visuals: [...buildVisuals(data, theta), ...buildMatrices(data, theta)],
    math,
    narration: first
      ? `Epoch 0 (init = ${initLabel}): ${wStr} — CE = ${m.ce.toFixed(4)}, accuracy = ${(m.accuracy * 100).toFixed(1)}%`
      : `Epoch ${epoch}: ${wStr} — CE = ${m.ce.toFixed(4)}, accuracy = ${(m.accuracy * 100).toFixed(1)}%`,
    explanation: {
      changed: first ? [] : [`epoch → ${epoch}`, `CE = ${m.ce.toFixed(4)}`],
      why: first
        ? `Initialization (${initLabel}, standardized features): ${initLabel === 'zero' ? 'every probability is exactly 0.5 — the maximum-entropy start, CE = ln 2 on balanced data' : 'small random weights — every probability is near 0.5'}`
        : `One full-batch GD epoch: θ ← θ − lr·(1/n)Σ(ŷ−y)x (features z-scored; the bias is not penalized)`,
      formulaRef: 'ce-gradient',
      dependsOn: ['gradient-descent', 'sigmoid'],
      gateConcepts: ['logistic regression', 'cross-entropy', 'gradient descent'],
    },
    highlights: [],
    metrics: m,
    events,
    timeline: first ? ['Initialize', 'Evaluate'] : ['Fit', 'Evaluate'],
  };
}

export const simulation = {
  /** Epoch 0: the init weights (before any update) in original coordinates. */
  initialState: (p: Params): SimState => {
    const data = generateData(p);
    const sc = standardize(data);
    const theta = fromStandard(initWeights(p), sc);
    return snapshotAt(p, data, sc, theta, 0, [{ type: 'init', label: 'initialized', step: 0 }], true);
  },

  /** ONE gradient-descent epoch. Returns null when the epoch budget is exhausted. */
  step: (p: Params, s: SimState): SimState | null => {
    const data = generateData(p);
    const epochs = (p.epochs as number) ?? 200;
    const current = ((s.algorithm.epoch as number) ?? 0) + 1;
    if (current > epochs) return null;
    const sc = standardize(data);
    const theta = fromStandard(
      gdEpochStd(p, data, toStandard(thetaFromAlgorithm(s.algorithm, 2), sc), sc),
      sc,
    );
    return snapshotAt(p, data, sc, theta, current, [...s.events], false);
  },
};

function thetaFromAlgorithm(a: Record<string, ParamValue>, d: number): number[] {
  const ws: number[] = [];
  for (let j = 0; j < d; j++) ws.push(a[`w${j + 1}`] as number);
  ws.push(a.b as number);
  return ws;
}

export const logisticModule: TopicModule = {
  id: 'logistic-regression',
  title: 'Logistic Regression',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 4, coding: 2, visualization: 3, gateFrequency: 5 },
    estimatedHours: 6,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['linear-algebra', 'calculus', 'simple-linear-regression', 'gradient-descent'],
    relatedTopics: ['multiple-linear-regression', 'cross-entropy-loss', 'perceptron', 'svm', 'naive-bayes'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Points Heat-Colored by Predicted Probability + Boundary' },
      { slot: 'primary', component: 'decision-boundary', title: 'Decision Boundary: Linear Frontier at p = 0.5' },
      { slot: 'primary', component: 'loss-curve', title: 'Cross-Entropy Loss over Epochs' },
      { slot: 'sidebar', component: 'matrix-animator', title: 'Log-Odds z = w·x + b: w, x, Dot Product, σ(z)' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivations: CE from MLE, (ŷ−y)x Gradient' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Fit Stages' },
      { slot: 'sidebar', component: 'explain-step', title: 'Step Explanation' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'nPerClass', label: 'Points per class', type: 'number', min: 2, max: 100, step: 1, default: 20 },
    { id: 'margin', label: 'Cluster separation margin', type: 'number', min: 0.2, max: 4, step: 0.2, default: 2 },
    { id: 'noise', label: 'Cluster spread σ', type: 'number', min: 0.2, max: 2.5, step: 0.1, default: 1 },
    { id: 'lr', label: 'Learning rate η', type: 'number', min: 0.01, max: 1, step: 0.01, default: 0.3 },
    { id: 'epochs', label: 'Epochs (GD)', type: 'number', min: 10, max: 1000, step: 10, default: 200 },
    {
      id: 'init', label: 'Initialization', type: 'select',
      options: [
        { value: 'random', label: 'Random (small, seeded)' },
        { value: 'zero', label: 'Zero (all p = 0.5)' },
      ],
      default: 'random',
    },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
  ],
  simulation,
  formulas: logisticFormulas,
  derivations: logisticDerivations,
  questions: logisticQuestions,
  comparisons: logisticComparisons,
  failureDemos: logisticFailureDemos,
  mistakes: logisticMistakes,
  testCases: logisticTestCases,
  lossMetricKey: 'ce',

  validateParams: (p) => {
    const issues: string[] = [];
    const nPerClass = (p.nPerClass as number) ?? 20;
    const margin = (p.margin as number) ?? 2;
    const noise = (p.noise as number) ?? 1;
    const lr = (p.lr as number) ?? 0.3;
    const epochs = (p.epochs as number) ?? 200;
    if (!Number.isFinite(nPerClass) || nPerClass < 2) {
      issues.push('nPerClass must be ≥ 2 — each class needs at least 2 points for a meaningful class region');
    }
    if (nPerClass > 200) issues.push('nPerClass > 200 exceeds the lightweight demo size (keep n ≤ ~200 for smooth scrubbing)');
    if (!Number.isFinite(margin) || margin < 0) issues.push('margin must be non-negative (the signed cluster-center offset)');
    if (!Number.isFinite(noise) || noise <= 0) issues.push('noise (cluster spread σ) must be positive');
    if (!Number.isFinite(lr) || lr <= 0) issues.push('Learning rate must be positive');
    if (lr > 1) issues.push('Learning rate > 1 will oscillate or diverge for this full-batch update');
    if (!Number.isFinite(epochs) || epochs < 1) issues.push('epochs must be ≥ 1 (each step is one full-batch GD epoch)');
    return issues;
  },
};

export function register() {
  registerTopic(logisticModule);
  // Classifier registry provided by the Wave-2 decision-boundary view task
  // (viewRegistry.ts): the view's getClassifier('logistic-regression') lookup
  // resolves this callback to paint class regions + the boundary line.
  registerClassifier(logisticModule.id, classifyByParams);
}
