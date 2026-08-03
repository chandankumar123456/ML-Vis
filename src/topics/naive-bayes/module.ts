// src/topics/naive-bayes/module.ts
import type { TopicModule, Params, SimState, VisualCommand, MathStep } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import * as viewRegistry from '../../registry/viewRegistry';
import { nbTestCases } from './testCases';
import { nbFormulas } from './formulas';
import { nbDerivations } from './derivations';
import { nbMistakes } from './mistakes';
import { nbQuestions } from './questions';
import { nbComparisons } from './comparisons';
import { nbFailureDemos } from './failures';

// ============================================================================
// Model decision (documented deviation from the plan draft — see report):
//  - PRIMARY model: Gaussian NB over CONTINUOUS features (matches plan test 4).
//    The `smoothing` param is a VARIANCE FLOOR: σ̃² = σ̂² + α (additive α on the
//    per-class per-feature variance). α = 0 means no floor.
//  - The plan's test 3 ("unseen feature value") is inherently CATEGORICAL. It is
//    delivered by a `discrete` toggle that swaps in a tiny categorical NB over a
//    hard-coded 2-class × 4-sample table (values {0,1} for class 0, {2,3} for
//    class 1), where `smoothing` is Laplace's additive α: (count + α)/(n + α·V).
//  - The "without independence" contrast (plan test 2) is the full-covariance
//    bivariate Gaussian per class (the true generative model): the naive model
//    is exactly the diagonal-covariance restriction of it.
// ============================================================================

// Per-class centers in feature space. With the constant-marginal latent
// construction below, the within-class correlation is EXACTLY `correlation`.
export const CENTERS: Record<number, [number, number][]> = {
  2: [[0, 0], [3, 0]],
  3: [[0, 0], [3, 0], [1.5, 2.6]],
};

export const CLASS_COLORS = ['#2563eb', '#f97316', '#10b981'];
export const TRUE_BOUNDARY_COLOR = '#3b82f6';
export const QUERY_COLOR = '#ef4444';

// Sweep steps MUST match the UI slider steps (0.05) so the last snapshot is
// always exactly the slider's value (mirrors ridge's LAMBDA_STEP discipline).
export const CORR_STEP = 0.05;
export const SMOOTH_STEP = 0.05;

// Default query point — sits just off the class-1 blob so the high-correlation
// demo shows the naive-vs-true posterior divergence (and the argmax flip)
// immediately at the default parameters.
export const DEFAULT_QUERY: [number, number] = [2.4, 2];

export interface NbPoint { c: number; x1: number; x2: number; }
export interface ContinuousData { points: NbPoint[]; nClasses: number; nPerClass: number; }
export interface GaussianFit { mean: [number, number]; var: [number, number]; cov: number; prior: number; n: number; }
export interface CategoricalFit { counts1: number[]; counts2: number[]; vocab: number; prior: number; n: number; }

/**
 * Hard-coded categorical table for the smoothing / zero-probability story.
 * Class 0 lives on values {0,1}, class 1 on {2,3} — the vocabularies are
 * disjoint, so ANY query mixing the two ranges has an "unseen value" in each
 * class (deterministic, hand-verifiable, no RNG).
 */
export const CRAFTED_DISCRETE: NbPoint[] = [
  { c: 0, x1: 0, x2: 0 }, { c: 0, x1: 0, x2: 1 }, { c: 0, x1: 1, x2: 0 }, { c: 0, x1: 1, x2: 1 },
  { c: 1, x1: 2, x2: 2 }, { c: 1, x1: 2, x2: 3 }, { c: 1, x1: 3, x2: 2 }, { c: 1, x1: 3, x2: 3 },
];

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

/** Box–Muller standard normal from the seeded stream. */
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
 * Continuous data synthesis. Each sample shares one latent draw u across its
 * two features: x₁ = μ₁ + √ρ·u + √(1−ρ)·w₁, x₂ = μ₂ + √ρ·u + √(1−ρ)·w₂, with
 * u, w₁, w₂ i.i.d. N(0,1) (seeded). Then within a class Var(x₁)=Var(x₂)=1 for
 * every ρ (constant marginals — the naive model's fitted variances do NOT move
 * with ρ, so the boundary story is clean) and Corr(x₁,x₂) = ρ EXACTLY. This is
 * the "two features sharing a latent" construction the plan calls for, with the
 * correlation mapping ρ(feature correlation) = slider value (the drift note's
 * x₁=u+c·v / x₂=u−c·v form would give the nonlinear map (1−c²)/(1+c²)).
 */
export function generateData(p: Params): ContinuousData {
  const nClasses = (p.nClasses as number) ?? 2;
  const nPerClass = (p.nPerClass as number) ?? 25;
  const rho = (p.correlation as number) ?? 0.9;
  const rng = mulberry32((p.seed as number) ?? 42);
  const normal = makeNormal(rng);
  const centers = CENTERS[nClasses];
  const a = Math.sqrt(rho);
  const b = Math.sqrt(1 - rho);
  const points: NbPoint[] = [];
  for (let c = 0; c < nClasses; c++) {
    const [m1, m2] = centers[c];
    for (let i = 0; i < nPerClass; i++) {
      const u = normal();
      points.push({ c, x1: m1 + a * u + b * normal(), x2: m2 + a * u + b * normal() });
    }
  }
  return { points, nClasses, nPerClass };
}

/**
 * Gaussian NB fit: per-class means, MLE per-feature variances + the variance
 * floor α (σ̃² = σ̂² + α, so `smoothing` = 0 disables the floor), the within-class
 * covariance (for the full-covariance contrast), and the empirical prior n_c/N.
 * The 1e-9 clamp only guards the numerically degenerate n_c=1-with-zero-spread
 * case (never reachable from the seeded draws, but keeps densities finite).
 */
export function fitGaussianNB(data: ContinuousData, nClasses: number, alpha: number): GaussianFit[] {
  const fits: GaussianFit[] = [];
  for (let c = 0; c < nClasses; c++) {
    const pts = data.points.filter((d) => d.c === c);
    const n = Math.max(pts.length, 1);
    const m1 = pts.reduce((s, d) => s + d.x1, 0) / n;
    const m2 = pts.reduce((s, d) => s + d.x2, 0) / n;
    const v1 = Math.max(pts.reduce((s, d) => s + (d.x1 - m1) ** 2, 0) / n, 1e-9) + alpha;
    const v2 = Math.max(pts.reduce((s, d) => s + (d.x2 - m2) ** 2, 0) / n, 1e-9) + alpha;
    const cov = pts.reduce((s, d) => s + (d.x1 - m1) * (d.x2 - m2), 0) / n;
    fits.push({ mean: [m1, m2], var: [v1, v2], cov, prior: pts.length / data.points.length, n: pts.length });
  }
  return fits;
}

/**
 * Categorical NB fit over the crafted table. `vocab` = number of distinct
 * feature values in the training data (4: {0,1,2,3} for both features) — the V
 * in the Laplace denominator (count + α)/(n + α·V). Priors are empirical (½/½).
 * Smoothing is applied at likelihood time (see categoricalLogLik), so the
 * fitted counts are α-independent.
 */
export function fitCategoricalNB(): CategoricalFit[] {
  // counts per class (rows) per value in {0,1,2,3} (columns), for each feature
  const counts1 = [[0, 0, 0, 0], [0, 0, 0, 0]];
  const counts2 = [[0, 0, 0, 0], [0, 0, 0, 0]];
  for (const d of CRAFTED_DISCRETE) {
    counts1[d.c][d.x1]++;
    counts2[d.c][d.x2]++;
  }
  const vocab = 4;
  return [
    { counts1: counts1[0], counts2: counts2[0], vocab, prior: 0.5, n: 4 },
    { counts1: counts1[1], counts2: counts2[1], vocab, prior: 0.5, n: 4 },
  ];
}

/** Log P(x|C) for the naive Gaussian model — the independence (diagonal) product. */
export function gaussianLogLik(fit: GaussianFit, x1: number, x2: number): number {
  const [v1, v2] = fit.var;
  return -0.5 * (Math.log(2 * Math.PI * v1) + (x1 - fit.mean[0]) ** 2 / v1)
    - 0.5 * (Math.log(2 * Math.PI * v2) + (x2 - fit.mean[1]) ** 2 / v2);
}

/**
 * Log P(x|C) for the FULL-covariance bivariate Gaussian — the generative truth
 * the naive model approximates. det is clamped to 1e-12 purely as a numerical
 * guard against the astronomically-rare exactly-collinear sample.
 */
export function jointGaussianLogLik(fit: GaussianFit, x1: number, x2: number): number {
  const [v1, v2] = fit.var;
  const det = Math.max(v1 * v2 - fit.cov * fit.cov, 1e-12);
  const inv11 = v2 / det;
  const inv12 = -fit.cov / det;
  const inv22 = v1 / det;
  const dx1 = x1 - fit.mean[0];
  const dx2 = x2 - fit.mean[1];
  const q = dx1 * (inv11 * dx1 + inv12 * dx2) + dx2 * (inv12 * dx1 + inv22 * dx2);
  return -0.5 * (Math.log(2 * Math.PI * Math.sqrt(det)) * 2 + q);
}

/** Log P(x|C) for the categorical model: product of two Laplace-smoothed margins. */
export function categoricalLogLik(fit: CategoricalFit, v1: number, v2: number, alpha: number): number {
  const p1 = (fit.counts1[v1] + alpha) / (fit.n + alpha * fit.vocab);
  const p2 = (fit.counts2[v2] + alpha) / (fit.n + alpha * fit.vocab);
  return Math.log(p1) + Math.log(p2);
}

/** Numerically stable softmax; a fully degenerate input (all −∞) maps to zeros. */
export function softmax(logs: number[]): number[] {
  const m = Math.max(...logs);
  if (!Number.isFinite(m)) return logs.map(() => 0);
  const exps = logs.map((l) => Math.exp(l - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => (s > 0 ? e / s : 0));
}

/** Argmax over class log-posteriors; −1 when every class has zero evidence. */
export function argmaxOf(logs: number[]): number {
  if (logs.length === 0) return -1;
  let best = 0;
  for (let i = 1; i < logs.length; i++) if (logs[i] > logs[best]) best = i;
  if (!Number.isFinite(logs[best])) return -1; // all −∞ (no class has any evidence)
  return best;
}

/** Round a continuous query onto the discrete grid {0,1,2,3} (clamped). */
export function discreteQuery(x1: number, x2: number): [number, number] {
  const clamp = (v: number) => Math.min(3, Math.max(0, Math.round(v)));
  return [clamp(x1), clamp(x2)];
}

/** Naive Gaussian NB posterior P(C|x) (normalized, log-space safe). */
export function posteriorOf(fits: GaussianFit[], x1: number, x2: number): number[] {
  return softmax(fits.map((f) => Math.log(f.prior) + gaussianLogLik(f, x1, x2)));
}

/** Full-covariance ("without independence") posterior P(C|x). */
export function jointPosteriorOf(fits: GaussianFit[], x1: number, x2: number): number[] {
  return softmax(fits.map((f) => Math.log(f.prior) + jointGaussianLogLik(f, x1, x2)));
}

// Fit memo for the decision-boundary grid: DecisionBoundary samples the
// classifier GRID² (50×50 = 2500) times per redraw; generateData + fitGaussianNB
// are fully deterministic in (nClasses, nPerClass, correlation, smoothing, seed),
// so one fit per distinct parameter tuple is cached. Bounded: cleared wholesale
// once it exceeds MAX_FIT_CACHE entries (a scrub sees ≤ ~20 distinct tuples).
const MAX_FIT_CACHE = 64;
const gaussianFitCache = new Map<string, GaussianFit[]>();

function fitKey(p: Params): string {
  return [p.nClasses ?? 2, p.nPerClass ?? 25, p.correlation ?? 0.9, p.smoothing ?? 0.1, p.seed ?? 42].join(':');
}

function cachedGaussianFits(p: Params, nClasses: number, alpha: number): GaussianFit[] {
  const key = fitKey(p);
  const hit = gaussianFitCache.get(key);
  if (hit) return hit;
  const fits = fitGaussianNB(generateData(p), nClasses, alpha);
  if (gaussianFitCache.size >= MAX_FIT_CACHE) gaussianFitCache.clear();
  gaussianFitCache.set(key, fits);
  return fits;
}

/**
 * The registered classifier: Gaussian NB (or categorical NB in discrete mode),
 * fully deterministic from params-seeded data. Consumed by the decision-boundary
 * view via getClassifier('naive-bayes').
 */
export function classifyNB(x: number, y: number, params: Params): number {
  if (params.discrete === true) {
    const alpha = (params.smoothing as number) ?? 0;
    const [v1, v2] = discreteQuery(x, y);
    const fits = fitCategoricalNB();
    return argmaxOf(fits.map((f) => Math.log(f.prior) + categoricalLogLik(f, v1, v2, alpha)));
  }
  const nClasses = (params.nClasses as number) ?? 2;
  const alpha = (params.smoothing as number) ?? 0.1;
  const fits = cachedGaussianFits(params, nClasses, alpha);
  return argmaxOf(fits.map((f) => Math.log(f.prior) + gaussianLogLik(f, x, y)));
}

/** Fraction of training points whose posterior argmax matches the true label. */
function trainAccuracy(fits: GaussianFit[], data: ContinuousData): number {
  if (data.points.length === 0) return 0;
  let correct = 0;
  for (const d of data.points) {
    const logs = fits.map((f) => Math.log(f.prior) + gaussianLogLik(f, d.x1, d.x2));
    if (argmaxOf(logs) === d.c) correct++;
  }
  return correct / data.points.length;
}

/** Scan the 2-class decision curve where logPost(C₀) = logPost(C₁) (single branch per column). */
function trueBoundary(fits: GaussianFit[], lo: number, hi: number): [number, number][] {
  const diff = (x1: number, x2: number) =>
    (Math.log(fits[0].prior) + jointGaussianLogLik(fits[0], x1, x2))
    - (Math.log(fits[1].prior) + jointGaussianLogLik(fits[1], x1, x2));
  const STEP = 0.25;
  const pts: [number, number][] = [];
  for (let x1 = lo; x1 <= hi + 1e-9; x1 += STEP) {
    let prev = diff(x1, lo);
    let prevX2 = lo;
    for (let x2 = lo + STEP; x2 <= hi + 1e-9; x2 += STEP) {
      const cur = diff(x1, x2);
      if ((prev > 0) !== (cur > 0) && Number.isFinite(prev) && Number.isFinite(cur)) {
        const t = prev / (prev - cur);
        pts.push([x1, prevX2 + t * STEP]);
        break; // the true (full-covariance) boundary is single-branched here
      }
      prev = cur;
      prevX2 = x2;
    }
  }
  return pts;
}

function dataBounds(data: ContinuousData): { lo: number; hi: number } {
  let lo = Infinity;
  let hi = -Infinity;
  for (const d of data.points) {
    lo = Math.min(lo, d.x1, d.x2);
    hi = Math.max(hi, d.x1, d.x2);
  }
  const pad = (hi - lo) * 0.1 + 0.5;
  return { lo: Math.floor(lo - pad), hi: Math.ceil(hi + pad) };
}

// ============================================================================
// Metrics + visuals
// ============================================================================

function continuousMetrics(
  fits: GaussianFit[], data: ContinuousData, post: number[], postJ: number[],
  naiveLogs: number[], alpha: number, correlation: number,
): Record<string, number> {
  const m: Record<string, number> = {
    nClasses: data.nClasses, nPerClass: data.nPerClass,
    correlation, smoothing: alpha, eps: alpha, // eps = the variance floor applied
    postSum: post.reduce((a, b) => a + b, 0),
    postJointSum: postJ.reduce((a, b) => a + b, 0),
    naiveJointDiff: Math.max(...fits.map((_, c) => Math.abs(post[c] - postJ[c]))),
    acc: trainAccuracy(fits, data),
  };
  fits.forEach((f, c) => {
    m[`prior${c}`] = f.prior;
    m[`mu1_${c}`] = f.mean[0];
    m[`mu2_${c}`] = f.mean[1];
    m[`var1_${c}`] = f.var[0];
    m[`var2_${c}`] = f.var[1];
    m[`cov${c}`] = f.cov;
    m[`post${c}`] = post[c];
    m[`postJoint${c}`] = postJ[c];
    m[`logLik${c}`] = naiveLogs[c];
    m[`rawLik${c}`] = Math.exp(naiveLogs[c]); // tiny (underflow story) but finite
  });
  return m;
}

function discreteMetrics(alpha: number, v1: number, v2: number): Record<string, number> {
  const fits = fitCategoricalNB();
  // lik_c = the pure class-conditional likelihood P(x|C) (NO prior — Bayes factors
  // are the pedagogically meaningful numbers); post applies the prior + normalizes.
  const logLiks = fits.map((f) => categoricalLogLik(f, v1, v2, alpha));
  const liks = logLiks.map((l) => (Number.isFinite(l) ? Math.exp(l) : 0));
  const post = softmax(fits.map((f, i) => Math.log(f.prior) + logLiks[i]));
  return {
    nClasses: 2, nPerClass: CRAFTED_DISCRETE.length / 2, smoothing: alpha,
    // ε = the Laplace-smoothed likelihood of an UNSEEN value: (0 + α)/(n + α·V)
    eps: (0 + alpha) / (fits[0].n + alpha * fits[0].vocab),
    prior0: fits[0].prior, prior1: fits[1].prior,
    lik0: liks[0], lik1: liks[1],
    post0: post[0], post1: post[1],
    postSum: post[0] + post[1],
    acc: 1, // class vocabularies are disjoint, so every training point wins its own class
  };
}

function buildScatter(data: ContinuousData, fits: GaussianFit[], q: [number, number], discrete: boolean): VisualCommand[] {
  const cmds: VisualCommand[] = data.points.map((d, i) => ({
    type: 'point', id: `d${i}`, x: d.x1, y: d.x2, color: CLASS_COLORS[d.c],
  }));
  cmds.push({ type: 'point', id: 'query', x: q[0], y: q[1], color: QUERY_COLOR });
  if (!discrete && data.nClasses === 2) {
    // The true generative (full-covariance) boundary — the naive model's regions
    // are rendered by the decision-boundary view, so scrubbing ρ shows this curve
    // tilting away from the (roughly vertical) naive boundary.
    const { lo, hi } = dataBounds(data);
    const b = trueBoundary(fits, lo, hi);
    if (b.length >= 2) cmds.push({ type: 'line', id: 'true-boundary', points: b, color: TRUE_BOUNDARY_COLOR });
  }
  return cmds;
}

function buildMatricesContinuous(fits: GaussianFit[], post: number[], postJ: number[]): VisualCommand[] {
  const mu = fits.map((f) => [f.mean[0], f.mean[1]]);
  const varmat = fits.map((f) => [f.var[0], f.var[1]]);
  const priors = fits.map((f) => [f.prior]);
  const cmds: VisualCommand[] = [
    { type: 'matrix', id: 'μ (class means)', rows: fits.length, cols: 2, cells: mu },
    { type: 'matrix', id: 'σ² (per-feature)', rows: fits.length, cols: 2, cells: varmat },
    { type: 'matrix', id: 'P(C)', rows: fits.length, cols: 1, cells: priors },
    { type: 'matrix', id: 'P(C|x) naive', rows: fits.length, cols: 1, cells: post.map((v) => [v]) },
    { type: 'matrix', id: 'P(C|x) joint', rows: fits.length, cols: 1, cells: postJ.map((v) => [v]) },
  ];
  // full covariance per class (the "without independence" ingredient)
  fits.forEach((f, c) => {
    cmds.push({ type: 'matrix', id: `Σ${c}`, rows: 2, cols: 2, cells: [[f.var[0], f.cov], [f.cov, f.var[1]]] });
  });
  return cmds;
}

function buildMatricesDiscrete(fits: CategoricalFit[], liks: number[], post: number[]): VisualCommand[] {
  return [
    { type: 'matrix', id: 'counts(x₁)', rows: 2, cols: 4, cells: fits.map((f) => f.counts1) },
    { type: 'matrix', id: 'counts(x₂)', rows: 2, cols: 4, cells: fits.map((f) => f.counts2) },
    { type: 'matrix', id: 'P(C)', rows: 2, cols: 1, cells: fits.map((f) => [f.prior]) },
    { type: 'matrix', id: 'P(x|C)', rows: 2, cols: 1, cells: liks.map((v) => [v]) },
    { type: 'matrix', id: 'P(C|x)', rows: 2, cols: 1, cells: post.map((v) => [v]) },
  ];
}

// ============================================================================
// Snapshots
// ============================================================================

function postStr(post: number[]): string {
  return post.map((v, c) => `P(C${c}|x) = ${v.toFixed(3)}`).join(', ');
}

function continuousSnapshot(p: Params, correlation: number, first: boolean): SimState {
  const data = generateData({ ...p, correlation });
  const nClasses = data.nClasses;
  const alpha = (p.smoothing as number) ?? 0.1;
  const qx = (p.queryX1 as number) ?? DEFAULT_QUERY[0];
  const qy = (p.queryX2 as number) ?? DEFAULT_QUERY[1];
  const fits = fitGaussianNB(data, nClasses, alpha);
  const naiveLogs = fits.map((f) => Math.log(f.prior) + gaussianLogLik(f, qx, qy));
  const post = softmax(naiveLogs);
  const jointLogs = fits.map((f) => Math.log(f.prior) + jointGaussianLogLik(f, qx, qy));
  const postJ = softmax(jointLogs);
  const predClass = argmaxOf(naiveLogs);
  const predJointClass = argmaxOf(jointLogs);
  const m = continuousMetrics(fits, data, post, postJ, naiveLogs, alpha, correlation);
  const math: MathStep[] = [
    { latex: 'P(C \\mid x) = \\frac{P(C)\\,P(x\\mid C)}{P(x)}', id: 'bayes' },
    { latex: 'P(x\\mid C) = \\prod_{j} P(x_j \\mid C)', id: 'naive-likelihood' },
    { latex: 'P(x_j\\mid C) = \\frac{1}{\\sqrt{2\\pi\\sigma_{jC}^2}}\\exp\\!\\left(-\\frac{(x_j-\\mu_{jC})^2}{2\\sigma_{jC}^2}\\right)', id: 'gaussian-likelihood' },
  ];
  if (correlation > 0.001) {
    math.push({
      latex: 'P(x_1, x_2\\mid C) = \\frac{1}{2\\pi\\sqrt{|\\Sigma_C|}}\\exp\\!\\left(-\\frac{1}{2}(x-\\mu_C)^T\\Sigma_C^{-1}(x-\\mu_C)\\right)',
      caption: 'true generative (full-covariance) likelihood — the independence-free contrast',
    });
  }
  const divergent = Math.abs(post[predClass] - postJ[predClass]) > 0.1;
  const narration = `ρ = ${correlation.toFixed(2)}, variance floor α = ${alpha.toFixed(2)}: ${postStr(post)} — `
    + `full-covariance (true generative) posterior: ${postStr(postJ)}. `
    + (divergent
      ? `The posteriors diverge: the naive model treats the correlated features as independent and double-counts the shared evidence.`
      : `With (nearly) independent features the naive and true posteriors agree.`);
  return {
    algorithm: { mode: 'gaussian', nClasses, correlation, smoothing: alpha, predClass, predJointClass },
    visuals: [
      ...buildScatter(data, fits, [qx, qy], false),
      ...buildMatricesContinuous(fits, post, postJ),
    ],
    math,
    narration,
    explanation: {
      changed: first ? [] : [`ρ → ${correlation.toFixed(2)}`, `naive ${predClass === 1 ? 'P(C₁|x)' : 'P(C₀|x)'} = ${post[predClass].toFixed(3)}`, `joint ${predJointClass === 1 ? 'P(C₁|x)' : 'P(C₀|x)'} = ${postJ[predJointClass].toFixed(3)}`],
      why: first
        ? `Gaussian NB fitted per-class means and per-feature variances from the seeded data; posteriors at the query point follow Bayes' theorem with the independence (diagonal-covariance) assumption`
        : `As the within-class correlation grows, the full-covariance (true) posterior moves away from the naive one — the independence assumption double-counts the shared latent evidence`,
      formulaRef: 'bayes',
      dependsOn: ['probability', 'bayes-theorem', 'gaussian-distribution'],
      gateConcepts: ['naive bayes', 'bayes theorem', 'independence assumption', 'posterior'],
    },
    highlights: [],
    metrics: m,
    events: [{ type: 'fit', label: 'gaussian-nb-fit', step: 0 }],
    timeline: ['Data', 'Fit', 'Posterior', 'Evaluate'],
  };
}

function discreteSnapshot(p: Params, alpha: number, first: boolean): SimState {
  const [qv1, qv2] = discreteQuery((p.queryX1 as number) ?? DEFAULT_QUERY[0], (p.queryX2 as number) ?? DEFAULT_QUERY[1]);
  const fits = fitCategoricalNB();
  const logLiks = fits.map((f) => categoricalLogLik(f, qv1, qv2, alpha));
  const liks = logLiks.map((l) => (Number.isFinite(l) ? Math.exp(l) : 0));
  const logs = fits.map((f, i) => Math.log(f.prior) + logLiks[i]);
  const post = softmax(logs);
  const predClass = argmaxOf(logs);
  const m = discreteMetrics(alpha, qv1, qv2);
  const unseenBoth = alpha === 0 && post[0] === 0 && post[1] === 0;
  const math: MathStep[] = [
    { latex: 'P(x_j = v \\mid C) = \\frac{n_{C,v} + \\alpha}{n_C + \\alpha\\, V}', id: 'laplace-smoothing' },
    { latex: 'P(x\\mid C) = P(x_1 = v_1\\mid C)\\, P(x_2 = v_2\\mid C)', id: 'naive-likelihood' },
  ];
  const narration = unseenBoth
    ? `α = 0 (no smoothing): the query (${qv1}, ${qv2}) contains a value unseen in EACH class — both class likelihoods are exactly 0, so the posterior is undefined (0/0). The model cannot classify this point.`
    : `α = ${alpha.toFixed(2)}: Laplace smoothing gives the unseen value ε = ${m.eps.toFixed(4)} (not 0), so P(x|C₀) = ${liks[0].toFixed(4)} and P(x|C₁) = ${liks[1].toFixed(4)} → ${postStr(post)}.`;
  return {
    algorithm: { mode: 'categorical', nClasses: 2, smoothing: alpha, predClass },
    visuals: [
      ...buildScatter({ points: CRAFTED_DISCRETE, nClasses: 2, nPerClass: 4 }, [], [qv1, qv2], true),
      ...buildMatricesDiscrete(fits, liks, post),
    ],
    math,
    narration,
    explanation: {
      changed: first ? [] : [`α → ${alpha.toFixed(2)}`, `ε = ${m.eps.toFixed(4)}`],
      why: first
        ? `Categorical NB over the crafted 2-class table; the query (${qv1}, ${qv2}) has an unseen value in each class, so its fate is decided entirely by smoothing`
        : `Laplace smoothing adds α to every count and α·V to the denominator, turning the unseen value's 0/4 likelihood into (0+α)/(n+α·V) = ε > 0`,
      formulaRef: 'laplace-smoothing',
      dependsOn: ['probability', 'bayes-theorem', 'naive bayes'],
      gateConcepts: ['laplace smoothing', 'zero probability', 'naive bayes'],
    },
    highlights: [],
    metrics: m,
    events: [{ type: 'fit', label: 'categorical-nb-fit', step: 0 }],
    timeline: ['Data', 'Counts', 'Smooth', 'Posterior'],
  };
}

/**
 * Step model (documented): Gaussian NB is a single-shot fit, so a single
 * snapshot would suffice — but mirroring ridge's sweep, the run scrubs the
 * CORRELATION slider from 0 to its target (continuous mode) or the SMOOTHING
 * slider from 0 to its target (discrete mode). Scrubbing therefore shows the
 * independence violation growing (the true boundary tilting away from the
 * naive regions) or the zero-probability posterior being rescued by α.
 */
export const simulation = {
  initialState: (p: Params): SimState => (
    p.discrete === true ? discreteSnapshot(p, 0, true) : continuousSnapshot(p, 0, true)
  ),

  step: (p: Params, s: SimState): SimState | null => {
    if (p.discrete === true) {
      const target = (p.smoothing as number) ?? 0;
      const current = (s.algorithm.smoothing as number) ?? 0;
      const next = current + SMOOTH_STEP;
      if (next > target + 1e-9) return null;
      return discreteSnapshot(p, next, false);
    }
    const target = (p.correlation as number) ?? 0;
    const current = (s.algorithm.correlation as number) ?? 0;
    const next = current + CORR_STEP;
    if (next > target + 1e-9) return null;
    return continuousSnapshot(p, next, false);
  },
};

export const nbModule: TopicModule = {
  id: 'naive-bayes',
  title: 'Naive Bayes Classifier',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 3, coding: 2, visualization: 3, gateFrequency: 4 },
    estimatedHours: 5,
    revisionPriority: 'P1',
    examFrequency: 'Every year',
    prerequisites: ['probability', 'bayes-theorem', 'gaussian-distribution'],
    relatedTopics: ['logistic-regression', 'knn', 'decision-trees', 'lda', 'probability'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Feature Space: Data, Query Point, True Generative Boundary' },
      { slot: 'primary', component: 'decision-boundary', title: 'Naive Bayes Decision Regions' },
      { slot: 'sidebar', component: 'matrix-animator', title: 'Per-Class Tables: μ, σ², Σ, counts, P(C|x)' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivation: Naive Posterior & Log-Space' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Fit Stages' },
      { slot: 'primary', component: 'explain-step', title: 'Why This Step' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'nClasses', label: 'Number of classes (C)', type: 'number', min: 2, max: 3, step: 1, default: 2 },
    { id: 'nPerClass', label: 'Samples per class', type: 'number', min: 5, max: 60, step: 1, default: 25 },
    { id: 'correlation', label: 'Within-class correlation ρ', type: 'number', min: 0, max: 0.95, step: 0.05, default: 0.9 },
    { id: 'smoothing', label: 'Smoothing α (0 = none)', type: 'number', min: 0, max: 1, step: 0.05, default: 0.1 },
    { id: 'queryX1', label: 'Query point x₁', type: 'number', min: -6, max: 6, step: 0.1, default: 2.4 },
    { id: 'queryX2', label: 'Query point x₂', type: 'number', min: -6, max: 6, step: 0.1, default: 2 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
    { id: 'discrete', label: 'Discrete features (categorical NB demo)', type: 'toggle', default: false },
  ],
  simulation,
  formulas: nbFormulas,
  derivations: nbDerivations,
  questions: nbQuestions,
  comparisons: nbComparisons,
  failureDemos: nbFailureDemos,
  mistakes: nbMistakes,
  testCases: nbTestCases,

  validateParams: (p) => {
    const issues: string[] = [];
    const nClasses = p.nClasses as number | undefined;
    if (nClasses !== undefined && (!Number.isInteger(nClasses) || nClasses < 2 || nClasses > 3)) {
      issues.push('nClasses must be 2 or 3 (integer)');
    }
    const nPerClass = p.nPerClass as number | undefined;
    if (nPerClass !== undefined && (!Number.isInteger(nPerClass) || nPerClass < 1)) {
      issues.push('nPerClass must be a positive integer (≥ 1)');
    }
    const corr = p.correlation as number | undefined;
    if (corr !== undefined && !(corr >= 0 && corr < 1)) {
      issues.push('correlation must be in [0, 1) — at ρ = 1 the shared latent collapses the within-class spread onto a line');
    }
    const alpha = p.smoothing as number | undefined;
    if (alpha !== undefined && !Number.isFinite(alpha)) issues.push('smoothing α must be a finite number');
    if (alpha !== undefined && alpha < 0) issues.push('smoothing α must be non-negative (0 = no smoothing, α > 0 = Laplace / variance floor)');
    const qx = p.queryX1 as number | undefined;
    const qy = p.queryX2 as number | undefined;
    if (qx !== undefined && !Number.isFinite(qx)) issues.push('queryX1 must be finite');
    if (qy !== undefined && !Number.isFinite(qy)) issues.push('queryX2 must be finite');
    return issues;
  },
};

// ============================================================================
// Classifier registration. The Task 5 parallel agent adds registerClassifier to
// viewRegistry (plan lines 301-305); resolve defensively through the namespace so
// this topic's lint/vitest gates do not depend on that change having landed in
// the shared tree yet. At integration the real registration runs.
// ============================================================================
type ClassifierFn = (x: number, y: number, params: Params) => number;
const classifierRegistry = viewRegistry as unknown as {
  registerClassifier?: (id: string, fn: ClassifierFn) => void;
};

export function register() {
  registerTopic(nbModule);
  classifierRegistry.registerClassifier?.('naive-bayes', classifyNB);
}
