// src/topics/cross-entropy-loss/module.ts
// TWO FACETS in one module (plan: "same module, two facets"), toggled by the
// `facet` param — mirrors how SLR toggles normal-equation vs GD:
//
//  Facet A 'cross-entropy': binary distributions p = [p0, 1−p0] (true/reference)
//    and q = [q0, 1−q0] (predicted). The run is a SWEEP over q0 from 0.05 up to
//    the slider value at 0.05 steps (≤ 19 snapshots, last exactly the slider, the
//    ridge λ-sweep convention) — scrubbing animates the −log penalty and the
//    H / KL / CE readouts. K = 2 is chosen deliberately: with p1 = 1−p0 and
//    q1 = 1−q0 the simplex constraint Σp = Σq = 1 is automatic, and the plan's
//    own sweep guidance ("sweep q's first probability 0.05→0.95") is only
//    meaningful for two classes (a 3-class q2 = 1−q0−q1 would go negative).
//    The K-class generalization lives in formulas/derivations/questions.
//
//  Facet B 'mle': a deterministic coin-flip dataset (nFlips, heads — no RNG:
//    fully determined by explicit params). The run sweeps θ over a fixed 21-point
//    grid 0.02 → 0.98 (plan's sweep) and reports the Bernoulli likelihood
//    L(θ) = θ^h(1−θ)^(n−h), log-likelihood, per-sample NLL and the MLE at the
//    argmax. The binomial coefficient C(n,h) is a θ-independent constant — it
//    cancels in the argmax, so omitting it makes the per-sample NLL EXACTLY the
//    cross-entropy between the empirical distribution and the model:
//    −log L/n = CE(p̂, q_θ). That is the plan's "NLL = CE for classification".
//
// UNIFIED loss metric: every snapshot emits `cePQ` (the per-sample loss) and `hP`
// (the entropy floor) so the loss-curve view plots "CE vs swept parameter" in BOTH
// facets with the CE = H + KL decomposition visible as the gap between the curves.
//
// 0·log 0 handling (documented in formulas + failures): the term x·log(y) with
// x = 0 contributes 0 (limit x·ln x → 0 as x → 0⁺, the standard information-theory
// convention); with x > 0 and y = 0 the term is −∞ (log 0) — NOT clamped, because
// CE(p,q) is genuinely infinite when q assigns zero probability where p has mass.
// The live sim keeps probabilities in [0.05, 0.95] (full support); the boundary
// cases are demonstrated by the failure demos.
import type { TopicModule, Params, SimState, VisualCommand, ParamValue, MathStep } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { ceTestCases } from './testCases';
import { ceFormulas } from './formulas';
import { ceDerivations } from './derivations';
import { ceMistakes } from './mistakes';
import { ceQuestions } from './questions';
import { ceComparisons } from './comparisons';
import { ceFailureDemos } from './failures';

export const Q0_MIN = 0.05;
export const Q0_MAX = 0.95;
export const Q0_STEP = 0.05;
export const THETA_MIN = 0.02;
export const THETA_MAX = 0.98;

export type Facet = 'cross-entropy' | 'mle';

export function isFacet(v: unknown): v is Facet {
  return v === 'cross-entropy' || v === 'mle';
}

// ===== Information-theory core (natural log — NATS) =====

/** x·log(y) with the 0·log 0 = 0 convention: an x ≤ 0 weight contributes 0. */
export function xlogy(x: number, y: number): number {
  return x <= 0 ? 0 : x * Math.log(y);
}

/** H(p) = −Σ pᵢ·log pᵢ — entropy (nats). */
export function entropy(p: number[]): number {
  return -p.reduce((a, pi) => a + xlogy(pi, pi), 0);
}

/** CE(p,q) = −Σ pᵢ·log qᵢ — cross-entropy (nats). */
export function crossEntropy(p: number[], q: number[]): number {
  return -p.reduce((a, pi, i) => a + xlogy(pi, q[i]), 0);
}

/** KL(p‖q) = Σ pᵢ·log(pᵢ/qᵢ) — KL divergence (nats). */
export function klDivergence(p: number[], q: number[]): number {
  return p.reduce((a, pi, i) => a + xlogy(pi, pi / q[i]), 0);
}

/** Binary distribution with class-1 probability p0 → [p0, 1−p0] (sums to 1 by construction). */
export function dist2(p0: number): [number, number] {
  return [p0, 1 - p0];
}

// ===== MLE core =====

/**
 * Bernoulli sequence log-likelihood: log L(θ) = h·log θ + (n−h)·log(1−θ).
 * The product over n flips collapses to counts (h, n) — the sufficiency of the
 * empirical frequency for the Bernoulli model. The binomial coefficient C(n,h)
 * is omitted: it is a θ-independent constant that cancels in the argmax, and
 * omitting it makes −log L/n EXACTLY CE(empirical, model) — the plan's
 * "negative log likelihood = CE for classification" identity.
 */
export function logLikelihood(theta: number, heads: number, n: number): number {
  return heads * Math.log(theta) + (n - heads) * Math.log(1 - theta);
}

export function likelihood(theta: number, heads: number, n: number): number {
  return Math.exp(logLikelihood(theta, heads, n));
}

/** The MLE of a Bernoulli bias is the empirical frequency heads/n (see derivations: d/dθ log L = 0). */
export function mleOf(heads: number, n: number): number {
  return heads / n;
}

/** Sweep grid for the CE facet: 0.05 → target at 0.05 steps (last value = target when on-grid). */
export function q0Grid(target: number): number[] {
  const g: number[] = [];
  for (let q = Q0_MIN; q <= target + 1e-9; q += Q0_STEP) g.push(q);
  return g;
}

/** Fixed 21-point θ grid covering the plan's sweep 0.02 → 0.98: 0.02, 0.05…0.95, 0.98. */
export const THETA_GRID: number[] = (() => {
  const g = [THETA_MIN];
  for (let t = 0.05; t <= 0.95 + 1e-9; t += 0.05) g.push(t);
  g.push(THETA_MAX);
  return g;
})();

const fmt4 = (v: number): string => (Number.isFinite(v) ? v.toFixed(4) : '∞');

/** Dense t-grid on [0.05, 0.95] for the static curves of the CE facet. */
function denseGrid(): number[] {
  const g: number[] = [];
  for (let t = Q0_MIN; t <= Q0_MAX + 1e-9; t += 0.01) g.push(t);
  return g;
}

/** Relative (max-normalized) Bernoulli likelihood — peak = 1 at θ̂. Stable at the θ̂ ∈ {0,1} boundary. */
function relLike(t: number, heads: number, n: number, thetaMle: number): number {
  return Math.pow(t / thetaMle, heads) * Math.pow((1 - t) / (1 - thetaMle), n - heads);
}

// ===== Facet A: cross-entropy snapshots =====

function ceSnapshotAt(p: Params, q0: number, first: boolean): SimState {
  const p0 = (p.p0 as number) ?? 0.7;
  const q = dist2(q0);
  const dist = dist2(p0);
  const hP = entropy(dist);
  const kl = klDivergence(dist, q);
  const ce = crossEntropy(dist, q);

  // Static curves over the q0 axis: the CE loss surface (blue), the −log penalty
  // for class 0 (red — the "confident wrong" spike as q0 → 0) and the entropy
  // floor (green). The moving markers show the current (q0, CE), (q0, H) — the
  // vertical gap between them IS the KL — and (q0, −log q0).
  const ts = denseGrid();
  const visuals: VisualCommand[] = [
    { type: 'line', id: 'ce-surface', points: ts.map((t) => [t, crossEntropy(dist, dist2(t))] as [number, number]), color: '#3b82f6' },
    { type: 'line', id: 'penalty-curve', points: ts.map((t) => [t, -Math.log(t)] as [number, number]), color: '#ef4444' },
    { type: 'line', id: 'h-floor', points: [[Q0_MIN, hP], [Q0_MAX, hP]] as [number, number][], color: '#22c55e' },
    { type: 'point', id: 'ce-now', x: q0, y: ce, color: '#f59e0b' },
    { type: 'point', id: 'h-now', x: q0, y: hP, color: '#22c55e' },
    { type: 'point', id: 'pen-now', x: q0, y: -Math.log(q0), color: '#ef4444' },
  ];

  const math: MathStep[] = [
    { latex: `H(p) = -\\sum_i p_i \\log p_i = ${fmt4(hP)}`, id: 'entropy' },
    { latex: `KL(p\\|q) = \\sum_i p_i \\log \\frac{p_i}{q_i} = ${fmt4(kl)}`, id: 'kl-divergence' },
    { latex: `CE(p,q) = -\\sum_i p_i \\log q_i = H(p) + KL(p\\|q) = ${fmt4(ce)}`, id: 'cross-entropy' },
  ];

  return {
    algorithm: { facet: 'cross-entropy', p0, q0 } as Record<string, ParamValue>,
    visuals,
    math,
    narration: `p = [${p0.toFixed(2)}, ${(1 - p0).toFixed(2)}], q = [${q0.toFixed(2)}, ${(1 - q0).toFixed(2)}] — H(p) = ${fmt4(hP)}, KL(p‖q) = ${fmt4(kl)}, CE(p,q) = ${fmt4(ce)}. CE = H + KL; as q0 approaches p0 = ${p0.toFixed(2)}, KL → 0 and CE → H(p).`,
    explanation: {
      changed: first ? [] : [`q0 → ${q0.toFixed(2)}`, `CE = ${fmt4(ce)}`],
      why: first
        ? 'The reference distribution p is fixed by p0; the sweep animates the predicted q0 from 0.05 up to the slider value, moving q = [q0, 1−q0] along the CE loss surface'
        : `q0 = ${q0.toFixed(2)}: CE(p,q) = H(p) + KL(p‖q) = ${fmt4(hP)} + ${fmt4(kl)} — the −log penalty for the class the truth favors is ${fmt4(-Math.log(q0))}`,
      formulaRef: 'cross-entropy',
      dependsOn: ['probability', 'entropy'],
      gateConcepts: ['cross-entropy', 'KL divergence', 'entropy'],
    },
    highlights: [],
    metrics: { p0, q0, hP, klPQ: kl, cePQ: ce },
    events: [{ type: 'fit', label: 'cross-entropy-sweep', step: 0 }],
    timeline: first ? ['Distributions', 'Evaluate'] : ['Evaluate'],
  };
}

// ===== Facet B: MLE snapshots =====

function mleSnapshotAt(p: Params, theta: number, first: boolean, snapIdx: number, mleGridIdx: number): SimState {
  const n = (p.nFlips as number) ?? 20;
  const heads = (p.heads as number) ?? 12;
  const thetaMle = mleOf(heads, n);
  const logL = logLikelihood(theta, heads, n);
  const L = likelihood(theta, heads, n);
  const pHat = heads / n;
  const ce = crossEntropy([pHat, 1 - pHat], [theta, 1 - theta]); // per-sample NLL = −log L/n
  const hP = entropy([pHat, 1 - pHat]);
  const kl = ce - hP; // CE = H + KL

  // Relative likelihood curve over the full θ grid (peak 1 at θ̂) + the MLE line.
  const curve = THETA_GRID.map((t) => [t, relLike(t, heads, n, thetaMle)] as [number, number]);
  const isMle = snapIdx === mleGridIdx;
  const visuals: VisualCommand[] = [
    { type: 'line', id: 'rel-like', points: curve, color: '#3b82f6' },
    { type: 'line', id: 'mle-line', points: [[thetaMle, 0], [thetaMle, 1]] as [number, number][], color: '#22c55e' },
    { type: 'point', id: 'like-now', x: theta, y: relLike(theta, heads, n, thetaMle), color: '#f59e0b' },
    { type: 'point', id: 'mle-now', x: thetaMle, y: 1, color: '#22c55e' },
  ];

  const math: MathStep[] = [
    { latex: `L(\\theta) = \\theta^{${heads}}(1-\\theta)^{${n - heads}}`, id: 'bernoulli-likelihood' },
    { latex: `\\log L(\\theta) = ${heads}\\log\\theta + ${n - heads}\\log(1-\\theta) = ${fmt4(logL)}`, id: 'log-likelihood' },
    { latex: `-\\frac{1}{n}\\log L(\\theta) = CE(\\hat p, q_\\theta) = ${fmt4(ce)}`, id: 'nll-is-ce' },
    { latex: `\\hat\\theta = \\frac{h}{n} = \\frac{${heads}}{${n}} = ${thetaMle.toFixed(3)}`, id: 'mle-argmax' },
  ];

  const events: SimState['events'] = [{ type: 'fit', label: 'mle-sweep', step: 0 }];
  if (isMle) events.push({ type: 'max-likelihood', label: 'mle-at-argmax', step: snapIdx });

  return {
    algorithm: { facet: 'mle', theta, thetaMle } as Record<string, ParamValue>,
    visuals,
    math,
    narration: `θ = ${theta.toFixed(2)}: L(θ) = ${L.toExponential(2)}, log L = ${fmt4(logL)}, per-sample NLL = CE = ${fmt4(ce)}. MLE θ̂ = h/n = ${thetaMle.toFixed(3)}${isMle ? ' — the argmax snapshot (likelihood maximized ⟺ CE minimized)' : ''}. Only the counts (${heads} heads in ${n} flips) matter: the binomial coefficient is a θ-independent constant.`,
    explanation: {
      changed: first ? [] : [`θ → ${theta.toFixed(2)}`, `CE = ${fmt4(ce)}`],
      why: first
        ? 'The Bernoulli likelihood L(θ) = θ^h(1−θ)^(n−h) is the probability of the observed flip sequence as a function of the bias θ; its argmax is the MLE θ̂ = h/n'
        : `θ = ${theta.toFixed(2)}: relative likelihood L(θ)/L(θ̂) = ${relLike(theta, heads, n, thetaMle).toFixed(3)}, per-sample NLL = CE(p̂, q_θ) = ${fmt4(ce)}`,
      formulaRef: 'bernoulli-likelihood',
      dependsOn: ['probability', 'mle'],
      gateConcepts: ['MLE', 'cross-entropy', 'log-likelihood'],
    },
    highlights: [],
    metrics: { theta, thetaMle, heads, nFlips: n, likelihood: L, logLike: logL, cePQ: ce, hP, klPQ: kl },
    events,
    timeline: first ? ['Coin data', 'Evaluate'] : ['Evaluate'],
  };
}

// ===== Simulation (sweep per facet) =====

export const simulation = {
  initialState: (p: Params): SimState => {
    const facet = (p.facet as string) ?? 'cross-entropy';
    if (facet === 'mle') {
      const n = (p.nFlips as number) ?? 20;
      const heads = (p.heads as number) ?? 12;
      const mleGridIdx = mleGridIndex(mleOf(heads, n));
      return mleSnapshotAt(p, THETA_GRID[0], true, 0, mleGridIdx);
    }
    return ceSnapshotAt(p, Q0_MIN, true);
  },

  step: (p: Params, s: SimState): SimState | null => {
    if (s.algorithm.facet === 'mle') {
      const cur = (s.algorithm.theta as number) ?? THETA_GRID[0];
      const i = THETA_GRID.indexOf(cur);
      if (i < 0 || i + 1 >= THETA_GRID.length) return null; // sweep complete
      const n = (p.nFlips as number) ?? 20;
      const heads = (p.heads as number) ?? 12;
      return mleSnapshotAt(p, THETA_GRID[i + 1], false, i + 1, mleGridIndex(mleOf(heads, n)));
    }
    const cur = (s.algorithm.q0 as number) ?? Q0_MIN;
    const target = (p.q0 as number) ?? 0.3;
    const next = cur + Q0_STEP;
    if (next > target + 1e-9) return null; // sweep complete (last snapshot exactly the slider)
    return ceSnapshotAt(p, next, false);
  },
};

/** Grid index nearest θ̂ — the argmax snapshot (exact when θ̂ is on-grid, e.g. θ̂ = 0.6). */
function mleGridIndex(thetaMle: number): number {
  return THETA_GRID.reduce(
    (best, t, i) => (Math.abs(t - thetaMle) < Math.abs(THETA_GRID[best] - thetaMle) ? i : best),
    0
  );
}

export const ceModule: TopicModule = {
  id: 'cross-entropy-loss',
  title: 'Cross-Entropy Loss & Maximum Likelihood',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 4, mathematical: 4, coding: 2, visualization: 3, gateFrequency: 4 },
    estimatedHours: 5,
    revisionPriority: 'P1',
    examFrequency: 'Frequent',
    prerequisites: ['probability', 'calculus'],
    relatedTopics: ['logistic-regression', 'softmax-regression', 'decision-trees', 'gradient-descent'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Geometry: CE Loss Surface / Relative Likelihood' },
      { slot: 'primary', component: 'loss-curve', title: 'Cross-Entropy vs Swept Parameter (per-sample loss)' },
      { slot: 'sidebar', component: 'explain-step', title: 'Step Explanation' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivation: MLE → NLL → CE; CE = H + KL' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Sweep Stages' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'facet', label: 'Facet', type: 'select', options: [
      { value: 'cross-entropy', label: 'Cross-Entropy (Facet A)' },
      { value: 'mle', label: 'MLE (Facet B)' },
    ], default: 'cross-entropy' },
    { id: 'p0', label: 'True class-1 prob p₀', type: 'number', min: 0.05, max: 0.95, step: 0.05, default: 0.7 },
    { id: 'q0', label: 'Predicted class-1 prob q₀ (swept)', type: 'number', min: 0.05, max: 0.95, step: 0.05, default: 0.3 },
    { id: 'nFlips', label: 'Coin flips (n)', type: 'number', min: 1, max: 100, step: 1, default: 20 },
    { id: 'heads', label: 'Heads count (h)', type: 'number', min: 0, max: 100, step: 1, default: 12 },
  ],
  simulation,
  formulas: ceFormulas,
  derivations: ceDerivations,
  questions: ceQuestions,
  comparisons: ceComparisons,
  failureDemos: ceFailureDemos,
  mistakes: ceMistakes,
  testCases: ceTestCases,
  lossMetricKey: 'cePQ',
  lossMetricKey2: 'hP',

  validateParams: (p) => {
    const issues: string[] = [];
    const facet = (p.facet as string) ?? 'cross-entropy';
    if (!isFacet(facet)) issues.push(`facet must be 'cross-entropy' or 'mle', got '${facet}'`);
    const p0 = p.p0 as number | undefined;
    const q0 = p.q0 as number | undefined;
    if (p0 !== undefined) {
      if (!Number.isFinite(p0)) issues.push('p0 must be a finite number');
      else if (p0 <= 0 || p0 > 1) issues.push(`probabilities must lie in (0, 1]: p0 = ${p0} is out of range`);
    }
    if (q0 !== undefined) {
      if (!Number.isFinite(q0)) issues.push('q0 must be a finite number');
      else if (q0 <= 0 || q0 >= 1) issues.push(`q0 = ${q0} must lie in (0, 1): a degenerate q makes CE infinite (log(0) → −∞)`);
      else if (q0 < Q0_MIN || q0 > Q0_MAX) {
        issues.push(`q0 must be on the sweep grid [${Q0_MIN}, ${Q0_MAX}] so the run ends exactly on the slider value`);
      }
    }
    if (facet === 'mle') {
      const n = p.nFlips as number | undefined;
      const heads = p.heads as number | undefined;
      if (n !== undefined && (!Number.isInteger(n) || n < 1)) issues.push('nFlips must be an integer ≥ 1');
      if (heads !== undefined) {
        if (!Number.isInteger(heads) || heads < 0) issues.push('heads must be a non-negative integer');
        else if (n !== undefined && heads > n) issues.push(`heads = ${heads} cannot exceed nFlips = ${n}`);
      }
    }
    return issues;
  },
};

export function register() {
  registerTopic(ceModule);
}
