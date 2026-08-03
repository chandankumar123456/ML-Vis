// src/topics/cross-entropy-loss/testCases.ts
import type { TestCase } from '../../engine/types';

// Hand-verified constants (natural log — all quantities are in NATS):
// H([0.7, 0.3]) = −(0.7·ln 0.7 + 0.3·ln 0.3) = 0.610864302…
//   ln 0.7 = −0.35667494, ×0.7 = −0.24967246; ln 0.3 = −1.20397280, ×0.3 = −0.36119184.
// KL([0.7,0.3] ‖ [0.35,0.65]) = 0.7·ln 2 + 0.3·ln(0.3/0.65) = 0.48520303 − 0.23195697 = 0.25324606…
// CE = H + KL = 0.61086430 + 0.25324606 = 0.86411036…
// CE([0.8,0.2] ‖ [0.3,0.7]) = −(0.8·ln 0.3 + 0.2·ln 0.7) = 0.96317824 + 0.07133499 = 1.03451323…
export const H_07 = 0.6108643020548935;

export const ceTestCases: TestCase[] = [
  {
    // Plan spec case 1. Honest framing (per the drift warning): CE(p,p) = H(p), which
    // is ≥ 0 and only 0 for a degenerate distribution. The informative claims are
    // CE(p,p) = H(p) EXACTLY and KL(p‖p) = 0. The last snapshot of the q0 sweep is
    // exactly the slider value q0 = 0.7 = p0, so q = p there (sweep ends ON-STEP).
    name: 'cross-entropy equals entropy when q = p (KL is zero); CE ≥ 0',
    params: { facet: 'cross-entropy', p0: 0.7, q0: 0.7 },
    maxSteps: 20, // 0.05 → 0.70 at 0.05 steps = 14 snapshots
    expect: {
      finalMetrics: {
        cePQ: (v: number) => Math.abs(v - H_07) < 1e-9, // CE(p,p) = H(p)
        hP: (v: number) => Math.abs(v - H_07) < 1e-9,
        klPQ: (v: number) => Math.abs(v) < 1e-12, // KL(p‖p) = 0 exactly
      },
    },
  },
  {
    // Plan spec case 2: CE asymmetric. Last snapshot q0 = 0.3, p0 = 0.8.
    // CE(p,q) = 1.03451323… (hand-verified); the swapped CE(q,p) = 1.19354960…
    // differs by 0.159 — the asymmetry margin (> 0.1) is asserted in testCases.test.ts
    // with an independent local CE implementation.
    name: 'cross-entropy is asymmetric: CE(p,q) ≠ CE(q,p) for p ≠ q',
    params: { facet: 'cross-entropy', p0: 0.8, q0: 0.3 },
    maxSteps: 10, // 0.05 → 0.30 = 6 snapshots
    expect: {
      finalMetrics: {
        cePQ: (v: number) => Math.abs(v - 1.03451323) < 1e-6,
      },
    },
  },
  {
    // Plan spec case 3: CE = H(p) + KL(p‖q) numeric decomposition (constants above).
    name: 'cross-entropy decomposes as H(p) + KL(p‖q) numerically',
    params: { facet: 'cross-entropy', p0: 0.7, q0: 0.35 },
    maxSteps: 10, // 0.05 → 0.35 = 7 snapshots
    expect: {
      finalMetrics: {
        hP: (v: number) => Math.abs(v - 0.610864302) < 1e-6,
        klPQ: (v: number) => Math.abs(v - 0.253246060) < 1e-6,
        cePQ: (v: number) => Math.abs(v - 0.864110362) < 1e-6,
      },
    },
  },
  {
    // Plan spec case 4: MLE maximizes likelihood ⟺ minimizes CE. The Bernoulli
    // likelihood L(θ) = θ^h(1−θ)^(n−h) is used (no binomial coefficient — a θ-
    // independent constant that cancels in the argmax), so the per-sample NLL
    // −log L/n IS CE(empirical, model) EXACTLY. The argmax-likelihood snapshot and
    // argmin-CE snapshot coincide (asserted in testCases.test.ts); the event is
    // raised on that snapshot. θ̂ = heads/n = 12/20 = 0.6 exactly.
    name: 'MLE maximizes likelihood exactly when cross-entropy is minimized',
    params: { facet: 'mle', nFlips: 20, heads: 12 },
    maxSteps: 25, // θ grid 0.02 → 0.98 = 21 snapshots
    expect: {
      finalMetrics: {
        thetaMle: (v: number) => Math.abs(v - 12 / 20) < 1e-9,
      },
      eventLabels: ['mle-at-argmax'],
    },
  },
  {
    // Extra: CE(p, q) is strictly convex in q0 (d²CE/dq0² = p0/q0² + p1/(1−q0)² > 0)
    // and its unique minimum is at q0 = p0. p0 = 0.7 lies on the sweep grid, so the
    // argmin snapshot is EXACTLY q0 = p0 (asserted in testCases.test.ts).
    name: 'cross-entropy is minimized exactly at q0 = p0 (convexity)',
    params: { facet: 'cross-entropy', p0: 0.7, q0: 0.95 },
    maxSteps: 20, // 0.05 → 0.95 = 19 snapshots
    expect: {
      finalMetrics: {
        cePQ: (v: number) => v > 0,
      },
    },
  },
];
