// src/topics/svm-hard-margin/testCases.ts
import type { TestCase } from '../../engine/types';

// Empirical anchors (all measured by running the simulator — see testCases.test.ts):
//   default (nPerClass 12, margin 1.5, noise 0.45, seed 42): ‖w‖ = 1.567,
//     margin = 2/‖w‖ = 1.276, γ = 0.638, ½‖w‖² = 1.228, svCount = 2 (d9, d21),
//     run = 40 snapshots (SWEEP_CAP), loss descends 1.2409 → 1.2276, trainError 0.
//   scale 2 on the default: ‖w‖ = 0.783 (= 1.567/2), margin = 2.553 (= 2×1.276).
//   seed 7: margin = 1.365, ½‖w‖² = 1.074, svCount = 3 (two class-1 SVs).
//   margin 1.0, noise 0.45, seed 42: margin = 0.372 (tight but separable).
export const svmTestCases: TestCase[] = [
  {
    // Plan case 1: max-margin solution on separable data — objective ½‖w‖² is at
    // its minimum (lossMetricKey, lower-better), margin = 2/‖w‖ ≈ 1.276 holds
    // (asserted exactly in testCases.test.ts), and the run converges cleanly.
    name: 'max-margin solution on separable data (2 support vectors, train error 0)',
    params: { nPerClass: 12, margin: 1.5, noise: 0.45, seed: 42 },
    maxSteps: 500,
    expect: {
      converged: true,
      eventLabels: ['max-margin separator found'],
      finalMetrics: {
        halfWSq: (v: number) => v < 1.24,           // minimum over the sweep (default: 1.228)
        margin: (v: number) => v > 1.25 && v < 1.30, // 2/‖w‖ with ‖w‖ = 1.567 → 1.276
        svCount: (v: number) => v === 2,
        trainError: (v: number) => v === 0,          // hard margin: perfectly separable
      },
    },
  },
  {
    // Plan case 2: support vectors are the closest points — exactly 2 points on
    // the margin at distance 1/‖w‖ (per-point distance asserted in .test.ts).
    name: 'support vectors are the closest points (exactly 2, seed 42)',
    params: { nPerClass: 12, margin: 1.5, noise: 0.45, seed: 42 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        svCount: (v: number) => v === 2,
        gamma: (v: number) => v === 0.6382038210798113, // 1/‖w‖ — the SV distance
      },
    },
  },
  {
    // Plan case 3: scaling invariance — scaled data keeps the SAME boundary while
    // ‖w‖ and the margin rescale by 1/c and c (margin·‖w‖ = 2 invariant; the exact
    // cross-run assertions live in testCases.test.ts). This entry asserts the
    // scale-2 optimum metrics.
    name: 'scaling invariance: data × 2 halves ‖w‖ and doubles the margin',
    params: { nPerClass: 12, margin: 1.5, noise: 0.45, seed: 42, scale: 2 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        normW: (v: number) => v < 0.79 && v > 0.77,  // 1.567/2 = 0.783
        margin: (v: number) => v > 2.5 && v < 2.6,   // 2×1.276 = 2.553
        trainError: (v: number) => v === 0,
      },
    },
  },
  {
    // Plan case 4: classifier correctness — every point classified correctly with
    // geometric margin ≥ γ = 1/‖w‖ (constraint feasibility asserted in .test.ts).
    name: 'classifier correctness: train error 0, every point at distance ≥ 1/‖w‖',
    params: { nPerClass: 12, margin: 1.5, noise: 0.45, seed: 42 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        trainError: (v: number) => v === 0,
        svCount: (v: number) => v >= 2 && v <= 3, // 2D: at most d+1 = 3 support vectors
      },
    },
  },
  {
    // Extra: seed 7 produces the 3-support-vector configuration (two class-1 SVs
    // share one margin line) — the case the geometric solver's same-class-pair
    // candidate family exists for. margin = 1.365, ½‖w‖² = 1.074.
    name: 'seed 7: three support vectors (two from class 1 on one margin line)',
    params: { nPerClass: 12, margin: 1.5, noise: 0.45, seed: 7 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        svCount: (v: number) => v === 3,
        margin: (v: number) => v > 1.35 && v < 1.38,
        halfWSq: (v: number) => v < 1.2,
      },
    },
  },
  {
    // Extra: tight clusters (separation 1.0, σ 0.45) — separable but the margin
    // collapses to 0.372 with 3 support vectors squeezed into the band.
    name: 'tight clusters: tiny margin, objective ½‖w‖² ≈ 14.4',
    params: { nPerClass: 12, margin: 1.0, noise: 0.45, seed: 42 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        margin: (v: number) => v < 0.5,   // 0.372
        halfWSq: (v: number) => v > 10,   // 14.43
      },
    },
  },
];
