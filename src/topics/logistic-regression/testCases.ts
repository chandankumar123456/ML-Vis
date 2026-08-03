// src/topics/logistic-regression/testCases.ts
import type { TestCase } from '../../engine/types';

// Simulation model: one snapshot per epoch (epoch 0 = init, then one full-batch
// GD epoch per step) → snapshots = epochs + 1. maxSteps must EXCEED the snapshot
// count (a run that reaches exactly maxSteps is flagged as a step-budget failure
// even if it terminated cleanly).
//
// Case-level interpretation of the plan's 4 prescribed tests:
//  - Case 2 (sigmoid maps to (0,1)) is exercised here at the RUN level
//    (well-separated data → trained probabilities saturate to 0/1 → CE → 0);
//    the pure unit assertions σ(0)=0.5, σ(10)≈1, σ(−10)≈0 live in testCases.test.ts.
//  - Case 3's "monotone non-increasing over epochs" is a TRAJECTORY property —
//    asserted snapshot-by-snapshot in testCases.test.ts; the run-level check here
//    asserts the endpoint (CE well below the ln 2 start).
//  - Case 4's weak check "p > 0.5 ⟺ prediction" is asserted per-point in
//    testCases.test.ts; the run-level check here asserts the strong calibration
//    identity mean(ŷ) ≈ fraction of positives (exact at the CE optimum).
export const logisticTestCases: TestCase[] = [
  {
    // Plan case 1: linearly separable clusters (margin 2.5, σ 0.5 → centres ~10σ
    // apart → the drawn sample is separable) → GD reaches 100% train accuracy.
    name: 'decision boundary separates linearly separable data (accuracy = 1)',
    params: { nPerClass: 20, margin: 2.5, noise: 0.5, lr: 0.5, epochs: 300, init: 'zero', seed: 42 },
    maxSteps: 320,
    expect: {
      finalMetrics: {
        accuracy: (v: number) => v === 1,
        ce: (v: number) => v < 0.03, // average misconfidence < 3% — sigmoid saturated
      },
    },
  },
  {
    // Plan case 2 (run level): well-separated data → sigmoid probabilities
    // saturate to the extremes → cross-entropy → 0 and 100% accuracy.
    name: 'well-separated data saturates sigmoid probabilities to 0/1 (CE → 0)',
    params: { nPerClass: 25, margin: 4, noise: 0.3, lr: 0.5, epochs: 500, init: 'random', seed: 42 },
    maxSteps: 520,
    expect: {
      finalMetrics: {
        ce: (v: number) => v < 0.02, // average ŷ > 0.996 on the correct class
        accuracy: (v: number) => v === 1,
      },
    },
  },
  {
    // Plan case 3 (run level): on overlapping-but-well-conditioned data CE starts
    // near ln 2 ≈ 0.693 and converges well below it. The monotone trajectory is
    // asserted per-epoch in testCases.test.ts.
    name: 'cross-entropy loss decreases over epochs (final CE well below ln 2)',
    params: { nPerClass: 30, margin: 2, noise: 1, lr: 0.3, epochs: 300, init: 'zero', seed: 42 },
    maxSteps: 320,
    expect: {
      finalMetrics: {
        ce: (v: number) => v < 0.55,
        accuracy: (v: number) => v > 0.8,
      },
    },
  },
  {
    // Plan case 4 (run level): probability calibration — at the CE optimum the
    // bias gradient vanishes ⟹ Σŷ = Σy, so mean predicted probability equals the
    // empirical positive rate (0.5 for the balanced draw).
    name: 'probability calibration: mean predicted p ≈ empirical positive rate',
    params: { nPerClass: 30, margin: 2, noise: 1, lr: 0.3, epochs: 300, init: 'random', seed: 42 },
    maxSteps: 320,
    expect: {
      finalMetrics: {
        meanP: (v: number) => Math.abs(v - 0.5) < 0.05,
        posFrac: (v: number) => Math.abs(v - 0.5) < 1e-9, // balanced data
      },
    },
  },
  {
    // Extra (cheap, useful): CE is convex in w, so the initialization must not
    // change the optimum — zero and random init converge to the same low-loss
    // solution (the pair of cases cross-checks each other).
    name: 'zero init converges to the CE optimum (convexity: init-invariant)',
    params: { nPerClass: 25, margin: 2, noise: 1, lr: 0.3, epochs: 300, init: 'zero', seed: 7 },
    maxSteps: 320,
    expect: { finalMetrics: { ce: (v: number) => v < 0.25 } },
  },
  {
    name: 'random init converges to the CE optimum (convexity: init-invariant)',
    params: { nPerClass: 25, margin: 2, noise: 1, lr: 0.3, epochs: 300, init: 'random', seed: 7 },
    maxSteps: 320,
    expect: { finalMetrics: { ce: (v: number) => v < 0.25 } },
  },
];
