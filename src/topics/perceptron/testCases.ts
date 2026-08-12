// src/topics/perceptron/testCases.ts
import type { TestCase, ParamValue } from '../../engine/types';

// Step model: one snapshot per ONLINE update (snapshot 0 = the init state, then
// one mistake-driven update per step). The trajectory is precomputed in the
// module's buildPlan (memoized per params key) and bounded by MAX_UPDATES; runs
// on non-separable data terminate via an honest throw at the OSCILLATION_CAP
// (snapshot cap under the plan's ~200 bound) or at an exact weight-state cycle
// when one fires — long before the engine budget in either case.
//
// Case-level interpretation of the plan's 4 prescribed tests:
//  - Case 1 (converges within bounded iterations): asserted at run level here
//    (final accuracy 1, mistakesPerEpoch 0, updates under a generous explicit
//    bound) and against the (R·‖w*‖/γ)² theorem bound in testCases.test.ts.
//  - Case 2 (oscillation on non-separable data): run level asserts converged:
//    false + the honest telemetry (failedAtStep 181, reason mentions
//    oscillation on seed 42); the oscillation semantics are unit-tested in
//    testCases.test.ts (measured: OSCILLATION_CAP = 180 fires — snapshots 181,
//    one Oscillate stage on the final snapshot, mistakesPerEpoch 4 at the end).
//  - Case 3 (single update rule): the exact numeric hand example lives in
//    testCases.test.ts; the run-level case here exercises the rule end-to-end.
//  - Case 4 (weight norm growth): the per-update bound ‖Δw‖ ≤ η·√(R²+1) and
//    the O(R)-scale final norm are asserted snapshot-by-snapshot in
//    testCases.test.ts; the run-level case pins the endpoint metric.
export const perceptronTestCases: TestCase[] = [
  {
    // Plan case 1: linearly separable clusters (margin 1.2, σ 0.5, seed 42) →
    // zero train error within a GENEROUS explicit bound on the default seed.
    // Measured on the default seed: 4 updates — far below the theorem bound
    // (R·‖w*‖/γ)² ≈ 16982 for the separator the algorithm actually finds.
    name: 'converges on linearly separable data with zero train error within a bounded iteration count',
    params: { nPerClass: 20, margin: 1.2, noise: 0.5, eta: 1, init: 'zero', seed: 42, separable: true },
    maxSteps: 300,
    expect: {
      converged: true,
      finalMetrics: {
        accuracy: (v: number) => v === 1,
        mistakesPerEpoch: (v: number) => v === 0,
        // generous explicit bound on the default seed (measured: 4 updates)
        updates: (v: number) => v > 0 && v < 500,
      },
    },
  },
  {
    // Plan case 2: overlapping clusters (separable = false) → the perceptron
    // NEVER converges. On the default seed no exact weight-state cycle fires
    // within a runnable bound (measured: none within 5000 updates — the float
    // state drifts quasi-periodically), so the deterministic OSCILLATION_CAP
    // (180) fires first: step() throws with an honest oscillation reason →
    // computeRun records failedAtStep + the message (measured: 181 snapshots).
    // The sandbox does not hang: snapshots are capped well under ~200.
    name: 'oscillates on non-separable data (honest telemetry: oscillation cap, never converges)',
    params: { nPerClass: 20, margin: 1.2, noise: 0.5, eta: 1, init: 'zero', seed: 42, separable: false },
    maxSteps: 600,
    expect: {
      converged: false,
      finalMetrics: {
        accuracy: (v: number) => v < 1,         // never reaches perfect separation (measured 0.725)
        mistakesPerEpoch: (v: number) => v > 0, // mistakes keep firing forever (measured 4)
      },
    },
  },
  {
    // Plan case 3 (run level): zero init → the first update fires on point 0
    // (every score is 0 with zero weights) and each step applies exactly ONE
    // update, so the mistake counter advances by 1 per snapshot. The exact
    // numeric hand example is asserted in testCases.test.ts.
    name: 'zero init: first update fires on point 0; the mistake counter advances by exactly 1 per step',
    params: { nPerClass: 20, margin: 2.5, noise: 0.4, eta: 1, init: 'zero', seed: 42, separable: true },
    maxSteps: 200,
    expect: {
      converged: true,
      finalMetrics: {
        mistakes: (v: number) => v >= 1, // ≥ 1 mistake-driven update happened
        normW: (v: number) => v > 0,
      },
      finalAlgorithm: {
        w1: (v: ParamValue) => typeof v === 'number' && Number.isFinite(v),
      },
    },
  },
  {
    // Plan case 4 (run level): weight norm stays bounded at convergence on the
    // default seed (‖w‖ = 2.741 < 2R ≈ 4.489, measured) — the per-update growth
    // bound ‖Δw‖ ≤ η·√(R²+1) is asserted snapshot-by-snapshot in
    // testCases.test.ts (measured max ‖Δw‖ = 1.958 ≤ 2.457).
    name: 'weight norm stays bounded at convergence (‖w‖ within O(R) of the data radius)',
    params: { nPerClass: 20, margin: 1.2, noise: 0.5, eta: 1, init: 'zero', seed: 42, separable: true },
    maxSteps: 300,
    expect: {
      converged: true,
      finalMetrics: {
        normW: (v: number) => v > 0 && v < 5, // R ≈ 2.245, final ‖w‖ ≈ 2.741 — bounded
      },
    },
  },
  {
    // Extra (cheap, high-value): classic fixed-increment convergence is η-
    // INDEPENDENT — the mistake sequence is scale-invariant, so the update
    // count is identical for every η > 0 and the final weights scale by η
    // (measured on the default seed: 4 updates for both η = 1 and η = 0.5;
    // final weights scale exactly ×2 — asserted in testCases.test.ts).
    name: 'learning rate η does not change the update count (classic rule is η-invariant)',
    params: { nPerClass: 20, margin: 1.2, noise: 0.5, eta: 0.5, init: 'zero', seed: 42, separable: true },
    maxSteps: 300,
    expect: {
      converged: true,
      finalMetrics: { accuracy: (v: number) => v === 1 },
    },
  },
];