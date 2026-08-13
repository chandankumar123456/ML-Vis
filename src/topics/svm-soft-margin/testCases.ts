// src/topics/svm-soft-margin/testCases.ts
import type { TestCase } from '../../engine/types';

// Plan-spec cases 1, 2 and a sweep-semantics guard. The hinge-loss and slack
// cases (plan cases 3, 4) are pure/measured and live in testCases.test.ts.
//
// NOTE on the C-sweep simulation: a run emits one snapshot per log-grid value
// below params.C, then EXACTLY params.C last (mirrors ridge's λ-sweep). At
// C = 1000 that is 11 snapshots max (C_GRID has 11 values), so maxSteps = 20 always exceeds the count
// and the engine never flags a spurious step-budget failure. All runs terminate
// cleanly (sweep complete → null), so converged: true.
export const svmSoftTestCases: TestCase[] = [
  {
    // Plan case 1: on separable data a large C ≈ hard margin. The exact
    // comparison against the independent hard-margin reference (margin within
    // ~0.2%, same support vectors) lives in testCases.test.ts; here we assert
    // the soft fit is in the hard-margin regime: finite margin near the
    // cluster-gap value (2 for the default geometry), ~zero slack.
    name: 'C = 1000 on separable data is in the hard-margin regime (near-zero slack)',
    params: { C: 1000, nPerClass: 10, margin: 1.5, spread: 0.5, outlier: false, seed: 42 },
    maxSteps: 20,
    expect: {
      converged: true,
      finalAlgorithm: { C: 1000, mode: 'svm-soft-margin' },
      finalMetrics: {
        margin: (v: number) => v > 1.5 && v < 3,      // ≈ 2 (clusters 2 apart)
        slackSum: (v: number) => v < 0.01,             // essentially zero slack
        freeSupportCount: (v: number) => v >= 2,
        violatedCount: (v: number) => v <= 2,
      },
    },
  },
  {
    // Plan case 2 (single-run side): with the outlier on, a small C ignores it —
    // wide margin, at least the outlier violated (and misclassified). The
    // two-run "small C vs large C" margin comparison also lives in the .test.ts.
    name: 'C = 0.01 with an outlier ignores it: wide margin, outlier misclassified',
    params: { C: 0.01, nPerClass: 10, margin: 1.5, spread: 0.5, outlier: true, outlierStrength: 3, seed: 42 },
    maxSteps: 20,
    expect: {
      converged: true,
      finalMetrics: {
        margin: (v: number) => v > 3,                  // much wider than hard margin
        violatedCount: (v: number) => v >= 1,
        misclassifiedCount: (v: number) => v >= 1,     // the outlier is on the wrong side
      },
    },
  },
  {
    // Sweep semantics: last snapshot is exactly the slider C (here 50, off the
    // log grid) and the run ends cleanly.
    name: 'C-sweep: last snapshot equals the slider C and terminates cleanly',
    params: { C: 50, nPerClass: 10, margin: 1.5, spread: 0.5, outlier: false, seed: 42 },
    maxSteps: 20,
    expect: {
      converged: true,
      finalAlgorithm: { C: 50, mode: 'svm-soft-margin' },
    },
  },
];
