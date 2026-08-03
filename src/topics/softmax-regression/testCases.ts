// src/topics/softmax-regression/testCases.ts
import type { TestCase } from '../../engine/types';

// Run-based cases consumed by both the topic runner (testCases.test.ts) and the
// centralized runner (src/test/runTestCases.test.ts). The pure-function
// prescribed checks — (1) Σ softmax(z) = 1, (2) shift invariance, (4) numeric
// gradient check — need direct calls to module functions, so they live as
// explicit it() blocks in testCases.test.ts (MLR/ridge precedent).
//
// Determinism: data is seeded (mulberry32 on `seed`), GD is deterministic, so
// identical params → identical runs. Tolerance notes per case.
export const softmaxTestCases: TestCase[] = [
  {
    // Plan case 3: 3-class classification converges — accuracy → 1 on separable
    // 3-cluster data (triangle side 2·margin = 6 vs cluster σ = 0.5 → cleanly
    // separable). Zero-init GD at η = 0.1 on standardized features reaches
    // perfect train accuracy; misclassCount is the per-epoch count metric.
    name: '3-class separable data trains to 100% accuracy',
    params: { nPerClass: 20, margin: 3, learningRate: 0.1, epochs: 300, seed: 42 },
    maxSteps: 400,
    expect: {
      converged: true, // terminates at the epoch budget, not a failure
      finalMetrics: {
        accuracy: (v: number) => v === 1,
        misclassCount: (v: number) => v === 0,
        ce: (v: number) => v < 0.1, // CE ≈ 0 after separation (uniform baseline is ln 3 ≈ 1.099)
      },
    },
  },
  {
    // Plan CE-metric side: categorical CE falls well below the uniform baseline
    // ln(3) ≈ 1.099 once training separates the clusters. (The trajectory
    // monotonicity is asserted on the run in testCases.test.ts.)
    name: 'categorical CE drops far below the uniform baseline ln 3',
    params: { nPerClass: 20, margin: 3, learningRate: 0.1, epochs: 300, seed: 7 },
    maxSteps: 400,
    expect: {
      converged: true,
      finalMetrics: {
        ce: (v: number) => v < 0.5,
        accuracy: (v: number) => v > 0.99,
      },
    },
  },
  {
    // Honest failure-adjacent case: with margin = 1 the three Gaussian clusters
    // overlap heavily (σ = 0.5 ≈ inter-center distance/2), the classes are NOT
    // separable, so a linear softmax model cannot reach accuracy 1 — the
    // decision boundary necessarily misclassifies overlap points.
    name: 'overlapping low-margin clusters cannot reach perfect accuracy',
    params: { nPerClass: 20, margin: 1, learningRate: 0.1, epochs: 300, seed: 42 },
    maxSteps: 400,
    expect: {
      converged: true,
      finalMetrics: {
        accuracy: (v: number) => v < 1,
      },
    },
  },
  {
    // Extra (cheap, deterministic): high learning rate η = 0.5 still converges
    // on separable data — standardized features keep the gradient scale ~O(1),
    // so full-batch GD stays in the stable regime (no divergence, no NaN).
    name: 'high learning rate η = 0.5 trains stably on separable data',
    params: { nPerClass: 20, margin: 3, learningRate: 0.5, epochs: 300, seed: 42 },
    maxSteps: 400,
    expect: {
      converged: true,
      finalMetrics: {
        accuracy: (v: number) => v === 1,
      },
    },
  },
  {
    // Extra (cheap): with only 10 epochs the model still beats the 1/3 random
    // guessing baseline — early training already moves probability mass toward
    // the correct classes (CE < ln 3).
    name: 'a few epochs already beat the 1/3 random-guessing baseline',
    params: { nPerClass: 20, margin: 3, learningRate: 0.1, epochs: 10, seed: 42 },
    maxSteps: 50,
    expect: {
      converged: true,
      finalMetrics: {
        accuracy: (v: number) => v > 1 / 3,
        ce: (v: number) => v < Math.log(3), // ln 3 = uniform-guess CE
      },
    },
  },
];
