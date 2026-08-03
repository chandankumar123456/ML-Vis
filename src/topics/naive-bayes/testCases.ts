// src/topics/naive-bayes/testCases.ts
import type { TestCase } from '../../engine/types';

/**
 * Task 10 (Wave 2) prescribed cases. All runs are deterministic (seeded PRNG
 * for the continuous model; a hard-coded table for the categorical model).
 *
 * Thresholds were hand-verified against the exact math:
 *  - high-correlation independence violation: naive P(C₁|x) ≈ 0.87 vs joint
 *    ≈ 0.003 at ρ=0.9 (argmax FLIPS: naive→1, joint→0); naiveJointDiff ≈ 0.87.
 *  - low-correlation control: naive ≈ joint (diff ≈ 0.002 at n=300, ρ=0).
 *  - categorical smoothing: unseen value likelihood ε = (0+α)/(n+αV) = 1/8 at
 *    α=1 on the crafted table; class likelihoods 3/64 each → posterior ½/½.
 *  - without smoothing: both class likelihoods exactly 0 → degenerate posterior.
 */
export const nbTestCases: TestCase[] = [
  {
    // Plan spec case 1: Σ posterior over classes = 1 (both the naive and the
    // full-covariance "without independence" normalization — logsumexp exact).
    name: 'posterior normalizes to 1 (naive and joint)',
    params: { nClasses: 2, nPerClass: 25, correlation: 0.9, smoothing: 0.1, seed: 42, queryX1: 2.4, queryX2: 2 },
    maxSteps: 30,
    expect: {
      finalMetrics: {
        postSum: (v: number) => Math.abs(v - 1) < 1e-9,
        postJointSum: (v: number) => Math.abs(v - 1) < 1e-9,
        post0: (v: number) => v > 0.05 && v < 0.5,   // measured 0.128
        post1: (v: number) => v > 0.5,               // measured 0.872
      },
    },
  },
  {
    // Plan spec case 2 (high correlation): the independence assumption changes the
    // posterior — the naive model double-counts the shared latent evidence. The
    // posterior ARGMAX flips: naive picks class 1, the true generative (full
    // covariance) picks class 0.
    name: 'independence assumption changes posterior on correlated features',
    params: { nClasses: 2, nPerClass: 25, correlation: 0.9, smoothing: 0.1, seed: 42, queryX1: 2.4, queryX2: 2 },
    maxSteps: 30,
    expect: {
      finalMetrics: {
        naiveJointDiff: (v: number) => v > 0.5,        // measured 0.869
        post1: (v: number) => v > 0.5,                 // naive → class 1 strongly
        postJoint1: (v: number) => v < 0.1,            // joint → class 1 almost impossible
      },
      finalAlgorithm: {
        predClass: (v: number | string | boolean) => v === 1,
        predJointClass: (v: number | string | boolean) => v === 0,
      },
    },
  },
  {
    // Plan spec case 2 control: with INDEPENDENT features (ρ = 0) the naive and
    // full-covariance posteriors agree — the assumption is harmless exactly when
    // it holds. Large n shrinks the finite-sample covariance noise.
    name: 'independence assumption is harmless when features are independent (control)',
    params: { nClasses: 2, nPerClass: 300, correlation: 0, smoothing: 0.1, seed: 42, queryX1: 2.4, queryX2: 2 },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        naiveJointDiff: (v: number) => v < 0.05,       // measured 0.002
      },
    },
  },
  {
    // Plan spec case 3 (with smoothing): the crafted categorical table (values
    // {0,1} for class 0, {2,3} for class 1) makes query (x₁=0, x₂=3) have an
    // UNSEEN value in each class. Laplace α=1 gives the unseen value likelihood
    // ε = (0+1)/(4+1·4) = 1/8 (not 0), class likelihoods 3/64 each, posterior ½/½.
    name: 'Laplace smoothing gives an unseen feature value ε > 0 (not 0)',
    params: { nClasses: 2, nPerClass: 4, correlation: 0.5, smoothing: 1, seed: 42, discrete: true, queryX1: 0, queryX2: 3 },
    maxSteps: 30,
    expect: {
      finalMetrics: {
        eps: (v: number) => Math.abs(v - 1 / 8) < 1e-9,        // smoothed unseen-value likelihood
        lik0: (v: number) => Math.abs(v - 3 / 64) < 1e-9,      // (3/8)·(1/8)
        lik1: (v: number) => Math.abs(v - 3 / 64) < 1e-9,
        post0: (v: number) => Math.abs(v - 0.5) < 1e-9,
        post1: (v: number) => Math.abs(v - 0.5) < 1e-9,
        postSum: (v: number) => Math.abs(v - 1) < 1e-9,
      },
    },
  },
  {
    // Plan spec case 3 (without smoothing): the same unseen value has likelihood
    // exactly 0, so BOTH class posteriors collapse to 0 — the model cannot
    // classify the point (reported as the degenerate posterior and predClass = -1).
    name: 'without smoothing an unseen feature value zeros the posterior',
    params: { nClasses: 2, nPerClass: 4, correlation: 0.5, smoothing: 0, seed: 42, discrete: true, queryX1: 0, queryX2: 3 },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        eps: (v: number) => v === 0,
        lik0: (v: number) => v === 0,
        lik1: (v: number) => v === 0,
        post0: (v: number) => v === 0,
        post1: (v: number) => v === 0,
        postSum: (v: number) => v === 0,               // degenerate normalization
      },
      finalAlgorithm: {
        predClass: (v: number | string | boolean) => v === -1, // no class wins
      },
    },
  },
  {
    // Plan spec case 4: seeded Gaussian clusters → the fitted per-class means
    // recover the truth (0,0) and (3,0), and the posterior argmax matches the
    // true cluster label on (almost) every training point.
    name: 'Gaussian NB fits seeded Gaussians and posterior argmax matches cluster',
    params: { nClasses: 2, nPerClass: 60, correlation: 0.5, smoothing: 0.1, seed: 42, queryX1: 2.4, queryX2: 2 },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        acc: (v: number) => v >= 0.9,                  // measured 0.925
        mu1_0: (v: number) => Math.abs(v - 0) < 0.5,   // truth (0, 0)
        mu2_0: (v: number) => Math.abs(v - 0) < 0.5,
        mu1_1: (v: number) => Math.abs(v - 3) < 0.5,   // truth (3, 0)
        mu2_1: (v: number) => Math.abs(v - 0) < 0.5,
        postSum: (v: number) => Math.abs(v - 1) < 1e-9,
      },
    },
  },
  {
    // Extra (cheap): 3-class posterior normalization + a well-defined argmax.
    name: '3-class posterior normalizes and picks a valid class',
    params: { nClasses: 3, nPerClass: 25, correlation: 0.5, smoothing: 0.1, seed: 42, queryX1: 1.5, queryX2: 1.2 },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        postSum: (v: number) => Math.abs(v - 1) < 1e-9,
      },
      finalAlgorithm: {
        predClass: (v: number | string | boolean) => v === 0 || v === 1 || v === 2,
      },
    },
  },
];
