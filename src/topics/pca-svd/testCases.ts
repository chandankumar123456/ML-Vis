// src/topics/pca-svd/testCases.ts
// Data-driven test cases for the pca-svd topic. Every numeric anchor below was
// MEASURED by running the module (scratch.test.ts) before being hard-coded —
// nothing is estimated. The hand-written identity tests live in
// testCases.test.ts; this file feeds the centralized runner (computeRun +
// expect.finalMetrics/eventLabels/converged).
import type { TestCase } from '../../engine/types';

// The four plan cases map to test blocks as follows (measured in testCases.test.ts):
//   1. `SVD gives same PCs as eigen-decomposition` — V = eigenvectors of XᵀX:
//      G·v_k = λ_k·v_k exactly + orthonormality + X = UΣVᵀ reconstruction.
//   2. `singular values relate to eigenvalues` — σ_k² = λ_k(XᵀX) = n·λ_k(cov).
//   3. `low-rank approximation` — ‖X−X̂_k‖_F = ‖X−X̂_k‖₂ = σ_{k+1} (Eckart-Young);
//      per-sample MSE = σ_{k+1}²/n.
//   4. `economy SVD shapes` — U (n×k), Σ (k×k), Vᵀ (k×d).
// The data-driven cases below lock the MEASURED anchors + convergence contracts.

const DEFAULT = { n: 40, corr: 0.7, rotDeg: 30, noise: 0.15, seed: 42, rank: 2 };

export const svdTestCases: TestCase[] = [
  {
    // Default config, full rank. The 8-event chain fires in order and the final
    // snapshot reports the measured SVD of the seeded correlated cloud.
    name: 'default run converges: full build-up + rank-2 sweep with exact Eckart-Young errors',
    params: DEFAULT,
    expect: {
      converged: true,
      eventLabels: [
        'svd-seeded-correlated-gaussian',
        'gram-matrix-XᵀX',
        'XᵀX-eigendecomposition',
        'singular-values-extracted',
        'factorization-complete',
        'rank-1-reconstruction',
        'rank-2-reconstruction',
        'svd-rank-sweep-complete',
      ],
      finalMetrics: {
        step: 6,
        rank: 2,
        isOptimal: 1,
        n: 40,
        dataSeed: 42,
        rankDeficient: 0,
        // measured (default, seed 42): σ₁ = 8.804484511776364, σ₂ = 3.3373516320652685
        sigma1: 8.804484511776364,
        sigma2: 3.3373516320652685,
        singularRatio: 2.6381650729227606,
        // λ(XᵀX) = σ²; λ(cov) = σ²/n
        lambda1: 77.51894751810988,
        lambda2: 11.137915916048712,
        covLambda1: 1.9379736879527472,
        covLambda2: 0.2784478979012178,
        // rank-2 = full rank: the reconstruction is EXACT (Eckart-Young → σ₃ = 0)
        reconstructionError: (v: number) => v < 1e-20,
        frobErrK: (v: number) => v < 1e-12,
        spectralErrK: 0,
      },
    },
  },
  {
    // Rank slider = 1: the LAST snapshot is rank 1 exactly (ridge-λ-sweep
    // convention) and the loss equals the dropped energy σ₂²/n = covλ₂.
    name: 'rank-1 run converges: the loss is exactly the dropped energy σ₂²/n',
    params: { ...DEFAULT, rank: 1 },
    expect: {
      converged: true,
      eventLabels: ['rank-1-reconstruction', 'svd-rank-sweep-complete'],
      finalMetrics: {
        step: 5,
        rank: 1,
        isOptimal: 1,
        reconstructionError: 0.278447897901218,   // σ₂²/n = covλ₂ (measured)
        frobErrK: 3.33735163206527,               // = σ₂ exactly (measured)
        spectralErrK: 3.337351632065269,          // = σ₂ exactly (measured)
        sigma2: 3.3373516320652685,
      },
    },
  },
  {
    // A second seed locks the measured anchors on a different draw.
    name: 'seed 7 run: measured anchors on a different draw',
    params: { ...DEFAULT, seed: 7 },
    expect: {
      converged: true,
      finalMetrics: {
        sigma1: 9.460388559546717,
        sigma2: 3.2502580275094646,
        covLambda1: 2.23747379244006,
        covLambda2: 0.26410443113474286,
        reconstructionError: (v: number) => v < 1e-20,
      },
    },
  },
  {
    // n = 60, weaker correlation, stronger noise — the low-rank story still
    // holds: rank-2 reconstruction is exact, measured anchors locked.
    name: 'cfg2 (n 60, corr 0.3, rotDeg 60, noise 0.25, seed 123): measured anchors',
    params: { n: 60, corr: 0.3, rotDeg: 60, noise: 0.25, seed: 123, rank: 2 },
    expect: {
      converged: true,
      finalMetrics: {
        sigma1: 9.26783003261347,
        sigma2: 6.380133507392377,
        covLambda1: 1.4315445585568694,
        covLambda2: 0.6784350595358491,
        n: 60,
        reconstructionError: (v: number) => v < 1e-20,
      },
    },
  },
  {
    // Hand-crafted collinear points: σ₂ = 0, rank deficient — but the run
    // COMPLETES (the deterministic null-space completion keeps the
    // reconstruction exact) and the honest flags are reported. The ratio is
    // SATURATED at RATIO_CAP so no metric is ever non-finite.
    name: 'collinear data: rank-deficient but the run completes with the honest completion',
    params: { ...DEFAULT, points: '[[1,2],[3,4],[5,6],[7,8]]' },
    expect: {
      converged: true,
      eventLabels: ['rank-1-reconstruction', 'rank-2-reconstruction'],
      finalMetrics: {
        sigma2: 0,
        rankDeficient: 1,
        singularRatio: 1e9,                       // saturated (σ₂ = 0)
        reconstructionError: (v: number) => v < 1e-20,
      },
      finalAlgorithm: {
        mode: 'svd-full-rank',
        rank: 2,
        isOptimal: 1,
      },
    },
  },
  {
    // Zero-variance data (all points identical) → the Gram matrix is the zero
    // matrix → eigen2x2Symmetric throws → honest telemetry failure (the pca
    // precedent). No NaN ever escapes the sandbox.
    name: 'zero-variance data: honest telemetry failure (no NaN escapes)',
    params: { ...DEFAULT, points: '[[2,3],[2,3],[2,3]]' },
    expect: {
      converged: false,
    },
  },
  {
    // A MISSING ROW (null element) makes the data matrix ill-formed: the
    // row-to-point mapping throws before any math runs → telemetry failure.
    // (A row of [null,null] would silently coerce to 0 — a JS pitfall we
    // narrate in the failure demo instead of testing.)
    name: 'missing row: honest telemetry failure (SVD needs a complete matrix)',
    params: { ...DEFAULT, points: '[[1,2],[3,4],null,[7,8]]' },
    expect: {
      converged: false,
    },
  },
];