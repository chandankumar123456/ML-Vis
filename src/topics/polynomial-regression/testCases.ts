// src/topics/polynomial-regression/testCases.ts
import type { TestCase } from '../../engine/types';

// Truth model (see module.ts): y = w₂·u² + w₁·u + w₀ with u = x/3 ∈ [−1,1];
// w₀ = 1, w₁ = 0.5, w₂ = 1. Coefficients are reported in the NORMALIZED basis
// u = x/xMax — a degree-d polynomial in u is exactly a degree-d polynomial in x,
// so "w₂ within 0.05 of truth" tests the honest quadratic-coefficient recovery.
export const polyTestCases: TestCase[] = [
  {
    // Plan spec case 1: a degree-1 line cannot capture the quadratic curvature.
    // Deterministic (seed 42): trainMSE ≈ 0.128, R² ≈ 0.29 — bias-dominated fit.
    name: 'degree 1 on quadratic data underfits (high MSE, linear model)',
    params: { degree: 1, noise: 0.5, nTrain: 30, nTest: 20, seed: 42, fitOn: 'train' },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        trainMse: (v: number) => v > 0.1, // well above the degree-2 fit (~0.069)
        r2: (v: number) => v < 0.5,       // explains under half the variance
      },
    },
  },
  {
    // Plan spec case 2: with tiny noise the degree-2 model recovers the quadratic
    // truth in the u-basis: |w₂ − 1| < 0.05 (also w₁ ≈ 0.5, w₀ ≈ 1).
    name: 'degree 2 recovers the quadratic truth (w₂ within 0.05)',
    params: { degree: 2, noise: 0.05, nTrain: 30, nTest: 20, seed: 42, fitOn: 'train' },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        w0: (v: number) => Math.abs(v - 1) < 0.05,
        w1: (v: number) => Math.abs(v - 0.5) < 0.05,
        w2: (v: number) => Math.abs(v - 1) < 0.05,
        trainMse: (v: number) => v < 0.01, // near noise-free residual
      },
    },
  },
  {
    // Plan spec case 3: degree 15 overfits. trainMSE is tiny (~0.039), but the
    // power-basis coefficients explode (max |w| ≈ 28068 ≫ 10) and the held-out
    // test MSE explodes (~681). Primary signal: coefficient explosion.
    name: 'degree 15 overfits — tiny train MSE, exploding coefficients, huge test MSE',
    params: { degree: 15, noise: 0.5, nTrain: 30, nTest: 20, seed: 42, fitOn: 'train' },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        trainMse: (v: number) => v < 0.05,
        testMse: (v: number) => v > 10,          // held-out split is honest
        maxAbsW: (v: number) => v > 10,          // primary overfit signal
      },
    },
  },
  {
    // Plan spec case 4 (single-run representative): moderate degree keeps BOTH
    // train and test error low — the bias-variance sweet spot for a quadratic
    // truth. The full 5-seed-averaged error curve (U-shape) is asserted in
    // testCases.test.ts via simulation.initialState across degrees and seeds.
    name: 'moderate degree (5) keeps train and test error low (sweet spot)',
    params: { degree: 5, noise: 0.5, nTrain: 30, nTest: 20, seed: 42, fitOn: 'train' },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        trainMse: (v: number) => v < 0.08,
        testMse: (v: number) => v < 0.2,         // generalizes (vs 681 at d=15)
      },
    },
  },
  {
    // Extra cheap case: guards the explosion metric — at low degree the same
    // maxAbsW signal must stay bounded (explosion at d=15 is real, not an
    // artifact of the metric).
    name: 'coefficient magnitudes stay bounded at low degree (no false alarm)',
    params: { degree: 2, noise: 0.5, nTrain: 30, nTest: 20, seed: 42, fitOn: 'train' },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        maxAbsW: (v: number) => v < 2,
      },
    },
  },
];
