// src/topics/multiple-linear-regression/testCases.ts
import type { TestCase } from '../../engine/types';

// Truth model: y = 3·x₁ − 2·x₂ + 1.5·x₃ + 1 (per feature count), features stratified in [−5,5].
export const mlrTestCases: TestCase[] = [
  {
    // Plan spec case 1: normal equation on 2 features recovers truth with tiny noise.
    name: 'normal equation on 2 features recovers true coefficients',
    params: { n: 20, nFeatures: 2, noise: 0.05, useNormalEquation: true, seed: 42 },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        w1: (v: number) => Math.abs(v - 3) < 0.05,
        w2: (v: number) => Math.abs(v - (-2)) < 0.05,
        b: (v: number) => Math.abs(v - 1) < 0.05,
        mse: (v: number) => v < 0.02, // near-zero residual after exact recovery
      },
    },
  },
  {
    // Plan spec case 2: GD (feature-standardized, lr 0.01, 2000 epochs) ≈ normal equation
    // optimum on a NON-degenerate target (noise 0.1, same seed → both methods fit the
    // identical data, so agreement is exact even though the optimum is off-truth).
    name: 'gradient descent converges to the normal-equation optimum',
    params: { n: 30, nFeatures: 2, noise: 0.1, useNormalEquation: false, learningRate: 0.01, epochs: 2000, seed: 42 },
    maxSteps: 2001,
    expect: {
      finalMetrics: {
        w1: (v: number) => Math.abs(v - 3) < 0.05,
        w2: (v: number) => Math.abs(v - (-2)) < 0.05,
        b: (v: number) => Math.abs(v - 1) < 0.05,
      },
    },
  },
  {
    // Plan spec case 3: x₂ = 2x₁ exactly → XᵀX singular. validateParams flags it AND the
    // sandboxed run fails cleanly (non-finite θ from singular solve → failedAtStep defined).
    name: 'fails cleanly when features are collinear (XᵀX singular)',
    params: { n: 20, nFeatures: 2, noise: 0.0, collinear: true, useNormalEquation: true, seed: 42 },
    maxSteps: 3,
    expect: { converged: false },
  },
  {
    // Plan spec case 4: fixed held-out test point (x₁=1.5, x₂=−0.5) → true y = 3·1.5 − 2·(−0.5) + 1 = 6.5.
    name: 'predicts correctly on a held-out test point',
    params: { n: 20, nFeatures: 2, noise: 0.1, useNormalEquation: true, seed: 7 },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        testPred: (v: number) => Math.abs(v - 6.5) < 0.1,
        testTrue: (v: number) => Math.abs(v - 6.5) < 1e-9,
      },
    },
  },
];
