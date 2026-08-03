// src/topics/lasso-regression/testCases.ts
import type { TestCase } from '../../engine/types';

// Truth model: y = wTrue·x + bTrue (per feature count). Features are i.i.d. U[−5,5]
// (seeded) so columns are near-orthogonal — the well-conditioned regime where
// coordinate descent converges fast and the soft-threshold boundaries are clean.
// TRUE_W = [3, −2, 1.5, 0.4, 0.2, 0.1]: the trailing SMALL coefficients are what
// lasso zeroes exactly at moderate λ (standardized: |w̃₄| ≈ 1.16, |w̃₅| ≈ 0.58, |w̃₆| ≈ 0.29).
export const lassoTestCases: TestCase[] = [
  {
    // Plan case 1: at tiny λ the soft-threshold is (essentially) the identity, so
    // coordinate descent converges to the OLS solution on well-conditioned data.
    name: 'lasso at tiny λ recovers the OLS solution (λ ≈ 0)',
    params: { n: 25, nFeatures: 4, noise: 0.5, lambda: 1e-6, seed: 42 },
    maxSteps: 2000,
    expect: {
      converged: true,
      finalMetrics: {
        w1: (v: number) => Math.abs(v - 3) < 0.05,
        w2: (v: number) => Math.abs(v - (-2)) < 0.05,
        w3: (v: number) => Math.abs(v - 1.5) < 0.05,
        w4: (v: number) => Math.abs(v - 0.4) < 0.05,
        b: (v: number) => Math.abs(v - 1) < 0.05,
      },
    },
  },
  {
    // Plan case 2: at λ = 1.5 the normalized correlation of feature 4 (|ρ₄/n| ≈ 1.16)
    // falls below λ, so soft-thresholding sets w₄ to EXACTLY 0 while features 1–3
    // survive (shrunk but nonzero). nNonzero = 3 is the feature-selection count.
    name: 'lasso zeroes a small coefficient exactly at sufficient λ',
    params: { n: 40, nFeatures: 4, noise: 0.5, lambda: 1.5, seed: 42 },
    maxSteps: 2000,
    expect: {
      converged: true,
      finalMetrics: {
        w4: (v: number) => Math.abs(v) < 1e-12, // soft-threshold gives EXACT 0 for |z| ≤ λ
        w1: (v: number) => Math.abs(v) > 2,     // survives (shrunk from 3 but clearly nonzero)
        nNonzero: (v: number) => v === 3,
      },
    },
  },
  {
    // Plan case 3 (lasso half): same data/λ — lasso zeroes w₄. The ridge contrast
    // (ridge keeps |w₄| > 0) is asserted in testCases.test.ts with a self-contained
    // ridge closed-form helper (the ridge topic is written in parallel — no import).
    name: 'lasso zeroes w4 exactly where ridge keeps it nonzero (same data, same λ)',
    params: { n: 40, nFeatures: 4, noise: 0.5, lambda: 1.5, seed: 42 },
    maxSteps: 2000,
    expect: {
      converged: true,
      finalMetrics: { w4: (v: number) => Math.abs(v) < 1e-12 },
    },
  },
  {
    // Plan case 4 (run-level): coordinate descent terminates cleanly and the fit is
    // good (r² ≈ 0.99 at λ = 0.5 — the shrunk weights still explain the data; MSE ≈ 1.2
    // includes noise ≈ 0.08 plus the L1 shrinkage bias ≈ Σ(λ/σⱼ)²·Var(xⱼ)). The
    // per-sweep MONOTONE objective decrease is asserted on the trajectory in the test file.
    name: 'coordinate descent converges to a good fit at moderate λ',
    params: { n: 40, nFeatures: 4, noise: 0.5, lambda: 0.5, seed: 42 },
    maxSteps: 2000,
    expect: {
      converged: true,
      finalMetrics: { r2: (v: number) => v > 0.9 },
    },
  },
  {
    // Extra (cheap): λ = 10 exceeds every normalized correlation (max |w̃| ≈ 8.7),
    // so EVERY coefficient is soft-thresholded to exactly 0 → pure-intercept underfit.
    name: 'λ larger than every correlation zeroes all coefficients (underfit)',
    params: { n: 40, nFeatures: 4, noise: 0.5, lambda: 10, seed: 42 },
    maxSteps: 2000,
    expect: {
      converged: true,
      finalMetrics: {
        w1: (v: number) => v === 0,
        nNonzero: (v: number) => v === 0,
      },
    },
  },
];
