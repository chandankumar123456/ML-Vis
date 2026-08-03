// src/topics/polynomial-regression/comparisons.ts
import type { Comparison } from '../../engine/types';

export const polyComparisons: Comparison[] = [
  {
    id: 'poly-vs-slr',
    title: 'Polynomial Regression vs Simple Linear Regression',
    topics: ['polynomial-regression', 'simple-linear-regression'],
    axes: [
      {
        axis: 'Model form',
        entries: [
          { topic: 'simple-linear-regression', value: 'ŷ = wx + b — a single straight line' },
          { topic: 'polynomial-regression', value: 'ŷ = w₀ + w₁u + … + w_d u^d — a flexible curve (d ≥ 2)' },
        ],
      },
      {
        axis: 'Bias–variance behavior',
        entries: [
          { topic: 'simple-linear-regression', value: 'Low variance, but high bias on curved data' },
          { topic: 'polynomial-regression', value: 'Degree controls the bias–variance trade-off directly' },
        ],
      },
      {
        axis: 'Solution',
        entries: [
          { topic: 'simple-linear-regression', value: 'Closed-form 2×2 solve (Σx², Σx, Σxy)' },
          { topic: 'polynomial-regression', value: 'Same normal equation, but on the (d+1)×(d+1) Vandermonde Gram' },
        ],
      },
    ],
    notes: [
      'SLR is polynomial regression with d = 1 (or d = 0 for the mean model).',
      'GATE: a line is the d = 1 special case of the polynomial family, not a different method.',
    ],
  },
  {
    id: 'poly-vs-ridge',
    title: 'Polynomial Regression vs Ridge (the overfitting fix)',
    topics: ['polynomial-regression', 'ridge-regression'],
    axes: [
      {
        axis: 'Objective',
        entries: [
          { topic: 'polynomial-regression', value: 'min ‖y − Φθ‖² — unconstrained least squares' },
          { topic: 'ridge-regression', value: 'min ‖y − Φθ‖² + λ‖θ‖² — penalizes coefficient magnitude' },
        ],
      },
      {
        axis: 'High degree behavior',
        entries: [
          { topic: 'polynomial-regression', value: 'Coefficients explode and test error blows up (variance)' },
          { topic: 'ridge-regression', value: 'λ shrinks the exploding coefficients — test error stays low at high degree' },
        ],
      },
      {
        axis: 'ΦᵀΦ invertible?',
        entries: [
          { topic: 'polynomial-regression', value: 'Required — fails at high degree / degree ≥ n' },
          { topic: 'ridge-regression', value: 'Not needed — ΦᵀΦ + λI is always invertible for λ > 0' },
        ],
      },
    ],
    notes: [
      'Ridge is the standard remedy for polynomial overfitting: it trades a little bias for a large reduction in variance.',
      'GATE: adding λ to the diagonal of the Vandermonde Gram simultaneously fixes conditioning AND overfitting.',
    ],
  },
  {
    id: 'poly-vs-lasso',
    title: 'Polynomial Regression vs Lasso (sparse basis selection)',
    topics: ['polynomial-regression', 'lasso-regression'],
    axes: [
      {
        axis: 'Penalty',
        entries: [
          { topic: 'polynomial-regression', value: 'None — all d + 1 basis coefficients stay' },
          { topic: 'lasso-regression', value: 'λ‖θ‖₁ — soft-thresholding drives small coefficients to exactly 0' },
        ],
      },
      {
        axis: 'Effect on the basis',
        entries: [
          { topic: 'polynomial-regression', value: 'Every power u^j contributes, even irrelevant ones' },
          { topic: 'lasso-regression', value: 'Irrelevant powers are zeroed — automatic degree selection' },
        ],
      },
      {
        axis: 'Solution',
        entries: [
          { topic: 'polynomial-regression', value: 'Closed-form normal equation' },
          { topic: 'lasso-regression', value: 'No closed form — coordinate descent / soft-thresholding' },
        ],
      },
    ],
    notes: [
      'For a quadratic truth, lasso with enough λ zeroes w₃ … w_d, effectively selecting degree 2.',
      'Ridge shrinks but never zeroes; lasso is the one that produces exact zeros (sparsity).',
    ],
  },
];
