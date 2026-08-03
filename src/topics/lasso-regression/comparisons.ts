// src/topics/lasso-regression/comparisons.ts
import type { Comparison } from '../../engine/types';

export const lassoComparisons: Comparison[] = [
  {
    id: 'lasso-vs-ridge',
    title: 'Lasso vs Ridge Regression — the Diamond vs the Circle',
    topics: ['lasso-regression', 'ridge-regression'],
    axes: [
      {
        axis: 'Penalty',
        entries: [
          { topic: 'lasso-regression', value: 'λ‖θ‖₁ — L1, non-differentiable at 0' },
          { topic: 'ridge-regression', value: 'λ‖θ‖₂² — L2, smooth everywhere' },
        ],
      },
      {
        axis: 'Constraint geometry',
        entries: [
          { topic: 'lasso-regression', value: 'Diamond with vertices on the axes → optimum can touch a corner' },
          { topic: 'ridge-regression', value: 'Circle with no corners → optimum touches the smooth arc' },
        ],
      },
      {
        axis: 'Sparsity',
        entries: [
          { topic: 'lasso-regression', value: 'Exact zeros — performs feature selection (nNonzero metric)' },
          { topic: 'ridge-regression', value: 'Never exactly 0 — shrinks proportionally, keeps all features' },
        ],
      },
      {
        axis: 'Solution',
        entries: [
          { topic: 'lasso-regression', value: 'No closed form — coordinate descent / LARS (iterative)' },
          { topic: 'ridge-regression', value: 'Closed form θ = (XᵀX + λI)⁻¹Xᵀy — one matrix solve' },
        ],
      },
      {
        axis: 'Collinear features',
        entries: [
          { topic: 'lasso-regression', value: 'Handles it but picks arbitrarily between the tied features' },
          { topic: 'ridge-regression', value: 'Shares the coefficient among them — stable, group-like shrinkage' },
        ],
      },
    ],
    notes: [
      'The diamond-vs-circle figure is the classic GATE contrast: L1 corners ⇒ exact zeros, L2 arc ⇒ no zeros.',
      'Both are "MLR + penalty" — identical at λ = 0 — and both need standardized features.',
    ],
  },
  {
    id: 'lasso-vs-ols',
    title: 'Lasso vs OLS (Multiple Linear Regression)',
    topics: ['lasso-regression', 'multiple-linear-regression'],
    axes: [
      {
        axis: 'Objective',
        entries: [
          { topic: 'lasso-regression', value: 'min MSE + λ‖θ‖₁ — trades bias for sparsity' },
          { topic: 'multiple-linear-regression', value: 'min MSE — unbiased, no penalty' },
        ],
      },
      {
        axis: 'At λ = 0',
        entries: [
          { topic: 'lasso-regression', value: 'Soft-threshold becomes the identity → exactly the OLS solution' },
          { topic: 'multiple-linear-regression', value: 'θ = (XᵀX)⁻¹Xᵀy (the same point)' },
        ],
      },
      {
        axis: 'Feature selection',
        entries: [
          { topic: 'lasso-regression', value: 'Drops irrelevant features automatically (interpretable model)' },
          { topic: 'multiple-linear-regression', value: 'Keeps every feature with a small nonzero weight — no selection' },
        ],
      },
      {
        axis: 'Variance',
        entries: [
          { topic: 'lasso-regression', value: 'Lower variance (regularized), at the cost of some bias' },
          { topic: 'multiple-linear-regression', value: 'High variance when d is large or features are correlated' },
        ],
      },
    ],
    notes: [
      'lasso(λ→0) = OLS is the baseline identity — the tiny-λ test case verifies it numerically.',
      'When every feature matters and d is small, OLS wins; lasso helps when many features are noise.',
    ],
  },
  {
    id: 'lasso-vs-polynomial',
    title: 'Lasso vs Polynomial Regression (high-dimensional features)',
    topics: ['lasso-regression', 'polynomial-regression'],
    axes: [
      {
        axis: 'Feature set',
        entries: [
          { topic: 'lasso-regression', value: 'Raw (or engineered) features — selects a sparse subset' },
          { topic: 'polynomial-regression', value: 'Basis expansion φ(x) = [1, x, x², …] — many correlated terms' },
        ],
      },
      {
        axis: 'Overfitting',
        entries: [
          { topic: 'lasso-regression', value: 'Penalty keeps weights small and sparse — resists overfit' },
          { topic: 'polynomial-regression', value: 'High degree → coefficients explode, test error rises' },
        ],
      },
      {
        axis: 'Typical fix',
        entries: [
          { topic: 'lasso-regression', value: 'Lasso directly — sparse regularization is its whole point' },
          { topic: 'polynomial-regression', value: 'Ridge or lasso on the expanded Φ is the standard cure' },
        ],
      },
    ],
    notes: [
      'Lasso on a high-degree polynomial basis performs sparse feature selection among the nonlinear terms.',
      'Standardization is especially important for polynomial bases — powers of x span wildly different scales.',
    ],
  },
];
