// src/topics/ridge-regression/comparisons.ts
import type { Comparison } from '../../engine/types';

export const ridgeComparisons: Comparison[] = [
  {
    id: 'ridge-vs-lasso',
    title: 'Ridge vs Lasso (L2 circle vs L1 diamond)',
    topics: ['ridge-regression', 'lasso-regression'],
    axes: [
      {
        axis: 'Penalty',
        entries: [
          { topic: 'ridge-regression', value: 'λ‖θ‖₂² — squared L2, smooth everywhere' },
          { topic: 'lasso-regression', value: 'λ‖θ‖₁ — absolute L1, kink at 0' },
        ],
      },
      {
        axis: 'Geometry',
        entries: [
          { topic: 'ridge-regression', value: 'Constraint ‖θ‖₂ ≤ t is a CIRCLE — touch point on a smooth arc' },
          { topic: 'lasso-regression', value: 'Constraint ‖θ‖₁ ≤ t is a DIAMOND — corners at the axes' },
        ],
      },
      {
        axis: 'Coefficients',
        entries: [
          { topic: 'ridge-regression', value: 'Shrinks every coefficient, never exactly to 0' },
          { topic: 'lasso-regression', value: 'Can drive coefficients to EXACT 0 (feature selection)' },
        ],
      },
      {
        axis: 'Closed form?',
        entries: [
          { topic: 'ridge-regression', value: 'Yes: θ = (XᵀX + λI)⁻¹Xᵀy' },
          { topic: 'lasso-regression', value: 'No — solved iteratively (coordinate descent / soft-threshold)' },
        ],
      },
      {
        axis: 'Correlated features',
        entries: [
          { topic: 'ridge-regression', value: 'Keeps all correlated features, shrinks them together' },
          { topic: 'lasso-regression', value: 'Picks one arbitrarily and zeroes the others' },
        ],
      },
    ],
    notes: [
      'The circle-vs-diamond figure is the classic GATE contrast: corners cause exact zeros.',
      'Ridge groups correlated features; lasso selects — for highly correlated groups ridge often generalizes better.',
    ],
  },
  {
    id: 'ridge-vs-ols',
    title: 'Ridge vs OLS (bias–variance and invertibility)',
    topics: ['ridge-regression', 'multiple-linear-regression'],
    axes: [
      {
        axis: 'Objective',
        entries: [
          { topic: 'ridge-regression', value: 'min ‖y − Xθ‖² + λ‖θ‖²' },
          { topic: 'multiple-linear-regression', value: 'min ‖y − Xθ‖² (λ = 0 case)' },
        ],
      },
      {
        axis: 'XᵀX invertible?',
        entries: [
          { topic: 'ridge-regression', value: 'Not needed — XᵀX + λI is always invertible for λ > 0' },
          { topic: 'multiple-linear-regression', value: 'Required — fails under collinearity / n < d+1' },
        ],
      },
      {
        axis: 'Estimator',
        entries: [
          { topic: 'ridge-regression', value: 'θ_λ = R_λ·θ_OLS — biased, smaller variance' },
          { topic: 'multiple-linear-regression', value: 'θ_OLS = (XᵀX)⁻¹Xᵀy — unbiased, full variance' },
        ],
      },
      {
        axis: 'When λ = 0',
        entries: [
          { topic: 'ridge-regression', value: 'Identical to OLS — the penalty vanishes' },
          { topic: 'multiple-linear-regression', value: 'The reference point of the ridge path' },
        ],
      },
    ],
    notes: [
      'Ridge = OLS plus an L2 penalty; θ = 0 is the special case.',
      'GATE: ridge guarantees a unique solution where OLS is singular, at the price of introducing bias.',
    ],
  },
  {
    id: 'ridge-vs-polynomial',
    title: 'Ridge vs Polynomial Regression (ridge as an overfitting fix)',
    topics: ['ridge-regression', 'polynomial-regression'],
    axes: [
      {
        axis: 'Where each is used',
        entries: [
          { topic: 'ridge-regression', value: 'Shrinks coefficients of ANY basis — applied to the polynomial design too' },
          { topic: 'polynomial-regression', value: 'High degree fits train noise → coefficients explode' },
        ],
      },
      {
        axis: 'Overfitting fix',
        entries: [
          { topic: 'ridge-regression', value: 'λ‖θ‖² tames the exploding coefficients of high-degree fits' },
          { topic: 'polynomial-regression', value: 'Degree alone cannot fix variance — needs regularization or more data' },
        ],
      },
      {
        axis: 'Bias–variance',
        entries: [
          { topic: 'ridge-regression', value: 'Explicit dial (λ) trading bias for variance' },
          { topic: 'polynomial-regression', value: 'Implicit: degree ↑ raises variance; degree ↓ raises bias' },
        ],
      },
    ],
    notes: [
      'Ridge-on-polynomial-features is the standard "regularized polynomial" — both GATE favourites combine cleanly.',
      'High-degree Vandermonde matrices are ill-conditioned; ridge (or feature scaling) is the practical fix.',
    ],
  },
];
