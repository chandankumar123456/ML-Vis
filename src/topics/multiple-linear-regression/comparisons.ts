// src/topics/multiple-linear-regression/comparisons.ts
import type { Comparison } from '../../engine/types';

export const mlrComparisons: Comparison[] = [
  {
    id: 'mlr-vs-slr',
    title: 'MLR vs Simple Linear Regression',
    topics: ['multiple-linear-regression', 'simple-linear-regression'],
    axes: [
      {
        axis: 'Features',
        entries: [
          { topic: 'simple-linear-regression', value: 'One feature x → line y = wx + b' },
          { topic: 'multiple-linear-regression', value: 'd ≥ 2 features → hyperplane ŷ = wᵀx + b' },
        ],
      },
      {
        axis: 'Geometry',
        entries: [
          { topic: 'simple-linear-regression', value: 'Best-fit line in 2D' },
          { topic: 'multiple-linear-regression', value: 'Best-fit hyperplane in (d+1)D' },
        ],
      },
      {
        axis: 'Solution',
        entries: [
          { topic: 'simple-linear-regression', value: 'Closed-form via Σx², Σxy (2×2 solve)' },
          { topic: 'multiple-linear-regression', value: 'Same normal equation, but (d+1)×(d+1) solve' },
        ],
      },
    ],
    notes: [
      'SLR is the d = 1 special case of MLR — the normal equation is the same formula.',
      'MLR introduces rank/conditioning concerns that never appear in SLR.',
    ],
  },
  {
    id: 'mlr-vs-ridge',
    title: 'MLR vs Ridge Regression (large d)',
    topics: ['multiple-linear-regression', 'ridge-regression'],
    axes: [
      {
        axis: 'Objective',
        entries: [
          { topic: 'multiple-linear-regression', value: 'min ‖y − Xθ‖²' },
          { topic: 'ridge-regression', value: 'min ‖y − Xθ‖² + λ‖θ‖²' },
        ],
      },
      {
        axis: 'XᵀX invertible?',
        entries: [
          { topic: 'multiple-linear-regression', value: 'Required — fails under collinearity' },
          { topic: 'ridge-regression', value: 'Not needed — XᵀX + λI is always invertible for λ > 0' },
        ],
      },
      {
        axis: 'When d is large',
        entries: [
          { topic: 'multiple-linear-regression', value: 'O(d³) inversion; unstable when n ≈ d' },
          { topic: 'ridge-regression', value: 'Shrinks weights, trades variance for bias' },
        ],
      },
    ],
    notes: [
      'Ridge is MLR plus a λ‖θ‖² penalty — it restores invertibility and tames overfitting.',
      'GATE: ridge guarantees a unique solution even when XᵀX is singular.',
    ],
  },
  {
    id: 'mlr-vs-polynomial',
    title: 'MLR vs Polynomial Regression (nonlinearity)',
    topics: ['multiple-linear-regression', 'polynomial-regression'],
    axes: [
      {
        axis: 'Model form',
        entries: [
          { topic: 'multiple-linear-regression', value: 'Linear in raw features x₁…x_d' },
          { topic: 'polynomial-regression', value: 'Linear in transformed features φ(x) = [1, x, x², …]' },
        ],
      },
      {
        axis: 'Nonlinear patterns',
        entries: [
          { topic: 'multiple-linear-regression', value: 'Missed — the fit is a hyperplane' },
          { topic: 'polynomial-regression', value: 'Captured via basis expansion (still linear in θ)' },
        ],
      },
      {
        axis: 'Fit algorithm',
        entries: [
          { topic: 'multiple-linear-regression', value: 'Normal equation on X' },
          { topic: 'polynomial-regression', value: 'Normal equation on the Vandermonde Φ — identical math' },
        ],
      },
    ],
    notes: [
      'Both use the SAME normal equation — polynomial regression is MLR on engineered features.',
      'High-degree polynomial bases reintroduce conditioning problems (Runge’s phenomenon).',
    ],
  },
];
