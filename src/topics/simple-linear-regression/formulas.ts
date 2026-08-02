// src/topics/simple-linear-regression/formulas.ts
import type { Formula } from '../../engine/types';

export const slrFormulas: Formula[] = [
  {
    id: 'hypothesis',
    latex: '\\hat{y} = w x + b',
    symbols: [
      { symbol: 'w', meaning: 'slope / weight', dimensions: 'y per unit x' },
      { symbol: 'b', meaning: 'intercept / bias', dimensions: 'y units' },
      { symbol: 'x', meaning: 'feature value', dimensions: 'input units' },
      { symbol: '\\hat{y}', meaning: 'prediction', dimensions: 'output units' },
    ],
    assumptions: ['Linear relationship between x and y'],
    failureCases: ['Nonlinear data', 'Outliers dominate fit'],
    connections: ['MSE', 'Normal equation'],
    whyWorks: 'A line is the simplest parametric model; linearity makes fitting tractable.',
  },
  {
    id: 'mse',
    latex: '\\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2',
    symbols: [
      { symbol: 'n', meaning: 'number of samples', dimensions: 'count' },
      { symbol: 'y_i', meaning: 'true target of sample i', dimensions: 'output units' },
      { symbol: '\\hat{y}_i', meaning: 'prediction of sample i', dimensions: 'output units' },
    ],
    assumptions: ['Errors symmetric (Gaussian-like)'],
    failureCases: ['Outliers get squared — huge influence', 'Heteroscedastic noise mis-modeled'],
    derivesFrom: ['hypothesis'],
    connections: ['MLE with Gaussian noise'],
    whyWorks: 'Quadratic loss makes the optimum closed-form (convex).',
  },
  {
    id: 'normal-equation',
    latex: '\\theta = (X^T X)^{-1} X^T y',
    symbols: [
      { symbol: 'X', meaning: 'design matrix (n × d)', dimensions: 'n samples × d features' },
      { symbol: 'y', meaning: 'target vector (n × 1)', dimensions: 'n samples' },
      { symbol: '\\theta', meaning: 'parameter vector (d × 1)', dimensions: 'd features' },
    ],
    assumptions: ['X^T X invertible (no perfect multicollinearity)'],
    failureCases: ['Rank-deficient X (collinear features)', 'n < d (underdetermined)'],
    derivesFrom: ['mse'],
    connections: ['Projection onto column space', 'SVD-based pseudo-inverse'],
    whyWorks: 'Minimizes MSE by setting gradient to zero — the normal equations.',
  },
];
