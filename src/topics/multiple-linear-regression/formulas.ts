// src/topics/multiple-linear-regression/formulas.ts
import type { Formula } from '../../engine/types';

export const mlrFormulas: Formula[] = [
  {
    id: 'hypothesis',
    latex: '\\hat{y} = \\theta^T x = w_1 x_1 + w_2 x_2 + \\dots + w_d x_d + b',
    symbols: [
      { symbol: 'x', meaning: 'feature vector (d × 1) of one sample', dimensions: 'd features' },
      { symbol: 'w_j', meaning: 'weight for feature j — change in ŷ per unit x_j', dimensions: 'y per unit x_j' },
      { symbol: 'b', meaning: 'bias / intercept', dimensions: 'y units' },
      { symbol: '\\hat{y}', meaning: 'prediction', dimensions: 'output units' },
    ],
    assumptions: ['Linear relationship between each feature and y', 'No perfect multicollinearity'],
    failureCases: ['Nonlinear feature–target relations', 'Collinear features make weights uninterpretable'],
    connections: ['MSE', 'Normal equation'],
    whyWorks: 'Each feature contributes additively; a hyperplane in d+1 dimensions generalizes the SLR line.',
  },
  {
    id: 'mse',
    latex: 'J(\\theta) = \\frac{1}{n} (y - X\\theta)^T (y - X\\theta)',
    symbols: [
      { symbol: 'X', meaning: 'design matrix (n × (d+1)) with bias column of 1s', dimensions: 'n samples × d+1 columns' },
      { symbol: 'y', meaning: 'target vector (n × 1)', dimensions: 'n samples' },
      { symbol: '\\theta', meaning: 'parameter vector (d+1) × 1 = [w₁ … w_d, b]', dimensions: 'd+1 parameters' },
      { symbol: 'n', meaning: 'number of samples', dimensions: 'count' },
    ],
    assumptions: ['Errors symmetric (Gaussian-like)', 'Homoscedastic noise'],
    failureCases: ['Outliers get squared — huge influence', 'Rank-deficient X still fits, but θ is meaningless'],
    derivesFrom: ['hypothesis'],
    connections: ['MLE with Gaussian noise'],
    whyWorks: 'Quadratic loss is convex in θ — the optimum is the closed-form least-squares solution.',
  },
  {
    id: 'normal-equation',
    latex: '\\theta = (X^T X)^{-1} X^T y',
    symbols: [
      { symbol: 'X', meaning: 'design matrix (n × (d+1))', dimensions: 'n samples × d+1 columns' },
      { symbol: 'X^T X', meaning: 'Gram matrix ((d+1) × (d+1)) — invertible iff X has full column rank', dimensions: 'd+1 × d+1' },
      { symbol: 'X^T y', meaning: 'cross-product vector ((d+1) × 1)', dimensions: 'd+1' },
      { symbol: '\\theta', meaning: 'parameter vector', dimensions: 'd+1 parameters' },
    ],
    assumptions: ['X^T X invertible (no perfect multicollinearity, n ≥ d+1)'],
    failureCases: ['Rank-deficient X (collinear features)', 'n < d+1 (underdetermined)'],
    derivesFrom: ['mse'],
    connections: ['Projection onto column space', 'Ridge adds λI to restore invertibility'],
    whyWorks: 'Setting the gradient of MSE to zero gives XᵀXθ = Xᵀy — the normal equations.',
  },
  {
    id: 'r2',
    latex: 'R^2 = 1 - \\frac{\\sum_i (y_i - \\hat{y}_i)^2}{\\sum_i (y_i - \\bar{y})^2}',
    symbols: [
      { symbol: 'y_i', meaning: 'true target of sample i', dimensions: 'output units' },
      { symbol: '\\hat{y}_i', meaning: 'prediction of sample i', dimensions: 'output units' },
      { symbol: '\\bar{y}', meaning: 'mean of targets', dimensions: 'output units' },
    ],
    assumptions: ['Constant term (bias) in the model'],
    failureCases: ['R² near 1 with garbage features when n ≈ d (overfitting)', 'Negative R² possible for terrible fits'],
    derivesFrom: ['mse'],
    connections: ['Coefficient of determination', 'Correlation squared in SLR'],
    whyWorks: 'Fraction of target variance explained by the model, relative to the mean-only baseline.',
  },
];
