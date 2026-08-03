// src/topics/polynomial-regression/formulas.ts
import type { Formula } from '../../engine/types';

export const polyFormulas: Formula[] = [
  {
    id: 'basis',
    latex: '\\phi(u) = [1,\\ u,\\ u^2,\\ \\dots,\\ u^d]^T,\\quad u = \\frac{x}{x_{\\max}}',
    symbols: [
      { symbol: 'x', meaning: 'raw input feature', dimensions: 'x units' },
      { symbol: 'u', meaning: 'normalized input u = x/x_max ∈ [−1, 1] — keeps every power u^j bounded', dimensions: 'dimensionless' },
      { symbol: 'd', meaning: 'polynomial degree (number of basis functions is d + 1)', dimensions: 'count' },
      { symbol: 'x_{\\max}', meaning: 'input range bound (3 in the simulation)', dimensions: 'x units' },
    ],
    assumptions: ['Input normalized to [−1, 1] before expanding (standard conditioning fix)', 'Basis is fixed before fitting — only θ is learned'],
    failureCases: ['Degree too high → Vandermonde ill-conditioned (κ grows super-exponentially)', 'Degree + 1 > n → rank-deficient basis, no unique fit'],
    connections: ['hypothesis', 'normal-equation'],
    whyWorks: 'Polynomial regression is linear regression on the expanded feature vector φ(u): the basis functions are fixed and only θ is learned, so the entire linear-regression machinery applies unchanged.',
  },
  {
    id: 'hypothesis',
    latex: '\\hat{y}(u) = \\theta^T \\phi(u) = w_0 + w_1 u + w_2 u^2 + \\dots + w_d u^d',
    symbols: [
      { symbol: 'w_j', meaning: 'coefficient on the basis function u^j (the fitted weight)', dimensions: 'y per u^j' },
      { symbol: 'w_0', meaning: 'intercept (constant term)', dimensions: 'y units' },
      { symbol: '\\theta', meaning: 'parameter vector (d+1) × 1', dimensions: 'd+1 parameters' },
      { symbol: '\\hat{y}', meaning: 'prediction', dimensions: 'output units' },
    ],
    assumptions: ['Target is well-approximated by a degree-d polynomial in u'],
    failureCases: ['Degree too low → bias (underfitting)', 'Degree too high → variance (overfitting, coefficients explode)'],
    derivesFrom: ['basis'],
    connections: ['mse', 'bias-variance'],
    whyWorks: 'The model stays LINEAR IN PARAMETERS — only the feature vector is nonlinear. That is why the normal equation applies and why a degree-d model has exactly d + 1 parameters.',
  },
  {
    id: 'mse',
    latex: 'J(\\theta) = \\frac{1}{n} (y - \\Phi\\theta)^T (y - \\Phi\\theta)',
    symbols: [
      { symbol: '\\Phi', meaning: 'Vandermonde design matrix (n × (d+1)), row i = [1, u_i, u_i², …, u_i^d]', dimensions: 'n samples × d+1 columns' },
      { symbol: 'y', meaning: 'target vector (n × 1)', dimensions: 'n samples' },
      { symbol: 'n', meaning: 'number of training samples', dimensions: 'count' },
      { symbol: '\\theta', meaning: 'parameter vector (d+1) × 1', dimensions: 'd+1 parameters' },
    ],
    assumptions: ['Symmetric zero-mean noise (Gaussian-like)', 'Homoscedastic errors'],
    failureCases: ['Outliers get squared — huge influence', 'On the train split the MSE is optimistically small when d ≈ n (overfitting)'],
    derivesFrom: ['hypothesis'],
    connections: ['MLE with Gaussian noise', 'normal-equation'],
    whyWorks: 'Quadratic loss in θ is convex, so the closed-form least-squares solution is the global optimum — the same proof as in linear regression.',
  },
  {
    id: 'normal-equation',
    latex: '\\theta = (\\Phi^T \\Phi)^{-1} \\Phi^T y',
    symbols: [
      { symbol: '\\Phi^T \\Phi', meaning: 'Gram matrix ((d+1) × (d+1)) — invertible iff Φ has full column rank', dimensions: 'd+1 × d+1' },
      { symbol: '\\Phi^T y', meaning: 'cross-product vector ((d+1) × 1)', dimensions: 'd+1' },
      { symbol: '\\theta', meaning: 'least-squares parameter vector', dimensions: 'd+1 parameters' },
    ],
    assumptions: ['ΦᵀΦ invertible: degree + 1 ≤ n and the basis is well-conditioned'],
    failureCases: ['degree + 1 > n → singular (infinite solutions)', 'High degree → numerically unstable inverse (huge condition number, e.g. degree 30 fails outright)'],
    derivesFrom: ['mse'],
    connections: ['projection onto column space', 'ridge adds λI to restore invertibility'],
    whyWorks: 'Setting ∇J = 0 on the polynomial basis gives exactly XᵀXθ = Xᵀy — the basis expansion never changes the derivation, only the columns of X.',
  },
  {
    id: 'train-test-split',
    latex: '\\text{Test MSE} = \\frac{1}{n_{\\text{test}}} \\sum_{i \\in \\text{test}} (y_i - \\hat{y}_i)^2',
    symbols: [
      { symbol: 'n_{\\text{test}}', meaning: 'number of held-out test samples (never used for fitting)', dimensions: 'count' },
      { symbol: '\\hat{y}_i', meaning: 'prediction on a held-out test point', dimensions: 'output units' },
    ],
    assumptions: ['Test points are drawn from the same distribution as train', 'The model never sees test targets during fitting'],
    failureCases: ['Fitting on all data → test error is in-sample (optimistic, hides overfitting)', 'Test set too small → noisy generalization estimate'],
    connections: ['bias-variance', 'cross-validation'],
    whyWorks: 'A model can always memorize its training data; only performance on unseen data estimates how well it generalizes.',
  },
  {
    id: 'bias-variance',
    latex: '\\mathbb{E}[(y - \\hat{f}(x))^2] = \\underbrace{\\text{Bias}(\\hat{f})^2}_{\\text{underfit}} + \\underbrace{\\text{Var}(\\hat{f})}_{\\text{overfit}} + \\underbrace{\\sigma^2}_{\\text{irreducible}}',
    symbols: [
      { symbol: '\\text{Bias}(\\hat{f})', meaning: 'systematic error — model too simple (low degree)', dimensions: 'y units' },
      { symbol: '\\text{Var}(\\hat{f})', meaning: 'sensitivity to the training sample — model too flexible (high degree)', dimensions: 'y²' },
      { symbol: '\\sigma^2', meaning: 'irreducible noise variance', dimensions: 'y²' },
    ],
    assumptions: ['Noise independent of the input', 'Same input distribution for train and test'],
    failureCases: ['Degree 1 → bias dominates; degree 15 → variance dominates; neither can be fixed by more data of the same kind'],
    connections: ['train-test-split', 'ridge-regression'],
    whyWorks: 'Expected error decomposes into the model’s systematic miss (bias), its instability across datasets (variance), and the noise floor — raising the degree trades bias for variance.',
  },
];
