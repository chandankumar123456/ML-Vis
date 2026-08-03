// src/topics/polynomial-regression/derivations.ts
import type { Derivation } from '../../engine/types';

export const polyDerivations: Derivation[] = [
  {
    id: 'poly-normal-equation',
    title: 'Deriving the Normal Equation for the Polynomial Basis',
    steps: [
      {
        latex: '\\Phi = \\begin{bmatrix} 1 & u_1 & u_1^2 & \\dots & u_1^d \\\\ 1 & u_2 & u_2^2 & \\dots & u_2^d \\\\ \\vdots & & & & \\vdots \\\\ 1 & u_n & u_n^2 & \\dots & u_n^d \\end{bmatrix}_{n \\times (d+1)}',
        justification: 'Vandermonde design matrix: the constant column comes first (polynomial convention), then the powers of each normalized input u_i = x_i/x_max.',
      },
      {
        latex: 'J(\\theta) = \\frac{1}{n} (y - \\Phi\\theta)^T (y - \\Phi\\theta)',
        justification: 'MSE on the transformed features — identical in form to linear regression because θ enters linearly.',
      },
      {
        latex: '\\nabla_\\theta J = \\frac{1}{n} \\left( -2 \\Phi^T y + 2 \\Phi^T \\Phi \\theta \\right)',
        justification: 'Matrix calculus: ∂(θᵀΦᵀΦθ)/∂θ = 2ΦᵀΦθ (Gram matrix symmetric), ∂(θᵀΦᵀy)/∂θ = Φᵀy.',
      },
      {
        latex: '0 = -2 \\Phi^T y + 2 \\Phi^T \\Phi \\theta \\quad\\Rightarrow\\quad \\Phi^T \\Phi \\theta = \\Phi^T y',
        justification: 'MSE is convex in θ, so the stationary point is the global minimum.',
      },
      {
        latex: '\\theta = (\\Phi^T \\Phi)^{-1} \\Phi^T y',
        justification: 'Left-multiply by (ΦᵀΦ)⁻¹ — valid exactly when the Gram matrix is invertible (degree + 1 ≤ n and a well-conditioned basis).',
      },
    ],
    derivedFrom: ['normal-equation'],
  },
  {
    id: 'basis-expansion-linearity',
    title: 'Why the Basis Expansion Keeps the Model Linear in Parameters',
    steps: [
      {
        latex: '\\hat{y}(u) = w_0 + w_1 u + w_2 u^2 + \\dots + w_d u^d',
        justification: 'The model is a linear combination of FIXED nonlinear functions of the input.',
      },
      {
        latex: '\\phi(u) = [1,\\ u,\\ u^2,\\ \\dots,\\ u^d]^T \\quad\\Rightarrow\\quad \\hat{y} = \\theta^T \\phi(u)',
        justification: 'Pack the basis functions into a feature vector φ(u); the model becomes θᵀφ — the standard linear form with engineered features.',
      },
      {
        latex: '\\frac{\\partial \\hat{y}}{\\partial w_j} = u^j \\quad\\text{(independent of every other } w_k\\text{)}',
        justification: 'Each parameter multiplies exactly one known basis function. There is no term where two parameters multiply each other — the nonlinearity lives entirely in the features, not in θ.',
      },
      {
        latex: '\\text{Polynomial regression} \\;\\equiv\\; \\text{linear regression on } \\phi(u)',
        justification: 'Consequence: the normal equation, its derivation, and its closed-form solution apply unchanged — the only novelty is the columns of the design matrix (a Vandermonde).',
      },
    ],
    derivedFrom: ['basis'],
  },
  {
    id: 'bias-variance-decomposition',
    title: 'Decomposing Expected Error into Bias, Variance, and Noise',
    steps: [
      {
        latex: '\\mathbb{E}_\\mathcal{D}[(y - \\hat{f}_\\mathcal{D}(x))^2]',
        justification: 'Expected squared error of the fitted model, averaged over random training datasets D — this is why a single training/test split is only an estimate.',
      },
      {
        latex: '\\hat{f}_\\mathcal{D}(x) - \\mathbb{E}_\\mathcal{D}[\\hat{f}_\\mathcal{D}(x)] = \\text{fluctuation around the average fit}',
        justification: 'Split the deviation of the fitted value from the true y into a part that depends on the dataset and a part that does not.',
      },
      {
        latex: '\\mathbb{E}[(y - \\hat{f})^2] = \\underbrace{(\\mathbb{E}[\\hat{f}] - f)^2}_{\\text{bias}^2} + \\underbrace{\\mathbb{E}[(\\hat{f} - \\mathbb{E}[\\hat{f}])^2]}_{\\text{variance}} + \\underbrace{\\sigma^2}_{\\text{noise}}',
        justification: 'Expanding the square and using E[(y−f)(f−E[f])] = 0 (noise is independent and zero-mean) gives the classic three-term decomposition.',
      },
      {
        latex: 'd \\uparrow \\Rightarrow \\text{bias} \\downarrow,\\; \\text{variance} \\uparrow',
        justification: 'Higher degree lets the polynomial hug the training points (lower bias) but makes it change violently between datasets (higher variance) — the trade-off that produces the U-shaped test-error curve.',
      },
    ],
    derivedFrom: ['bias-variance'],
  },
];
