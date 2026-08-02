// src/topics/multiple-linear-regression/derivations.ts
import type { Derivation } from '../../engine/types';

export const mlrDerivations: Derivation[] = [
  {
    id: 'mlr-normal-equation',
    title: 'Deriving the Normal Equation from the Matrix MSE',
    steps: [
      {
        latex: 'X = \\begin{bmatrix} x_{11} & x_{12} & \\dots & x_{1d} & 1 \\\\ \\vdots & & & \\vdots & \\vdots \\\\ x_{n1} & x_{n2} & \\dots & x_{nd} & 1 \\end{bmatrix}_{n \\times (d+1)}',
        justification: 'Design matrix: one row per sample, bias column of 1s appended last.',
      },
      {
        latex: 'J(\\theta) = \\frac{1}{n} (y - X\\theta)^T (y - X\\theta)',
        justification: 'MSE in matrix form — residual vector is (y − Xθ), length n.',
      },
      {
        latex: 'J(\\theta) = \\frac{1}{n} \\left( y^T y - 2\\theta^T X^T y + \\theta^T X^T X \\theta \\right)',
        justification: 'Expand (y−Xθ)ᵀ(y−Xθ) using (A+B)ᵀ = Aᵀ+Bᵀ and (Xθ)ᵀ = θᵀXᵀ.',
      },
      {
        latex: '\\nabla_\\theta J = \\frac{1}{n} \\left( -2 X^T y + 2 X^T X \\theta \\right)',
        justification: 'Matrix calculus: ∂(θᵀAθ)/∂θ = (A + Aᵀ)θ = 2Aθ for symmetric A; ∂(θᵀc)/∂θ = c.',
      },
      {
        latex: '0 = -2 X^T y + 2 X^T X \\theta \\quad\\Rightarrow\\quad X^T X \\theta = X^T y',
        justification: 'Set the gradient to zero. MSE is convex in θ, so this stationary point is the global minimum.',
      },
      {
        latex: '\\theta = (X^T X)^{-1} X^T y',
        justification: 'Left-multiply by (XᵀX)⁻¹ — valid exactly when XᵀX is invertible (full column rank).',
      },
    ],
    derivedFrom: ['normal-equation'],
  },
  {
    id: 'mlr-gradient-update',
    title: 'Gradient Descent Update from the MSE Gradient',
    steps: [
      {
        latex: '\\nabla_\\theta J = \\frac{2}{n} X^T (X\\theta - y)',
        justification: 'Reuse the gradient computed above (drop the factor 1/n notation difference: J includes 1/n).',
      },
      {
        latex: '\\theta \\leftarrow \\theta - \\eta \\cdot \\frac{2}{n} X^T (X\\theta - y)',
        justification: 'Move opposite the gradient by learning rate η — the full-batch GD update.',
      },
      {
        latex: 'X^T (X\\theta - y) = \\sum_{i=1}^{n} (\\hat{y}_i - y_i) x_i',
        justification: 'Per-sample view: the update is a weighted sum of each sample’s feature vector, weighted by its residual.',
      },
    ],
    derivedFrom: ['mse'],
  },
];
