// src/topics/simple-linear-regression/derivations.ts
import type { Derivation } from '../../engine/types';

export const slrDerivations: Derivation[] = [
  {
    id: 'ols-derivation',
    title: 'Deriving the Normal Equation from MSE',
    steps: [
      {
        latex: 'J(\\theta) = \\frac{1}{n} (y - X\\theta)^T (y - X\\theta)',
        justification: 'MSE in matrix form.',
      },
      {
        latex: 'J(\\theta) = \\frac{1}{n} (y^T y - 2\\theta^T X^T y + \\theta^T X^T X \\theta)',
        justification: 'Expand the product.',
      },
      {
        latex: '\\nabla_\\theta J = \\frac{1}{n} (-2 X^T y + 2 X^T X \\theta)',
        justification: 'Matrix calculus: d(θᵀAθ)/dθ = 2Aθ; d(θᵀc)/dθ = c.',
      },
      {
        latex: '0 = -2 X^T y + 2 X^T X \\theta',
        justification: 'Set gradient to zero (convexity ⇒ global min).',
      },
      {
        latex: '\\theta = (X^T X)^{-1} X^T y',
        justification: 'Solve for θ — the normal equation.',
      },
    ],
    derivedFrom: ['normal-equation'],
  },
];
