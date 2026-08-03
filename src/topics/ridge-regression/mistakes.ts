// src/topics/ridge-regression/mistakes.ts
import type { Mistake } from '../../engine/types';

export const ridgeMistakes: Mistake[] = [
  {
    id: 'ridge-vs-lasso-sparsity',
    pattern: 'Thinking ridge regression drives coefficients exactly to zero (confusing it with lasso)',
    example: '\\text{Ridge: } \\theta_j \\to 0 \\text{ but never } = 0 \\quad\\neq\\quad \\text{Lasso: } \\theta_j = 0',
    whyWrong: 'The L2 penalty is smooth with gradient 2λθ → 0 at θ = 0, so the optimum is interior. Exact zeros need the L1 kink that only lasso has. This ridge-vs-lasso sparsity contrast is a classic GATE trap.',
    gateTrap: true,
    relatedConcept: 'ridge-path',
  },
  {
    id: 'forgetting-plus-lambda-I',
    pattern: 'Writing the ridge closed form without the +λI term',
    example: '\\theta = (X^T X)^{-1} X^T y \\;\\text{(OLS)} \\quad\\text{instead of}\\quad \\theta = (X^T X + \\lambda I)^{-1} X^T y',
    whyWrong: 'Without +λI you have plain OLS — singular XᵀX (collinear features, n < d+1) fails. The +λI term is exactly what makes the Gram matrix invertible and shrinks the solution.',
    gateTrap: true,
    relatedConcept: 'ridge-closed-form',
  },
  {
    id: 'lambda-shrinks-only-large',
    pattern: 'Thinking λ shrinks only the large coefficients (or shrinks them all by the same amount)',
    example: '\\theta_j(\\lambda) = \\frac{\\mu_j}{\\mu_j + \\lambda}\\,\\theta_j(0) \\quad\\text{— every direction, scaled by its own factor}',
    whyWrong: 'In the eigenbasis each component is scaled by μₖ/(μₖ + λ): ALL coefficients are shrunk, proportionally — not just "the big ones", and not by a single shared amount. The magnitude of the shrinkage a coefficient experiences depends on its direction’s eigenvalue.',
    gateTrap: true,
    relatedConcept: 'ridge-path',
  },
  {
    id: 'lambda-direction',
    pattern: 'Reversing the λ direction: thinking a larger λ reduces bias / increases variance',
    whyWrong: 'λ is the shrinkage dial: larger λ → stronger penalty → MORE bias, LESS variance. Getting this backwards inverts the entire bias–variance story and the reading of the test-error curve.',
    gateTrap: true,
    relatedConcept: 'ridge-bias-variance',
  },
  {
    id: 'ridge-fixes-outliers',
    pattern: 'Assuming ridge makes the model robust to outliers',
    whyWrong: 'Ridge only changes the penalty — the loss is still the squared error, so outliers still get squared and dominate the fit. Robustness to outliers requires a different loss (e.g., Huber or L1 loss), not an L2 penalty on θ.',
    gateTrap: false,
    relatedConcept: 'ridge-objective',
  },
];
