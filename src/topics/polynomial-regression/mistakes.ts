// src/topics/polynomial-regression/mistakes.ts
import type { Mistake } from '../../engine/types';

export const polyMistakes: Mistake[] = [
  {
    id: 'poly-more-features-better',
    pattern: 'Assuming more features (higher degree) is always better',
    example: 'd = 15 \\;\\text{on 30 points} \\;\\Rightarrow\\; \\text{"more flexible = better"}',
    whyWrong: 'Every extra basis function adds a parameter. At degree 15 the model can fit the training noise itself — train MSE plummets while held-out test MSE explodes. This is the variance side of the bias-variance trade-off.',
    gateTrap: true,
    relatedConcept: 'bias-variance',
  },
  {
    id: 'poly-no-validation',
    pattern: 'Fitting on the full dataset and judging quality by train error',
    example: '\\text{train on all } n \\text{ points } \\Rightarrow \\text{ report train MSE as "accuracy"}',
    whyWrong: 'A flexible polynomial can memorize any finite dataset. Without a held-out split (or cross-validation) you cannot see overfitting — the training error is always optimistic. Always keep a test set the model never sees.',
    gateTrap: true,
    relatedConcept: 'train-test-split',
  },
  {
    id: 'poly-no-normalization',
    pattern: 'Ignoring input normalization / feature conditioning at high degree',
    example: '\\Phi_{ij} = x_i^j,\\quad x \\in [-3, 3],\\; j = 15 \\;\\Rightarrow\\; \\Phi \\text{ entries } \\approx 1.4 \\times 10^7',
    whyWrong: 'Raw powers of a wide-range input make the Vandermonde ill-conditioned (κ grows super-exponentially): the normal equation becomes numerically unstable long before degree 30. Normalize x to u = x/x_max ∈ [−1, 1] first (a change of variable that does not change the model class).',
    gateTrap: false,
    relatedConcept: 'conditioning',
  },
  {
    id: 'poly-calls-it-nonlinear',
    pattern: 'Calling polynomial regression a "nonlinear regression" method',
    example: '\\hat{y} = w_0 + w_1 u + w_2 u^2 \\;\\text{is NOT nonlinear-in-parameters}',
    whyWrong: 'It is nonlinear in the INPUT but linear in the parameters θ — that is exactly why the normal equation solves it. "Nonlinear regression" means θ appears nonlinearly (e.g., y = a·e^{bx}). GATE frequently tests this distinction.',
    gateTrap: true,
    relatedConcept: 'basis-expansion-linearity',
  },
];
