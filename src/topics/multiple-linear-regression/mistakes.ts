// src/topics/multiple-linear-regression/mistakes.ts
import type { Mistake } from '../../engine/types';

export const mlrMistakes: Mistake[] = [
  {
    id: 'mlr-bias-column',
    pattern: 'Forgetting the bias column of 1s in the design matrix',
    example: 'X = [x_1\\; x_2] \\text{ instead of } [x_1\\; x_2\\; 1]',
    whyWrong: 'Without the bias column the fitted hyperplane is forced through the origin — systematically biased whenever the true intercept b ≠ 0. Classic GATE trap.',
    gateTrap: true,
    relatedConcept: 'design-matrix',
  },
  {
    id: 'mlr-invert-wrong-matrix',
    pattern: 'Inverting X instead of XᵀX',
    example: '\\theta = X^{-1} y \\;\\neq\\; (X^T X)^{-1} X^T y',
    whyWrong: 'X is n×(d+1), usually non-square and not invertible. Only the square Gram matrix XᵀX can be inverted.',
    gateTrap: true,
    relatedConcept: 'normal-equation',
  },
  {
    id: 'mlr-dimension-mismatch',
    pattern: 'Writing XᵀX with the wrong dimension order (or using XXᵀ)',
    example: 'X^T X \\neq X X^T',
    whyWrong: 'XᵀX is (d+1)×(d+1) (n cancels); XXᵀ is n×n. Using the wrong product gives a matrix of the wrong size and a garbage θ.',
    gateTrap: true,
    relatedConcept: 'matrix-multiplication',
  },
  {
    id: 'mlr-nd-assumption',
    pattern: 'Assuming n > d is always satisfied (or irrelevant)',
    whyWrong: 'When n ≤ d, XᵀX is rank-deficient and the normal equation has infinitely many solutions. GATE asks what happens in the underdetermined regime.',
    gateTrap: true,
    relatedConcept: 'rank',
  },
  {
    id: 'mlr-feature-scaling',
    pattern: 'Running gradient descent without feature scaling in MLR',
    whyWrong: 'If one feature spans [0, 1] and another spans [0, 1000], the gradient is dominated by the large-scale feature and convergence crawls. Standardize (z-score) features first.',
    gateTrap: false,
    relatedConcept: 'gradient-descent',
  },
];
