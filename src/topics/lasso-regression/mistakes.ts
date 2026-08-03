// src/topics/lasso-regression/mistakes.ts
import type { Mistake } from '../../engine/types';

export const lassoMistakes: Mistake[] = [
  {
    id: 'lasso-uniform-shrink',
    pattern: 'Thinking lasso shrinks ALL coefficients uniformly like ridge',
    example: 'w_1 = 8 - 1 = 7,\\; w_4 = 1.2 - 1 = 0.2 \\;\\text{ vs }\\; S(1.2, 1) = 0',
    whyWrong: 'Lasso applies the soft-threshold S(z, λ) = sign(z)·max(|z| − λ, 0): every coefficient shifts by a CONSTANT λ, and any |z| ≤ λ is clamped to EXACTLY zero. Small coefficients vanish; large ones are not shrunk proportionally — unlike ridge, which scales every weight by the same factor.',
    gateTrap: true,
    relatedConcept: 'soft-threshold',
  },
  {
    id: 'lasso-lambda-direction',
    pattern: 'Confusing the direction of λ (more λ = more shrinkage)',
    example: '\\lambda\\uparrow\\; \\Rightarrow\\; \\text{more coefficients become } 0,\\text{ not fewer}',
    whyWrong: 'Larger λ subtracts more from every |z| before clamping, so the soft-threshold zeros MORE coordinates. The coefficient path always shrinks toward zero as λ grows — at λ beyond every |ρ/n| the model degenerates to ŷ = ȳ.',
    gateTrap: true,
    relatedConcept: 'lasso-objective',
  },
  {
    id: 'lasso-ridge-zeros',
    pattern: 'Thinking ridge regression also produces exact zeros',
    example: '\\theta^\\star_{\\text{ridge}} = (X^T X + \\lambda I)^{-1} X^T y \\;\\Rightarrow\\; \\theta_j \\neq 0 \\text{ almost surely}',
    whyWrong: 'The L2 penalty λ‖θ‖² is smooth — its gradient vanishes only at θ = 0 exactly, a measure-zero event. Ridge shrinks proportionally but never selects: |θ_j| → 0 only as λ → ∞. Exact zeros are the signature of the L1 (non-differentiable) penalty.',
    gateTrap: true,
    relatedConcept: 'ridge-objective',
  },
  {
    id: 'lasso-penalizes-bias',
    pattern: 'Penalizing the intercept in the L1 term',
    example: '\\lambda (|w_1| + |w_2| + |b|) \\;\\text{ instead of }\\; \\lambda (|w_1| + |w_2|)',
    whyWrong: 'The bias is NOT regularized: features are z-scored (centered), so the intercept absorbs the mean of y and shrinking it would bias every prediction. GATE asks about "lasso with penalty on all parameters" — the standard convention leaves b unpenalized.',
    gateTrap: true,
    relatedConcept: 'lasso-objective',
  },
  {
    id: 'lasso-threshold-scale',
    pattern: 'Soft-thresholding the raw correlation ρ instead of the normalized ρ/n, or skipping feature standardization',
    example: 'w_j \\leftarrow S(\\rho_j,\\ \\lambda) \\;\\text{ — wrong scale; correct is } S(\\rho_j/n,\\ \\lambda)',
    whyWrong: 'The threshold must be applied to z = ρⱼ/n (the OLS coordinate solution), not to the raw sum ρⱼ which grows with n. And on UNstandardized features, σⱼ differences make the penalty treat features unequally — standardize first so Σᵢzᵢⱼ² = n.',
    gateTrap: false,
    relatedConcept: 'cd-update',
  },
];
