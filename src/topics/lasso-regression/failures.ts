// src/topics/lasso-regression/failures.ts
import type { FailureDemo } from '../../engine/types';

export const lassoFailureDemos: FailureDemo[] = [
  {
    id: 'lasso-fail-correlated',
    title: 'Correlated Features: Lasso Selects Arbitrarily',
    scenario: 'correlated-features',
    params: { n: 40, nFeatures: 4, noise: 0.5, lambda: 1.0, correlated: true, seed: 42 },
    narration: 'Features 2–4 are generated as x₁ plus tiny noise (corr ≈ 0.9999), so they carry nearly the same signal. Lasso does not error — it just breaks the tie using tiny noise asymmetries: one of the near-duplicate features keeps a large coefficient while the others are soft-thresholded to exactly 0. The chosen feature is arbitrary: change the seed and a different one wins.',
    whyItBreaks: 'Nearly identical normalized correlations |ρⱼ/n| — the soft-threshold clamps everything below λ and keeps whichever correlation is slightly larger in magnitude. The selection is not uniquely determined by the data, so the model is unstable (different seeds → different chosen feature).',
  },
  {
    id: 'lasso-fail-lambda-large',
    title: 'λ Too Large: Every Coefficient Zeroed (Underfit)',
    scenario: 'lambda-too-large',
    params: { n: 40, nFeatures: 4, noise: 0.5, lambda: 10, seed: 42 },
    narration: 'λ = 10 exceeds every normalized correlation (the largest standardized weight is ≈ 8.7), so every feature is soft-thresholded to exactly 0. The model degenerates to ŷ = b̄ = ȳ: MSE balloons to the full variance of y and the fit is useless — an underfit by over-regularization.',
    whyItBreaks: 'The soft-threshold S(z, λ) returns 0 for every |z| ≤ λ. When λ is above the largest |ρ/n|, the L1 penalty dominates the quadratic loss entirely — the objective is minimized by the all-zero weight vector.',
  },
];
