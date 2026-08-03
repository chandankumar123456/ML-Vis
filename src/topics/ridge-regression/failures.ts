// src/topics/ridge-regression/failures.ts
import type { FailureDemo } from '../../engine/types';

export const ridgeFailureDemos: FailureDemo[] = [
  {
    id: 'ridge-fail-lambda-large',
    title: 'λ → ∞: every coefficient collapses to 0 (underfit)',
    scenario: 'lambda-too-large',
    params: { n: 25, nFeatures: 2, noise: 0.5, lambda: 50, seed: 42 },
    narration: 'At λ = 50 every eigenvalue of XᵀX + λI is at least 50, so the closed form shrinks every coefficient toward 0: ‖θ‖ → 0 and the fit collapses to predicting a constant. Training error rises steeply and the test point is badly missed — the "too much shrinkage" end of the path.',
    whyItBreaks: 'The penalty term λ‖θ‖² dominates the objective once λ ≫ eigenvalues of XᵀX. The bias term Bias² = ‖(R_λ − I)θ_true‖² saturates at ‖θ_true‖², so the model underfits: it behaves like a mean-predicting constant model.',
  },
  {
    id: 'ridge-fail-unscaled-features',
    title: 'Unscaled features → uneven (misleading) shrinkage',
    scenario: 'unscaled-features',
    params: { n: 25, nFeatures: 2, noise: 0.5, lambda: 2, seed: 42 },
    narration: 'The penalty ‖θ‖² is not scale-invariant: if x₁ spans [0, 1] and x₂ spans [0, 1000], the same λ shrinks the x₂-coefficient by a completely different relative amount than the x₁-coefficient. The shrinkage path becomes meaningless and the reported ‖θ‖ mixes incompatible units. Standardize (z-score) features first so λ means the same thing for every coefficient.',
    whyItBreaks: 'λ‖θ‖² penalizes large coefficients regardless of the feature scale that produced them. A coefficient on a tiny-scale feature must be large to matter, so it gets shrunk unfairly; the closed form is still "correct" but the λ path is distorted. The fix is per-feature standardization (or centering with scale normalization) before fitting.',
  },
];
