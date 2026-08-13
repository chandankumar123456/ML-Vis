// src/topics/lda/derivations.ts
import type { Derivation } from '../../engine/types';

export const ldaDerivations: Derivation[] = [
  {
    id: 'lda-fisher-eigen',
    title: 'Maximizing the Fisher Criterion → the Generalized Eigenproblem',
    steps: [
      {
        latex: 'J(w) = \\frac{w^T S_B w}{w^T S_W w}',
        justification: 'The Fisher criterion: the ratio of between-class spread to within-class spread along w. It is scale-invariant — only the direction of w matters — so we are free to fix a normalization below.',
      },
      {
        latex: '\\text{fix } w^T S_W w = 1, \\;\\text{ then maximize } w^T S_B w',
        justification: 'Scale-invariance lets us pin the denominator to 1 and maximize the numerator instead — a constrained optimization of a quadratic form.',
      },
      {
        latex: '\\mathcal{L} = w^T S_B w - \\lambda (w^T S_W w - 1) \\;\\Rightarrow\\; \\tfrac{\\partial}{\\partial w} \\mathcal{L} = 2S_B w - 2\\lambda S_W w = 0',
        justification: 'Lagrange multipliers with constraint wᵀS_Ww = 1. Taking the gradient (S_B and S_W symmetric) gives the stationarity condition — λ is the generalized eigenvalue.',
      },
      {
        latex: 'S_B w = \\lambda S_W w \\;\\iff\\; S_W^{-1} S_B w = \\lambda w',
        justification: 'Rearranged: w must be an eigenvector of the discrimination matrix S_W⁻¹S_B. The associated λ is the value of J at that direction: multiplying the eigen-equation by wᵀ gives wᵀS_Bw = λ·wᵀS_Ww = λ·(normalization) — so λ = J(w) exactly, which is why the simulation can report "the eigenvalue IS the Fisher value".',
      },
    ],
    derivedFrom: ['lda-fisher-criterion'],
  },
  {
    id: 'lda-two-class-closed-form',
    title: '2-Class Closed Form: w = S_W⁻¹(μ₁−μ₂)',
    steps: [
      {
        latex: 'S_B = (\\mu_1 - \\mu_2)(\\mu_1 - \\mu_2)^T \\;\\Rightarrow\\; \\operatorname{rank}(S_B) = 1',
        justification: 'The between-class scatter is an outer product of one vector — rank 1. Therefore S_W⁻¹S_B is also rank 1 and has exactly ONE nonzero eigenvalue (2-class LDA = one projection axis).',
      },
      {
        latex: 'S_B w = (\\mu_1 - \\mu_2)\\underbrace{(\\mu_1 - \\mu_2)^T w}_{\\text{scalar}} = c\\,(\\mu_1 - \\mu_2)',
        justification: 'Feeding any w into S_B returns a scalar multiple of the mean difference. So if w is an eigenvector, S_W⁻¹S_Bw = c·S_W⁻¹(μ₁−μ₂) must be parallel to w.',
      },
      {
        latex: 'w_* = S_W^{-1}(\\mu_1 - \\mu_2)',
        justification: 'The only direction that survives the eigen-equation is proportional to S_W⁻¹(μ₁−μ₂). This is the closed form the final simulation step evaluates exactly (no grid search needed) — the plan\'s "known direction" test case.',
      },
      {
        latex: '\\lambda_{\\max} = (\\mu_1 - \\mu_2)^T S_W^{-1}(\\mu_1 - \\mu_2) = J(w_*)',
        justification: 'The single nonzero eigenvalue (a scalar — the trace of the rank-1 S_W⁻¹S_B). The tests verify dᵀS_W⁻¹d equals the directly computed J(θ*) to 1e-9, and the simulations surface it as the eigen-Link.',
      },
    ],
    derivedFrom: ['lda-fisher-eigen'],
  },
  {
    id: 'lda-threshold-rule',
    title: 'From Shared-Covariance Gaussians to a Linear Threshold Rule',
    steps: [
      {
        latex: '\\ln \\frac{P(C_1 \\mid x)}{P(C_0 \\mid x)} = \\ln \\frac{P(C_1)}{P(C_0)} + \\frac12 (x-\\mu_0)^T\\Sigma^{-1}(x-\\mu_0) - \\frac12 (x-\\mu_1)^T\\Sigma^{-1}(x-\\mu_1)',
        justification: 'Bayes\' rule in log space. Because the covariance Σ is SHARED, the normalizing constants cancel — this is the step that fails for QDA.',
      },
      {
        latex: '\\ln \\frac{P(C_1 \\mid x)}{P(C_0 \\mid x)} = \\log \\frac{P(C_1)}{P(C_0)} + (\\mu_1 - \\mu_0)^T \\Sigma^{-1} x - \\tfrac12 \\mu_1^T\\Sigma^{-1}\\mu_1 + \\tfrac12 \\mu_0^T\\Sigma^{-1}\\mu_0',
        justification: 'Expanding the two quadratic forms: the xᵀΣ⁻¹x terms cancel exactly (shared Σ), leaving a LINEAR function of x — the decision rule is a hyperplane, equivalently a threshold on the 1-D projection.',
      },
      {
        latex: '\\hat{y} = 1 \\iff \\hat{w}^T x > \\tau, \\qquad \\tau = \\hat{w}^T \\frac{\\mu_0 + \\mu_1}{2} \\;\\text{ (equal priors)}',
        justification: 'With equal priors the log-ratio crosses 0 at the midpoint of the projected means. The 2-D decision boundary z = τ is a line perpendicular to ŵ, and the simulation\'s decision-boundary view renders exactly this (wx·x + wy·y + b > 0 with b = −τ).',
      },
      {
        latex: '\\text{unequal priors: } \\tau = \\hat{w}^T \\frac{\\mu_0 + \\mu_1}{2} - \\frac{\\ln(P(C_1)/P(C_0))}{\\|w\\|}, \\qquad w = \\Sigma^{-1}(\\mu_1 - \\mu_0)',
        justification: 'The prior ratio shifts the threshold along the axis (the log-odds intercept moves). With P(C₁) > P(C₀) the correction is SUBTRACTED, pushing τ toward class 0 so more of the axis is claimed by the more probable class 1. The scaling uses the norm of the UN-normalized direction w = Σ⁻¹(μ₁−μ₀), not ‖ŵᵀ(μ₁−μ₀)‖ — the two differ unless Σ ∝ I. The simulator keeps equal priors so the midpoint τ is the Bayes-optimal cut — documented so the trap question on priors has an answer.',
      },
    ],
    derivedFrom: ['lda-two-class-closed-form'],
  },
];