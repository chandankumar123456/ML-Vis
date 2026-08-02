// src/topics/multiple-linear-regression/failures.ts
import type { FailureDemo } from '../../engine/types';

export const mlrFailureDemos: FailureDemo[] = [
  {
    id: 'mlr-fail-collinearity',
    title: 'Multicollinearity: XᵀX becomes singular',
    scenario: 'collinearity',
    params: { n: 20, nFeatures: 2, noise: 0.0, collinear: true, useNormalEquation: true, seed: 42 },
    narration: 'Features are generated with x₂ = 2x₁ exactly, so the design columns are linearly dependent. XᵀX is singular: the normal equation (XᵀX)⁻¹Xᵀy has no unique answer — the run fails cleanly with a non-finite θ.',
    whyItBreaks: 'Rank-deficient design matrix. det(XᵀX) = 0, so the inverse does not exist. validateParams flags "Features are collinear" before the run starts.',
  },
  {
    id: 'mlr-fail-underdetermined',
    title: 'Underdetermined: n < d + 1',
    scenario: 'underdetermined',
    params: { n: 2, nFeatures: 3, noise: 0.0, useNormalEquation: true, seed: 42 },
    narration: 'Only n = 2 samples for d = 3 features plus bias: 4 unknowns, 2 equations. X is 2×4 with rank ≤ 2, so XᵀX (4×4) is singular — infinitely many θ interpolate the 2 points with zero error.',
    whyItBreaks: 'n < d + 1 ⇒ XᵀX is rank-deficient by the rank–nullity bound. validateParams flags "Underdetermined: n < d + 1".',
  },
];
