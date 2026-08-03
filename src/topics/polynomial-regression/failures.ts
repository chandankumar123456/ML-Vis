// src/topics/polynomial-regression/failures.ts
import type { FailureDemo } from '../../engine/types';

export const polyFailureDemos: FailureDemo[] = [
  {
    id: 'poly-fail-degree-30',
    title: 'Degree 30: the normal equation becomes numerically unstable',
    scenario: 'numerical-instability',
    params: { degree: 30, nTrain: 40, nTest: 20, noise: 0.5, seed: 42, fitOn: 'train' },
    narration: 'Even with 40 training points (degree 30 → 31 parameters is formally solvable), the Vandermonde Gram matrix ΦᵀΦ is numerically singular: its condition number is beyond double precision at these degrees (κ ≳ 10¹⁶ for the normalized basis, and grows super-exponentially — the raw power basis reaches ~10²⁵). The inversion fails and the run terminates cleanly with a non-finite θ.',
    whyItBreaks: 'κ(ΦᵀΦ) grows super-exponentially with degree (Vandermonde matrices are among the worst-conditioned matrices known). Pivots drop below machine-relative tolerance, so the solve returns NaN — the sandbox flags the run as failed. This is exactly why the simulation caps the degree slider at 15 and why ridge (ΦᵀΦ + λI) exists.',
  },
  {
    id: 'poly-fail-runge',
    title: "Runge's phenomenon: wild oscillations at the extremes",
    scenario: 'runge',
    params: { degree: 15, nTrain: 30, nTest: 20, noise: 0.5, seed: 42, fitOn: 'train' },
    narration: 'The degree-15 fit interpolates every training point (train MSE ≈ 0.04) but oscillates violently near the edges of the range — this is Runge-type behavior: high-degree polynomials cannot represent the flat center and steep edges of real data simultaneously. The held-out test MSE explodes to ≈ 680.',
    whyItBreaks: 'The fitted polynomial must satisfy many constraints, so it develops sharp wiggles whose amplitude grows toward the boundary of the interval. The training error stays tiny (it has memorized the noise) while out-of-sample predictions become garbage — the coefficient explosion (|w| ≈ 28000) is the visible symptom.',
  },
];
