// src/topics/ridge-regression/testCases.ts
import type { TestCase } from '../../engine/types';

// Truth model shared with multiple-linear-regression: y = 3·x₁ − 2·x₂ + 1.5·x₃ + 1
// (features i.i.d. U[−5,5], so XᵀX is well-conditioned unless the collinear path is used).
//
// NOTE on the λ-sweep simulation: a run emits one snapshot per λ on the grid
// [0, 0.5, …, params.lambda] — the LAST snapshot is always the fit at the slider's
// λ. maxSteps must therefore EXCEED the snapshot count (a run that reaches exactly
// maxSteps is flagged as a step-budget failure by the engine, even if it terminated
// cleanly). Snapshots = lambda / 0.5 + 1.
export const ridgeTestCases: TestCase[] = [
  {
    // Plan spec case 1: ridge(λ=0) ≡ OLS. Well-conditioned i.i.d. data → (XᵀX)
    // invertible, so the λ=0 closed form recovers the truth within tolerance.
    name: 'ridge with λ = 0 recovers the OLS (normal-equation) solution',
    params: { n: 20, nFeatures: 2, noise: 0.05, lambda: 0, seed: 42 },
    maxSteps: 3, // grid [0] → 1 snapshot
    expect: {
      finalMetrics: {
        w1: (v: number) => Math.abs(v - 3) < 0.05,
        w2: (v: number) => Math.abs(v - (-2)) < 0.05,
        b: (v: number) => Math.abs(v - 1) < 0.05,
        mse: (v: number) => v < 0.02, // near-zero residual after exact recovery
      },
    },
  },
  {
    // Plan spec case 3 (ridge side): near-collinear features (x₂ = 2x₁ + ε, ε ~ 1e-5)
    // make XᵀX nearly singular: OLS explodes (‖θ‖ > 1e3 — asserted in the .test.ts)
    // while ridge λ=1 stays small and finite. The λ-sweep passes THROUGH λ=0 (OLS)
    // with a huge-but-finite ‖θ‖, so the run still terminates cleanly.
    name: 'ridge with λ = 1 keeps coefficients finite on near-collinear features',
    params: { n: 20, nFeatures: 2, noise: 0.05, collinear: true, collinearJitter: 1e-5, lambda: 1, seed: 42 },
    maxSteps: 5, // grid [0, 0.5, 1] → 3 snapshots
    expect: {
      finalMetrics: {
        normTheta: (v: number) => v < 10, // λ=1 tames the OLS blow-up (‖θ‖ ≈ 2389)
      },
    },
  },
  {
    // Plan spec case 3 (OLS side): EXACT collinearity (x₂ = 2x₁, ε = 0) makes XᵀX
    // exactly singular → the λ=0 solve returns null → non-finite θ → clean failure.
    // (Ridge λ>0 solves the same data — asserted directly in the .test.ts, because
    //  the λ-sweep cannot pass through the singular λ=0 snapshot.)
    name: 'OLS (λ = 0) fails cleanly on exactly collinear features',
    params: { n: 20, nFeatures: 2, noise: 0.0, collinear: true, lambda: 0, seed: 42 },
    maxSteps: 3, // grid [0] → 1 snapshot, non-finite → failedAtStep = 0
    expect: { converged: false },
  },
];
