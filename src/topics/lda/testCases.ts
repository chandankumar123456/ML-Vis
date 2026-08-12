// src/topics/lda/testCases.ts
import type { TestCase } from '../../engine/types';

// ============================================================================
// Hand-derived anchors (toy set — class 0: (0,0)(2,0)(0,2)(2,2), class 1:
// (4,0)(4,2)(6,2)(6,4)). The scatter-matrix math, worked out BY HAND:
//
//   μ₀ = (1,1),  μ₁ = (5,2),  d = μ₁−μ₂ = (4,1)
//   C₀ = (1/4)Σ(x−μ₀)(x−μ₀)ᵀ = [[1,0],[0,1]]            (4 deviations (±1,±1))
//   C₁ = (1/4)Σ(x−μ₁)(x−μ₁)ᵀ = [[1,1],[1,2]]            (deviations
//        (−1,−2),(−1,0),(1,0),(1,2) → s₁₁=4/4, s₁₂=4/4, s₂₂=8/4)
//   S_W = C₀ + C₁ = [[2,1],[1,3]],  det S_W = 2·3−1·1 = 5
//   S_W⁻¹ = (1/5)·[[3,−1],[−1,2]]
//   w = S_W⁻¹ d = (1/5)(3·4 − 1·1, −1·4 + 2·1) = (11/5, −2/5) = (2.2, −0.4)
//   J(w*) = dᵀS_W⁻¹d = (4·11 − 1·2)/5 = 42/5 = 8.4   (= λ, the eigen-link)
//   ŵ = w/‖w‖ = (11,−2)/(5√5),  ŵᵀ(μ₀+μ₁)/2 = 60/(10√5) = 6/√5 ≈ 2.6833
//   Projected within-class variances: s₀² = ŵᵀC₀ŵ = 1; s₁² = ŵᵀC₁ŵ = 17/25.
//   Threshold-rule check (all 8 points): class-0 z-values {0, 22/√125, −4/√125,
//   18/√125} all < 6/√5; class-1 z-values {44/√125, 40/√125, 62/√125, 58/√125}
//   all > 6/√5 → trainError = 0.
//
// These numbers are asserted to 1e-9 in testCases.test.ts against the module's
// own computeLdaStats(TOY_POINTS) and against the computeRun final snapshot.

export const ldaTestCases: TestCase[] = [
  {
    // Plan case 1: projection direction known for 2 classes — the closed-form
    // w = S_W⁻¹(μ₁−μ₂) on the hand-derivable toy set.
    name: 'projection direction known: closed form w = S_W⁻¹(μ₁−μ₂) matches hand math',
    params: { toy: true, seed: 42 },
    maxSteps: 60,
    expect: {
      finalMetrics: {
        rawW1: (v: number) => Math.abs(v - 11 / 5) < 1e-9,
        rawW2: (v: number) => Math.abs(v - (-2 / 5)) < 1e-9,
        jOpt: (v: number) => Math.abs(v - 42 / 5) < 1e-9,
        eigenLambda: (v: number) => Math.abs(v - 42 / 5) < 1e-9,
        threshold: (v: number) => Math.abs(v - 6 / Math.sqrt(5)) < 1e-9,
        withinVar0: (v: number) => Math.abs(v - 1) < 1e-9,
        withinVar1: (v: number) => Math.abs(v - 17 / 25) < 1e-9,
        trainError: (v: number) => v === 0,
        isOptimal: (v: number) => v === 1,
      },
    },
  },
  {
    // Plan case 2: LDA projection maximizes class separation. The final
    // snapshot is the CLOSED-FORM optimum; its J is strictly greater than
    // EVERY sweep direction (measured: grid max J ≈ 3.8616 at θ = 140°, while
    // the closed form gives J(θ*) = 3.8690 at θ* ≈ 138.46° on the default
    // config nPerClass 15, separation 2, covAngleDeg 30, covShape 2, seed 42).
    // The per-angle strict inequality is asserted snapshot-by-snapshot in
    // testCases.test.ts; here the endpoint values are pinned.
    name: 'LDA projection maximizes class separation (final J ≥ measured grid max)',
    params: { nPerClass: 15, separation: 2, covAngleDeg: 30, covShape: 2, seed: 42 },
    maxSteps: 60,
    expect: {
      finalMetrics: {
        isOptimal: (v: number) => v === 1,
        jFisher: (v: number) => Math.abs(v - 3.869000715) < 1e-6,   // J(θ*) measured
        jOpt: (v: number) => Math.abs(v - 3.869000715) < 1e-6,
        gridMaxJ: (v: number) => Math.abs(v - 3.861638793) < 1e-6,  // measured grid maximum
        thetaOptDeg: (v: number) => Math.abs(v - 138.455844) < 1e-3, // analytic optimum (off-grid)
        trainError: (v: number) => Math.abs(v - 0.1333333) < 1e-4,  // measured 2/15
      },
    },
  },
  {
    // Plan case 3 (toy): 2-class LDA = threshold decision rule — zero error on
    // the hand-verified toy set (z vs τ for all 8 points computed in the
    // comments above; measured z-values: {0, 1.968, −0.358, 1.610} vs τ =
    // 2.683 for class 0, {3.935, 3.578, 5.545, 5.188} for class 1 — ALL
    // correctly ordered around τ).
    name: '2-class LDA = decision rule: threshold classifies the toy set perfectly',
    params: { toy: true, seed: 42 },
    maxSteps: 60,
    expect: {
      finalMetrics: {
        trainError: (v: number) => v === 0,
        n: (v: number) => v === 8,
        isOptimal: (v: number) => v === 1,
      },
    },
  },
  {
    // Plan case 3 (Gaussian classes): the threshold rule on the seeded
    // config nPerClass 30, separation 2.5, covAngleDeg 20, covShape 1.5,
    // seed 42 — measured threshold-rule error 0.0333 (1/30 misclassified).
    name: '2-class LDA = decision rule: Gaussian classes classified accurately',
    params: { nPerClass: 30, separation: 2.5, covAngleDeg: 20, covShape: 1.5, seed: 42 },
    maxSteps: 60,
    expect: {
      finalMetrics: {
        trainError: (v: number) => v < 0.2,
        isOptimal: (v: number) => v === 1,
        jFisher: (v: number) => Math.abs(v - 6.777010551) < 1e-6,
      },
    },
  },
  {
    // Plan case 4: reduces to 1-D — the final snapshot carries the ANALYTIC
    // axis (θ* ≈ 169.6952°, not on the 5° grid), and the projected 1-D
    // within-class variances are the analytically-derived values s₀² = 1,
    // s₁² = 17/25. The stronger measured claim — θ* compresses within-class
    // variance BELOW the naive mean-difference axis (whitening; 1.68 vs
    // 2.529 on the toy) and beats it on J — is asserted in testCases.test.ts
    // (the honest "minimal" reading: Fisher optimizes the between/within
    // RATIO; a bare minor-axis direction is even tighter but destroys
    // separation).
    name: 'reduces to 1-D: final axis is the analytic θ* (angleDeg === thetaOptDeg)',
    params: { toy: true, seed: 42 },
    maxSteps: 60,
    expect: {
      finalMetrics: {
        angleDeg: (v: number) => Math.abs(v - 169.695154) < 1e-6,   // analytic θ*, off the 5° grid
        thetaOptDeg: (v: number) => Math.abs(v - 169.695154) < 1e-6,
        withinVar0: (v: number) => Math.abs(v - 1) < 1e-9,
        withinVar1: (v: number) => Math.abs(v - 17 / 25) < 1e-9,
        jOpt: (v: number) => Math.abs(v - 42 / 5) < 1e-9,
        trainError: (v: number) => v === 0,
        isOptimal: (v: number) => v === 1,
      },
    },
  },
];