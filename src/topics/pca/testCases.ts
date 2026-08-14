// src/topics/pca/testCases.ts
import type { TestCase } from '../../engine/types';

// Empirical anchors (ALL measured by running the actual module — see
// scratch.test.ts measurements and testCases.test.ts):
//   default (n 40, corr 0.7, rotDeg 30, noise 0.15, seed 42): Σ =
//     [[0.37011827663219543, 0.37911198787583555],[0.37911198787583555, 1.8463033092217696]],
//     λ₁ = 1.9379736879527472, λ₂ = 0.2784478979012177, total = 2.216421585853965,
//     ratio₁ = 0.8743705170178919, ratio₂ = 0.1256294829821081,
//     PC1 = (0.23502957205216105, 0.9719882202274769) at 76.40663404901252°,
//     v₁·v₂ = 0 exactly (90°-rotation construction),
//     reconErrK1 = 0.2784478979012177 (= λ₂), reconErrK2 = 0;
//     run = 37 snapshots (36 sweep + 1 closed-form final);
//     grid sweep peaks at 75° with variance 1.9369736563193656 — λ₁ exceeds it
//     by 0.00100; a dense 0.25° scan over all 720 directions tops out at
//     1.9379692812332503 — λ₁ exceeds it by 4.4067e-6.
//   rotDeg 80 (centering contrast): centered PC1 at 127.15210735118922° vs
//     RAW (uncentered) PC1 at 33.29706869064782° — the raw PC points at the
//     data mean of that draw, μ = (2.3841, 1.6119) (mean direction ≈ 34.06°);
//     raw PC1 explains only 0.12583 of the CENTERED variance (true PC1: 0.87758).
//   seed 7: λ₁ = 2.23747379244006, λ₂ = 0.2641044311347427, PC1 at 68.85°.
//   cfg2 (n 60, corr 0.3, rotDeg 60, noise 0.25, seed 123): λ₁ = 1.4315445585568694,
//     λ₂ = 0.678435059535849, PC1 at 112.13°.
//   near-degenerate (noise 0, corr 0.98): λ₂ = 0.01600865622759695,
//     ratio₂ = 0.005884164837050432 — PC2 carries ~0.6% of the variance.
export const pcaTestCases: TestCase[] = [
  {
    // Plan case 1: first PC maximizes variance — λ₁ ≥ variance along EVERY
    // candidate direction. The data-driven check asserts the measured λ₁; the
    // full "≥ every grid angle AND every 0.25° direction" sweep is in .test.ts.
    name: 'first PC maximizes variance: λ₁ = 1.938 ≥ uᵀΣu along every sweep direction',
    params: { n: 40, corr: 0.7, rotDeg: 30, noise: 0.15, seed: 42 },
    maxSteps: 500,
    expect: {
      converged: true,
      eventLabels: ['exact-2x2-pca-solution'],
      finalMetrics: {
        lambda1: (v: number) => v > 1.93 && v < 1.95,        // measured 1.9379736879527472
        axisVariance: (v: number) => v > 1.93 && v < 1.95,   // final lands ON λ₁ (loss peak)
        explainedRatio1: (v: number) => v > 0.85 && v < 0.9, // measured 0.8744
        isOptimal: (v: number) => v === 1,
      },
    },
  },
  {
    // Plan case 2: PCs orthogonal — v₁·v₂ = 0 within 1e-6 (exact by the 90°
    // rotation construction; measured exactly 0, asserted to 1e-12 in .test.ts).
    name: 'PCs are orthogonal: v₁·v₂ = 0 (exact 90°-rotation construction)',
    params: { n: 40, corr: 0.7, rotDeg: 30, noise: 0.15, seed: 42 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalAlgorithm: {
        mode: 'pca-optimal',
        isOptimal: 1,
      },
    },
  },
  {
    // Plan case 3: eigenvalues = explained variance — λ_k equals the empirical
    // variance of the data projected on v_k (measured varianceAlong(v₁) =
    // 1.9379736879527472 = λ₁ exactly; asserted to 1e-9 in .test.ts).
    name: 'eigenvalues = explained variance: λ_k = variance of projections on v_k',
    params: { n: 40, corr: 0.7, rotDeg: 30, noise: 0.15, seed: 42 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        lambda1: (v: number) => Math.abs(v - 1.9379736879527472) < 1e-9,
        lambda2: (v: number) => Math.abs(v - 0.2784478979012177) < 1e-9,
        reconErrK1: (v: number) => Math.abs(v - 0.2784478979012177) < 1e-9, // = λ₂
        reconErrK2: (v: number) => Math.abs(v) < 1e-9,
      },
    },
  },
  {
    // Plan case 4: centering matters — the run converges on a clearly-rotated
    // config (rotDeg 80: centered PC1 at 127.15°); the deep centered-vs-RAW
    // contrast (raw PC1 at 33.30°, pointing at the mean) is measured in .test.ts.
    name: 'centering matters: uncentered covariance gives different (wrong) PCs',
    params: { n: 40, corr: 0.7, rotDeg: 80, noise: 0.15, seed: 42 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        angleDeg: (v: number) => v > 124 && v < 130, // centered PC1 ≈ 127.15°
        isOptimal: (v: number) => v === 1,
      },
    },
  },
  {
    // Plan case 5: reconstruction — projecting to k PCs and back: error =
    // Σ_{j>k} λ_j (k=1 → λ₂ = 0.27845, k=2 → 0), asserted exactly in .test.ts.
    name: 'reconstruction: k=1 error = λ₂ (dropped eigenvalue), k=2 error = 0',
    params: { n: 40, corr: 0.7, rotDeg: 30, noise: 0.15, seed: 42 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        reconErrK1: (v: number) => v > 0.27 && v < 0.29, // measured 0.27845
        reconErrK2: (v: number) => Math.abs(v) < 1e-9,
      },
    },
  },
  {
    // Extra: a second seed — the same identities hold at measured seed-7 values.
    name: 'seed 7: eigenvalues and PC angle measured at their own values',
    params: { n: 40, corr: 0.7, rotDeg: 30, noise: 0.15, seed: 7 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        lambda1: (v: number) => Math.abs(v - 2.23747379244006) < 1e-6,
        angleDeg: (v: number) => Math.abs(v - 68.85395927722712) < 1e-3,
        isOptimal: (v: number) => v === 1,
      },
    },
  },
  {
    // Extra: a second config (different n/corr/rot/noise/seed) — measured values.
    name: 'cfg2 (n 60, corr 0.3, rotDeg 60, noise 0.25, seed 123): measured anchors',
    params: { n: 60, corr: 0.3, rotDeg: 60, noise: 0.25, seed: 123 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        lambda1: (v: number) => Math.abs(v - 1.4315445585568694) < 1e-6,
        lambda2: (v: number) => Math.abs(v - 0.678435059535849) < 1e-6,
        angleDeg: (v: number) => Math.abs(v - 112.13177090203905) < 1e-3,
      },
    },
  },
  {
    // Extra: near-degenerate data (noise 0, |corr| → 1) — the closed form stays
    // exact but λ₂ collapses toward 0 (PC2 meaningless, honest warning).
    name: 'near-degenerate: noise 0, corr 0.98 → λ₂ = 0.0160, PC2 carries 0.6%',
    params: { n: 40, corr: 0.98, rotDeg: 30, noise: 0, seed: 42 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        lambda2: (v: number) => Math.abs(v - 0.01600865622759695) < 1e-6,
        explainedRatio2: (v: number) => Math.abs(v - 0.005884164837050432) < 1e-6,
      },
    },
  },
];