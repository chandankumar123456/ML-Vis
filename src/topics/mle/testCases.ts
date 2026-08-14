// src/topics/mle/testCases.ts
import type { TestCase } from '../../engine/types';

// Empirical anchors (all measured by running the simulator — see testCases.test.ts):
//   coin seed 42, pTrue 0.7, n=1000: k = 707, p̂ = 0.707 (pErr 0.007),
//     run = 5 snapshots ([10, 30, 100, 300, 1000]), nll ascends 0.5004 → 0.6048
//     (true entropy H(0.7) = 0.6108643020548935); score(p̂) = 1.14e−13.
//   gaussian seed 42, μ=1, σ=1.5, n=100: μ̂ = 0.7585326658425336,
//     σ̂²(÷n) = 2.317688007784234, σ̂²ᵤ(÷(n−1)) = 2.341098997761853,
//     biasGap = 0.0234109899776187 (biasGapRel = 1/n = 0.01 exactly).
//   gaussian n=10: σ̂² = 0.8504281631796738, σ̂²ᵤ = 0.9449201813107487 →
//     σ̂²ᵤ/σ̂² = 10/9 = n/(n−1) exactly.
//   linear seed 42, slope 1.5, intercept −0.5, noise 0.8, n=100:
//     slopê = 1.590870319456517, intercept̂ = −0.4672611084691402,
//     rss = 71.0527268349805, σ̂² = RSS/n = 0.7105272683498051,
//     σ̂²ᵤ = RSS/(n−2) = 0.7250278248467399; score at θ̂ ≈ 1e−14.
export const mleTestCases: TestCase[] = [
  {
    // Plan case 1: MLE of p on clean Bernoulli samples = empirical frequency.
    // Measured endpoint: k = 707, p̂ = 0.707 on the full seeded n=1000 stream.
    name: 'MLE recovers the true Bernoulli parameter (p̂ = k/n = empirical frequency)',
    params: { family: 'coin', n: 1000, seed: 42, pTrue: 0.7 },
    maxSteps: 500,
    expect: {
      converged: true,
      eventLabels: ['mle-at-requested-n'],
      finalMetrics: {
        k: (v: number) => v === 707,             // measured head count at n=1000
        pHat: (v: number) => v > 0.70 && v < 0.72, // 0.707
        pErr: (v: number) => v < 0.02,           // 0.007
      },
    },
  },
  {
    // Plan case 2: Gaussian MLE recovers the mean, and the variance estimate is
    // the BIASED ÷n estimator (the MLE), not ÷(n−1) — the measured gap is
    // σ̂²ᵤ − σ̂² = 0.0234 at n=100 (the exact ÷n-vs-÷(n−1) identity is asserted
    // in testCases.test.ts at n=10 and n=1000).
    name: 'Gaussian MLE: μ̂ ≈ sample mean, σ̂² = biased ÷n variance (not ÷(n−1))',
    params: { family: 'gaussian', n: 100, seed: 42, muTrue: 1, sigmaTrue: 1.5 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        muHat: (v: number) => v > 0.72 && v < 0.80,     // 0.758533 (μ̂ → 1)
        sigmaHatSq: (v: number) => v > 2.2 && v < 2.45, // 2.317688 (÷n — the MLE)
        sigmaUnbSq: (v: number) => v > 2.3 && v < 2.4,  // 2.341099 (÷(n−1))
        biasGap: (v: number) => v > 0.02 && v < 0.03,   // 0.023411 — the bias gap
      },
    },
  },
  {
    // Plan case 3: log-likelihood maximization — ℓ and L share the argmax.
    // At n=10 the seeded stream gives k = 8, and BOTH L(p) and ℓ(p) peak at
    // p = 0.8 = p̂ on the grid (numeric gradient check lives in .test.ts).
    name: 'log-likelihood maximization: argmax of ℓ and L coincide (numeric gradient check)',
    params: { family: 'coin', n: 10, seed: 42, pTrue: 0.7 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        k: (v: number) => v === 8,               // 8 heads in the first 10 flips
        pHat: (v: number) => v > 0.75 && v < 0.85, // 0.8
        pErr: (v: number) => v < 0.2,            // 0.1 at n=10
      },
    },
  },
  {
    // Plan case 4: MLE = OLS for Gaussian noise — the closed-form MLE solves
    // the normal equation θ̂ = (XᵀX)⁻¹Xᵀy exactly (matrix identity asserted
    // numerically in .test.ts). Measured at n=100 on the seeded design.
    name: 'MLE = OLS for Gaussian noise (normal equation, numeric)',
    params: { family: 'linear', n: 100, seed: 42, slopeTrue: 1.5, interceptTrue: -0.5, noiseSigma: 0.8 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        slopeHat: (v: number) => v > 1.55 && v < 1.63,       // 1.590870
        interceptHat: (v: number) => v > -0.52 && v < -0.42, // −0.467261
        rss: (v: number) => v > 68 && v < 74,                // 71.052727
        sigmaHatSq: (v: number) => v > 0.68 && v < 0.74,     // 0.710527 (RSS/n)
      },
    },
  },
  {
    // Plan case 5: consistency — the estimate improves as n grows. The seeded
    // coin run's |p̂ − p| is 0.1 at n=10 but 0.007 at n=1000 (cross-n assertions
    // in .test.ts); this entry pins the converged endpoint.
    name: 'MLE consistency: seed-42 coin error shrinks from n=10 (0.1) to n=1000 (0.007)',
    params: { family: 'coin', n: 1000, seed: 42, pTrue: 0.7 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        pHat: (v: number) => v > 0.70 && v < 0.72,   // 0.707
        pErr: (v: number) => v < 0.01,               // 0.007 ≪ the n=10 error 0.1
        nllPerSample: (v: number) => v < 0.6109,     // 0.6048 < H(0.7) = 0.610864
      },
    },
  },
];