// src/topics/mle/failures.ts
// Every numeric claim below is measured from the module:
//   - F1 (underflow): seed-42 coin n=1000, k=707 → L(p̂) = 2.1465e−263,
//     L(0.9) = 0 EXACTLY (double underflow), while ℓ(p̂) = −604.816.
//   - F2 (non-identifiable): points [[1,2],[1,5],[1,8]] (all x = 1) →
//     det(XᵀX) = n·Σx² − (Σx)² = 0; every θ with β₀ + β₁·1 = 5 gives the
//     SAME RSS = 18 and SAME ℓ = −6.9444548034561 (flat likelihood) —
//     initialState throws an honest telemetry failure.
//   - F3 (non-convex): mixture ½N(μ,1)+½N(−μ,1) on {−3,−1,1,3} — measured ℓ has
//     local maxima at μ ≈ ±2 (ℓ = −8.412) around a local minimum at μ = 0
//     (ℓ = −13.676): S(θ) = 0 does not imply the global MLE.
import type { FailureDemo } from '../../engine/types';

export const mleFailureDemos: FailureDemo[] = [
  {
    id: 'mle-fail-underflow',
    title: 'Underflow: the raw likelihood L(p) collapses to 0 — the log is mandatory',
    scenario: 'underflow',
    params: { family: 'coin', n: 1000, seed: 42, pTrue: 0.7 },
    narration: 'Run the coin family at n = 1000 (seed 42): the stream draws k = 707 heads. The likelihood at the MLE is L(p̂) = 0.707⁷⁰⁷·0.293²⁹³ = 2.1465e−263 — thirteen orders of magnitude below double precision\'s comfort zone. At p = 0.9 the product is EXACTLY 0: every factor is < 1 and 1000 of them multiply away to nothing. The log-likelihood has no such problem: ℓ(p̂) = −604.816, a perfectly smooth, finite, maximizable function. This is why the module (and every real MLE implementation) works with ℓ, and why the nllPerSample loss curve stays plottable.',
    whyItBreaks: 'IEEE-754 doubles cannot represent 2.1e−263 reliably and cannot represent the product at all once it rounds to 0. Numerically maximizing L would see an all-zero (or garbage) landscape; the log turns the product into a sum of 1000 well-scaled log-terms. The lesson: always maximize the log-likelihood, and per-sample average (−ℓ/n) when comparing across n.',
  },
  {
    id: 'mle-fail-nonidentifiable',
    title: 'Non-identifiable parameters: a flat likelihood has no unique MLE',
    scenario: 'non-identifiable',
    params: { family: 'linear', points: '[[1,2],[1,5],[1,8]]' },
    narration: 'This failure demo hands the linear family a degenerate design: every point has x = 1, so XᵀX = [[3, 3],[3, 3]] has det = n·Σx² − (Σx)² = 3·3 − 9 = 0 and cannot be inverted. The run fails cleanly via telemetry (converged: false) with an honest "not identifiable" message. The reason is a FLAT likelihood: any line with β₀ + β₁·1 = 5 predicts the same y = 5 for every observation, so RSS = 18 and ℓ = −6.9444548034561 for θ = (5, 0), (3, 2), (0, 5) — infinitely many parameters, one likelihood value.',
    whyItBreaks: 'Identifiability — distinct θ must give distinct distributions — fails when the design matrix is singular. The score equation Xᵀ(y−Xθ) = 0 then has a whole LINE of solutions, and the Fisher information is zero: the MLE does not exist as a point. The fix is a design with spread in x (the normal linear family draws x ~ U(−3, 3) precisely to keep det XᵀX well away from zero) or regularization (ridge adds λI to XᵀX).',
  },
  {
    id: 'mle-fail-nonconvex',
    title: 'Non-convex likelihood: S(θ) = 0 is necessary, not sufficient',
    scenario: 'non-convex',
    params: { family: 'coin', n: 100, seed: 42, pTrue: 0.7 },
    narration: 'CONCEPTUAL DEMO (the mixture is not one of the three simulation families — the numbers below are measured from the module\'s ℓ machinery): a mixture ½N(μ,1) + ½N(−μ,1) observed on {−3, −1, 1, 3} has log-likelihood ℓ(μ) = Σ ln(½·φ(xᵢ−μ) + ½·φ(xᵢ+μ)). Measured: ℓ(2) = −8.412 and ℓ(−2) = −8.412 are both LOCAL MAXIMA, while ℓ(0) = −13.676 is a local MINIMUM. The score S(μ) = ∂ℓ/∂μ vanishes at all three — a numeric optimizer started near 0 can crawl away from the truth in either direction, and the "MLE" is non-unique (the model is symmetric in μ → −μ). The Bernoulli/Gaussian/linear families in the simulator are all concave, which hides this failure from their score equations.',
    whyItBreaks: 'Concavity of ℓ is a property of the FAMILY, not of maximum likelihood itself. Mixtures, latent-variable models and many neural objectives have non-convex ℓ: the stationarity condition S(θ) = 0 is necessary but not sufficient, and multi-modal likelihoods break uniqueness and the consistency story (the asymptotic normality holds per-mode). The honest fix is to report all modes (or symmetry), use multiple restarts, or switch to EM/global optimization with initialization awareness.',
  },
];