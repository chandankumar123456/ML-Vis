// src/topics/mle/mistakes.ts
// Measured anchors (seed 42): p̂=0.707 at n=1000 (k=707); score(p̂)=1.14e−13;
//   σ̂²=0.850428 (÷n) vs σ̂²ᵤ=0.944920 (÷(n−1)) at n=10 — exactly 10/9 apart;
//   L(0.9)=0 vs ℓ(0.9)=−604.8 at n=1000 (underflow).
import type { Mistake } from '../../engine/types';

export const mleMistakes: Mistake[] = [
  {
    id: 'mle-likelihood-is-not-probability',
    pattern: 'Treating the likelihood L(θ) as a probability distribution over θ (integrating it to 1, reading it as "P(θ)")',
    example: '\\int L(\\theta)\\,d\\theta = 1 \\;\\;\\text{(wrong — L is NOT normalized over }\\theta\\text{)}',
    whyWrong: 'L(θ) = Π p(xᵢ|θ) is the probability of the DATA under each θ — a function of θ with no normalization requirement (the total area is not 1 in general). Confusing it with a posterior needs a prior and normalization; confusing it with p(x) forgets the θ-dependence. On the seeded coin run L(p) at p = 0.707 is 2.1465e−263 — clearly not a probability mass of a normalized distribution.',
    gateTrap: true,
    relatedConcept: 'mle-likelihood',
  },
  {
    id: 'mle-biased-variance',
    pattern: 'Writing the Gaussian MLE variance with ÷(n−1) instead of ÷n ("the sample variance, so n−1")',
    example: '\\hat\\sigma^2 = \\frac{1}{n-1}\\sum_i (x_i - \\bar x)^2 \\;\\;\\text{(the UNBIASED estimator — NOT the MLE)}',
    whyWrong: 'The MLE is σ̂² = Σ(xᵢ−μ̂)²/n, forced by ∂ℓ/∂σ² = 0. It is biased: E[σ̂²] = (n−1)/n·σ². Measured at n=10: σ̂² = 0.850428 vs the unbiased 0.944920 — the unbiased is exactly 10/9 × the MLE, a 10% gap that shrinks as 1/n. GATE repeatedly traps on this: the MLE divides by n, period.',
    gateTrap: true,
    relatedConcept: 'mle-gaussian-var',
  },
  {
    id: 'mle-maximize-l-underflow',
    pattern: 'Maximizing the raw product L(θ) numerically instead of the log ℓ(θ), then wondering why the optimizer sees flat zeros',
    example: '\\max_\\theta \\prod_i p(x_i \\mid \\theta) \\;\\;\\text{— at } n=1000 \\text{ this is } 0 \\text{ for most }\\theta\\text{, including the MLE region}',
    whyWrong: 'The product of 1000 factors below 1 underflows: measured L(0.707) = 2.1465e−263, and L(0.9) = 0 EXACTLY in double precision at n=1000. A gradient-based optimizer on L sees an all-zero landscape; the log-likelihood ℓ(0.9) = −604.8 stays finite and smooth. The log is not a convenience — it is numerically mandatory.',
    gateTrap: true,
    relatedConcept: 'mle-loglik',
  },
  {
    id: 'mle-plug-sample-variance',
    pattern: 'Plugging the unbiased sample variance into the Gaussian MLE without discussing the ÷n bias (using S² = Σ/(n−1) as if it were the MLE)',
    example: '\\hat\\sigma^2_{\\text{MLE}} \\stackrel{?}{=} S^2 = \\frac{1}{n-1}\\sum_i (x_i - \\bar x)^2',
    whyWrong: 'The two estimators differ by the factor n/(n−1) and by WHAT they estimate: S² is unbiased for σ², the MLE σ̂² is not. At n=10 the difference is 0.0945 (10%), at n=100 it is 0.0234 (1%), at n=1000 it is 0.0022 (0.1%) — the ÷n-vs-÷(n−1) distinction is a real, measurable quantity the module animates via the biasGap metric. Writing "MLE = S²" is wrong even though both converge to σ².',
    gateTrap: true,
    relatedConcept: 'mle-gaussian-var',
  },
  {
    id: 'mle-consistency-vs-unbiasedness',
    pattern: 'Confusing consistency with unbiasedness: believing the MLE "becomes unbiased" as n grows, or that small-n bias means inconsistency',
    example: 'E[\\hat\\sigma^2_n] = \\frac{n-1}{n}\\sigma^2 \\neq \\sigma^2 \\;\\;\\text{(biased for every finite } n\\text{, yet } \\hat\\sigma^2_n \\xrightarrow{p} \\sigma^2\\text{)}',
    whyWrong: 'Consistency is about convergence in probability to the truth as n → ∞ (the seeded sweep: p̂ error 0.1 → 0.007, μ̂ error 0.940 → 0.071). Unbiasedness is about the expectation at a FIXED n. The ÷n MLE is biased at every finite n but consistent; the ÷(n−1) estimator is unbiased at every n. Both are legitimate — the properties are independent, and GATE tests the distinction.',
    gateTrap: true,
    relatedConcept: 'mle-consistency',
  },
  {
    id: 'mle-local-maxima',
    pattern: 'Assuming every likelihood is concave, so S(θ) = 0 is always the global MLE',
    example: 'S(\\theta) = 0 \\;\\Rightarrow\\; \\theta = \\text{the MLE} \\;\\;\\text{(false for non-concave } \\ell \\text{ — e.g. mixtures)}',
    whyWrong: 'ℓ is concave for Bernoulli, Gaussian and linear families (those are the nice ones), but NOT in general. For the mixture ½N(μ,1)+½N(−μ,1) on {−3,−1,1,3} the measured ℓ has local maxima at μ ≈ ±2 (ℓ = −8.412) around a local minimum at μ = 0 (ℓ = −13.676) — the score vanishes at all three. Numeric MLE (EM, gradient descent) can stick in a local mode; the score condition is necessary, not sufficient.',
    gateTrap: false,
    relatedConcept: 'mle-score',
  },
];