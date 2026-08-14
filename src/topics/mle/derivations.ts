// src/topics/mle/derivations.ts
// Measured anchors (seed 42 — see testCases.test.ts):
//   Bernoulli: k = 707 heads in n = 1000 → p̂ = 0.707; score(p̂) = 1.14e−13;
//     L(p̂) = 2.1465e−263 vs ℓ(p̂) = −604.816 (the underflow that forces the log).
//   Gaussian: n=10 → μ̂ = 0.0597, σ̂² = 0.850428 (÷n), σ̂²ᵤ = 0.944920 (÷(n−1));
//     n=100 → μ̂ = 0.758533, σ̂² = 2.317688.
//   Linear: n=100 → θ̂ = (−0.467261, 1.590870), rss = 71.052727,
//     σ̂² = RSS/n = 0.710527, σ̂²ᵤ = RSS/(n−2) = 0.725028.
import type { Derivation } from '../../engine/types';

export const mleDerivations: Derivation[] = [
  {
    id: 'mle-likelihood-derivation',
    title: 'From Likelihood to Log-Likelihood (and why the log is mandatory)',
    steps: [
      {
        latex: 'L(\\theta) = \\prod_{i=1}^{n} p(x_i \\mid \\theta)',
        justification: 'By independence, the joint probability of the sample factorizes into a product of per-observation terms. L(θ) is the likelihood: a function of θ for the FIXED observed data.',
      },
      {
        latex: '\\ell(\\theta) = \\ln L(\\theta) = \\sum_{i=1}^{n} \\ln p(x_i \\mid \\theta)',
        justification: 'The natural log turns the product into a sum. Because ln is strictly increasing, the maximizer is unchanged: argmax ℓ = argmax L. Measured at n=10 (k=8): BOTH L(p) and ℓ(p) peak at p = 0.8.',
      },
      {
        latex: 'L(0.9) \\text{ at } n=1000 \\;=\\; 0.9^{707}\\cdot 0.1^{293} = 0 \\;\\text{ (double underflow)} \\quad \\text{vs} \\quad \\ell(0.9) = -749.147',
        justification: 'The product of 1000 factors below 1 collapses: even the MAXIMUM likelihood L(p̂) = 2.1465e−263 is far below double precision, and at p = 0.9 the product is exactly 0. The log-likelihood stays a perfectly representable −749.147 — this is why every MLE computation works with ℓ.',
      },
    ],
    derivedFrom: ['mle-likelihood', 'mle-loglik'],
  },
  {
    id: 'mle-bernoulli-derivation',
    title: 'MLE for the Bernoulli (coin flip): score = 0 → p̂ = k/n',
    steps: [
      {
        latex: '\\ell(p) = k \\ln p + (n-k)\\ln(1-p), \\quad k = \\sum_i x_i',
        justification: 'Each flip contributes ln p (heads) or ln(1−p) (tails). With k heads out of n the log-likelihood is the sum above — a sum of two log terms, concave in p.',
      },
      {
        latex: "S(p) = \\frac{d\\ell}{dp} = \\frac{k}{p} - \\frac{n-k}{1-p}",
        justification: 'Differentiate term by term: d/dp ln p = 1/p and d/dp ln(1−p) = −1/(1−p). The score S(p) measures how ℓ changes with p.',
      },
      {
        latex: "\\frac{k}{p} = \\frac{n-k}{1-p} \\;\\Rightarrow\\; k(1-p) = (n-k)p \\;\\Rightarrow\\; \\hat p = \\frac{k}{n}",
        justification: 'Set the score to zero (stationarity). Cross-multiplying gives k − kp = np − kp → k = np → p̂ = k/n. On the seeded stream k = 707, n = 1000 → p̂ = 0.707, and the measured score at p̂ is 1.14e−13 ≈ 0.',
      },
      {
        latex: '\\text{consistency: } |\\hat p - p| = 0.1 \\,(n{=}10) \\;\\to\\; 0.04 \\,(n{=}100) \\;\\to\\; 0.007 \\,(n{=}1000)',
        justification: 'p̂ = k/n is the empirical frequency: by the law of large numbers it converges to the true p = 0.7. The growing-n sweep shows the estimate tightening toward p — the honest demonstration of consistency.',
      },
    ],
    derivedFrom: ['mle-score', 'mle-bernoulli'],
  },
  {
    id: 'mle-gaussian-derivation',
    title: 'MLE for the Gaussian (μ, σ²): both partial scores vanish',
    steps: [
      {
        latex: "\\ell(\\mu, \\sigma^2) = -\\frac{n}{2}\\ln(2\\pi\\sigma^2) - \\frac{1}{2\\sigma^2}\\sum_{i=1}^{n}(x_i - \\mu)^2",
        justification: 'Substitute the Gaussian density into the log-likelihood sum: each ln p(xᵢ|μ,σ²) = −½ln(2πσ²) − (xᵢ−μ)²/(2σ²). Summing gives the two-term form — a constant part (in μ) and a squared-deviation part.',
      },
      {
        latex: "\\frac{\\partial \\ell}{\\partial \\mu} = \\frac{1}{\\sigma^2}\\sum_{i=1}^{n}(x_i - \\mu) = 0 \\;\\Rightarrow\\; \\hat\\mu = \\frac{1}{n}\\sum_i x_i = \\bar x",
        justification: 'Differentiate with respect to μ: d/dμ(xᵢ−μ)² = −2(xᵢ−μ), and the ½·(1/σ²) prefactor leaves (1/σ²)Σ(xᵢ−μ). Setting to zero gives Σxᵢ = nμ → μ̂ = x̄. Measured at n=100: μ̂ = 0.758533 (true μ = 1).',
      },
      {
        latex: "\\frac{\\partial \\ell}{\\partial \\sigma^2} = -\\frac{n}{2\\sigma^2} + \\frac{1}{2\\sigma^4}\\sum_{i=1}^{n}(x_i - \\hat\\mu)^2 = 0 \\;\\Rightarrow\\; \\hat\\sigma^2 = \\frac{1}{n}\\sum_{i=1}^{n}(x_i - \\hat\\mu)^2",
        justification: 'Differentiate with respect to σ²: d/dσ²[−n/2·ln(2πσ²)] = −n/(2σ²) and d/dσ²[−SS/(2σ²)] = SS/(2σ⁴). Setting the sum to zero: SS/σ⁴ = n/σ² → σ̂² = SS/n. The MLE divides by n — NOT n−1 — because it measures spread around the FITTED mean μ̂, which already minimizes those squared deviations.',
      },
      {
        latex: "\\hat\\sigma^2 = 0.850428 \\;\\text{(÷n)} \\;\\text{vs} \\; \\hat\\sigma^2_u = 0.944920 \\;\\text{(÷(n−1)) at } n=10",
        justification: 'The ÷n estimator is biased: E[σ̂²] = (n−1)/n·σ². Measured at n=10 the unbiased estimate is exactly 10/9 × σ̂² — a 10% gap. The gap shrinks as 1/n: 0.01 at n=100, 0.001 at n=1000, visible in the module\'s σ̂²/σ̂²ᵤ matrix cells.',
      },
    ],
    derivedFrom: ['mle-gaussian-mean', 'mle-gaussian-var'],
  },
  {
    id: 'mle-linear-derivation',
    title: 'MLE for linear regression via the matrix score: OLS = MLE',
    steps: [
      {
        latex: "\\ell(\\theta, \\sigma^2) = -\\frac{n}{2}\\ln(2\\pi\\sigma^2) - \\frac{\\|y - X\\theta\\|^2}{2\\sigma^2}",
        justification: 'With Gaussian noise, each residual yᵢ − θᵀxᵢ enters as (yᵢ − θᵀxᵢ)². Summing gives ‖y − Xθ‖²/(2σ²) — the squared-error term that makes MLE and least squares identical.',
      },
      {
        latex: "S(\\theta) = \\frac{\\partial \\ell}{\\partial \\theta} = \\frac{1}{\\sigma^2} X^T (y - X\\theta) = 0",
        justification: 'Differentiating −‖y−Xθ‖²/(2σ²) with respect to θ gives Xᵀ(y−Xθ)/σ². The σ² factors away: stationarity is EXACTLY the least-squares normal equation.',
      },
      {
        latex: "X^T X \\hat\\theta = X^T y \\;\\Rightarrow\\; \\hat\\theta = (X^T X)^{-1} X^T y",
        justification: 'Setting the score to zero: Xᵀy = XᵀXθ. With X = [1 x] the Gram matrix is [[n, Σx],[Σx, Σx²]] and the 2×2 adjugate solves it exactly. Measured at n=100: XᵀX = [[100, 10.494344],[10.494344, 265.756110]], Xᵀy = [−30.030970, 417.879909] → θ̂ = (−0.467261, 1.590870).',
      },
      {
        latex: "\\hat\\sigma^2 = \\frac{\\text{RSS}}{n} = 0.710527 \\;\\text{(÷n)} \\quad \\text{vs} \\quad \\frac{\\text{RSS}}{n-2} = 0.725028 \\;\\text{(unbiased)}",
        justification: 'The residual variance MLE is RSS/n (the ÷n bias, now with n−2 for the unbiased version because two parameters were fitted). The measured score at θ̂ is [8.4e−15, 8.5e−14] ≈ 0 — the MLE and OLS coincide to machine precision, and the fitted line ŷ = −0.467 + 1.591·x tracks the true y = −0.5 + 1.5·x.',
      },
    ],
    derivedFrom: ['mle-score-matrix', 'mle-ols'],
  },
];