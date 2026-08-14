// src/topics/mle/questions.ts
// Simulation anchors referenced below (seed 42):
//   coin pTrue 0.7: n=10 → p̂=0.8 (k=8); n=100 → p̂=0.74 (k=74); n=1000 → p̂=0.707 (k=707);
//     L(0.707) = 2.1465e−263 at n=1000; ℓ(0.707) = −604.816; H(0.7) = 0.6108643020548935.
//   gaussian μ=1, σ=1.5: n=100 → μ̂=0.758533, σ̂²=2.317688 (÷n), σ̂²ᵤ=2.341099 (÷(n−1));
//     the ℓ(μ,σ²) grid's argmax row is 4 = the μ̂ row, for every σ² column.
//   linear slope 1.5, intercept −0.5, noise 0.8: n=100 → θ̂=(−0.467261, 1.590870).
import type { Question } from '../../engine/types';

export const mleQuestions: Question[] = [
  {
    id: 'mle-001',
    mode: 'nat',
    prompt: 'An exponential distribution with rate λ has mean 1/λ. For a sample with mean x̄ = 2.5, what is the MLE of λ? (Enter the number; λ̂ = n/Σxᵢ = 1/x̄.)',
    answer: 0.4,
    tolerance: 0.001,
    explanation: 'The exponential log-likelihood is ℓ(λ) = n ln λ − λΣxᵢ. Setting the score S(λ) = n/λ − Σxᵢ = 0 gives λ̂ = n/Σxᵢ = 1/x̄ = 1/2.5 = 0.4. The MLE of the rate is the reciprocal of the sample mean — the same "set score to zero" machinery as p̂ = k/n for the Bernoulli.',
    concepts: ['mle', 'exponential distribution', 'score function'],
    difficulty: 2,
    tags: ['numerical', 'formula', 'indirect'],
  },
  {
    id: 'mle-002',
    mode: 'nat',
    prompt: 'A coin is flipped 1000 times (seed-42 simulator) and lands heads k = 707 times. What is the MLE p̂ of the true head probability? (Enter the number; p̂ = k/n.)',
    answer: 0.707,
    tolerance: 0.001,
    explanation: 'p̂ = k/n = 707/1000 = 0.707. The Bernoulli MLE is the empirical frequency of heads — the closed form that solves S(p) = k/p − (n−k)/(1−p) = 0. The true p = 0.7; the estimate sits 0.007 away at n=1000 (0.1 away at n=10).',
    concepts: ['mle', 'bernoulli', 'empirical frequency'],
    difficulty: 1,
    tags: ['numerical', 'indirect'],
  },
  {
    id: 'mle-003',
    mode: 'gate-mcq',
    prompt: 'GATE-style: why do MLE computations almost always maximize the log-likelihood ℓ(θ) instead of the likelihood L(θ)?',
    options: [
      'Maximizing ℓ gives a DIFFERENT (better) estimate than maximizing L; the log is a correction term',
      'The log turns the product over n factors into a sum, which avoids numerical underflow (measured: L(0.9) = 0 exactly at n=1000 while ℓ(0.9) = −604.8) and makes derivatives tractable; argmax ℓ = argmax L exactly',
      'The log makes ℓ concave even when L is not, so every MLE problem becomes convex',
      'The log divides by n so that ℓ is a per-sample quantity comparable across datasets',
    ],
    answer: 'B',
    explanation: 'ln is strictly increasing, so argmax ℓ = argmax L — the maximizer never moves (measured: at n=10 both peak at p = 0.8). The real reason is numeric and analytic: the product of n factors below 1 collapses — even the maximum L(p̂) at n=1000 is 2.1465e−263, and at p = 0.9 it is exactly 0 in double precision — while ℓ stays finite at −604.8. Sums also differentiate far more easily than products.',
    trapExplanations: {
      A: 'Maximizing ℓ and L give the IDENTICAL maximizer — ln is monotone. No estimate changes.',
      C: 'ℓ is concave for the families in this topic, but NOT in general (mixture models have non-convex ℓ with local maxima) — the log does not buy convexity.',
      D: 'Dividing by n is a separate modeling choice (cross-entropy/nll per sample); ℓ itself is not normalized and is not comparable across n.',
    },
    concepts: ['mle', 'log-likelihood', 'numerical stability'],
    difficulty: 2,
    tags: ['conceptual', 'trap'],
  },
  {
    id: 'mle-004',
    mode: 'gate-mcq',
    prompt: 'GATE-style: for a Gaussian sample, the MLE of the variance is which of the following?',
    options: [
      'Σ(xᵢ − x̄)²/(n−1) — the unbiased sample variance, because the MLE must be unbiased',
      'Σ(xᵢ − x̄)²/n — the BIASED ÷n estimator, forced by the stationarity condition ∂ℓ/∂σ² = 0',
      'Σ(xᵢ − x̄)²/(n+1) — the minimum-MSE estimator',
      'Σ(xᵢ − x̄)²/√n — the asymptotic variance estimator',
    ],
    answer: 'B',
    explanation: 'Setting ∂ℓ/∂σ² = −n/(2σ²) + Σ(xᵢ−μ̂)²/(2σ⁴) = 0 gives σ̂² = Σ(xᵢ−μ̂)²/n — the MLE divides by n, NOT n−1. It is biased (E[σ̂²] = (n−1)/n·σ²): measured at n=10, σ̂² = 0.850428 vs the unbiased 0.944920 — exactly 10/9 apart. The ÷(n−1) estimator is unbiased but is NOT the MLE. The module reports both and animates the gap shrinking as 1/n.',
    trapExplanations: {
      A: 'The unbiased estimator is a fine estimator but it is not the maximum-likelihood one — the MLE divides by n by definition of the stationarity condition.',
      C: 'A minimum-MSE estimator exists (dividing by n+1) but it is neither the MLE nor unbiased — mixing optimality criteria.',
      D: '√n scaling is an asymptotic-normality fact (the MLE is √n-consistent), not the estimator itself.',
    },
    concepts: ['mle', 'variance', 'bias'],
    difficulty: 3,
    tags: ['trap', 'numerical'],
  },
  {
    id: 'mle-005',
    mode: 'matrix',
    prompt: 'Match each score / stationarity expression to the MLE it produces.',
    options: [
      'k/p − (n−k)/(1−p) = 0',
      'Xᵀ(y − Xθ) = 0',
      '−n/(2σ²) + Σ(xᵢ−μ̂)²/(2σ⁴) = 0',
      'n/λ − Σxᵢ = 0',
    ],
    answer: ['Bernoulli MLE p̂ = k/n', 'Linear-regression MLE = normal equation (OLS)', 'Gaussian variance MLE σ̂² = Σ(xᵢ−μ̂)²/n (÷n)', 'Exponential MLE λ̂ = 1/x̄'],
    explanation: 'Every MLE in this topic is found by setting its score to zero. The Bernoulli score gives p̂ = k/n (measured 0.707 on 707/1000 heads); the matrix score Xᵀ(y−Xθ) is the normal equation whose solution is the OLS/MLE θ̂ (measured (−0.467, 1.591) at n=100); the Gaussian variance score forces the ÷n estimator; the exponential score gives the reciprocal of the sample mean.',
    concepts: ['mle', 'score function', 'normal equation'],
    difficulty: 3,
    tags: ['matrix'],
  },
  {
    id: 'mle-006',
    mode: 'visual',
    prompt: 'Run the default coin simulation (family: coin flips, n = 100, seed 42) and scrub to the final snapshot. The fitted p̂ line sits at which value?',
    options: [
      '0.74 — k = 74 heads out of n = 100 flips (the empirical frequency)',
      '0.7 — the TRUE parameter the estimate converges to',
      '0.707 — the n=1000 endpoint of the growing-n sweep',
      '0.8 — the n=10 starting point of the sweep',
    ],
    answer: 'A',
    explanation: 'At n=100 the seeded stream draws k = 74 heads, so the final snapshot\'s fitted line (blue) sits at p̂ = 74/100 = 0.74, while the true-p line (red) stays at 0.7. Scrubbing the growing-n sweep shows the blue line migrating 0.8 → 0.74 → 0.707 → toward 0.7 — the visual consistency story. The nll per sample descends 0.5004 → 0.5731 → 0.6048 toward the true entropy 0.610864.',
    trapExplanations: {
      B: '0.7 is the red TRUE-p reference line; the fitted estimate is the empirical frequency 0.74 at n=100.',
      C: '0.707 is the n=1000 endpoint — the default n=100 run stops at 0.74.',
      D: '0.8 is the first sweep snapshot (n=10, k=8), not the final one.',
    },
    concepts: ['mle', 'consistency', 'visualization'],
    difficulty: 2,
    tags: ['visual'],
  },
  {
    id: 'mle-007',
    mode: 'conceptual-mcq',
    prompt: 'The MLE has the invariance property: if p̂ is the MLE of p, then the MLE of any function g(p) is g(p̂). Which use of this property is correct?',
    options: [
      'The MLE of the Gaussian standard deviation is σ̂ = √σ̂², and the MLE of the Bernoulli odds is p̂/(1−p̂) — apply g to the estimate, never to its expected value',
      'The MLE of the variance is the square of the MLE of the mean, so σ̂² = μ̂²',
      'Invariance lets you compute E[g(p̂)] as g(E[p̂]) — expectations commute with the MLE',
      'The property only holds for linear functions g',
    ],
    answer: 'A',
    explanation: 'Invariance: the maximizer of ℓ in a reparameterized family is the transformed maximizer. It applies to the ESTIMATE: the MLE of √σ² is √(σ̂²), and p̂/(1−p̂) is the MLE of the odds p/(1−p). It does NOT apply to expectations — E[g(θ̂)] ≠ g(E[θ̂]) in general (this is the source of the ÷n bias: σ̂² is unbiased only in the transformed sense, never literally). The module uses it when labelling the fitted curve N(μ̂, σ̂²) and drawing σ̂ = √σ̂².',
    trapExplanations: {
      B: 'σ̂² is a function of the DATA (the residual squares), not of μ̂ — the variance MLE is a different parameter with its own score equation.',
      C: 'Expectations do NOT commute through nonlinear g — exactly why the ÷n MLE is biased in expectation.',
      D: 'Invariance holds for ANY (measurable) function g, including nonlinear ones.',
    },
    concepts: ['mle', 'invariance', 'properties of estimators'],
    difficulty: 3,
    tags: ['conceptual', 'trap'],
  },
];