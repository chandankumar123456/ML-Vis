// src/topics/naive-bayes/questions.ts
import type { Question } from '../../engine/types';

export const nbQuestions: Question[] = [
  {
    // The classic GATE table problem: multiply the per-feature likelihoods, then
    // weigh by the prior, then normalize. HAND-VERIFIED: P(sweet,big|orange) = 0.7·0.8
    // = 0.56, numerator 0.56·0.6 = 0.336; P(sweet,big|apple) = 0.3·0.2 = 0.06,
    // numerator 0.06·0.4 = 0.024; evidence = 0.36 → posterior = 0.336/0.36 = 14/15.
    id: 'nb-001',
    mode: 'nat',
    prompt: 'A naive Bayes classifier labels fruits as orange or apple. P(orange) = 0.6, P(apple) = 0.4. From training data: P(sweet|orange) = 0.7, P(sweet|apple) = 0.3, P(big|orange) = 0.8, P(big|apple) = 0.2. A fruit is sweet AND big. What is P(orange | sweet, big)? (answer as a decimal)',
    answer: 14 / 15,
    tolerance: 0.01,
    explanation: 'P(sweet,big|orange) = 0.7 × 0.8 = 0.56, so the orange numerator is 0.56 × 0.6 = 0.336. P(sweet,big|apple) = 0.3 × 0.2 = 0.06, numerator 0.06 × 0.4 = 0.024. The evidence is 0.336 + 0.024 = 0.36, so P(orange|sweet,big) = 0.336/0.36 = 14/15 ≈ 0.9333.',
    concepts: ['naive-bayes', 'bayes-theorem', 'posterior'],
    difficulty: 2,
    tags: ['numerical', 'formula'],
  },
  {
    id: 'nb-002',
    mode: 'conceptual-mcq',
    prompt: 'Why is naive Bayes called "naive"?',
    options: [
      'It uses approximate Bayesian inference that never gives exact answers',
      'It assumes features are conditionally independent given the class — an assumption that rarely holds exactly in real data',
      'It cannot handle numerical features',
      'It ignores the class prior probabilities',
    ],
    answer: 'B',
    explanation: 'The only approximation is P(x|C) = Π P(x_j|C): features are assumed independent given the class. Real features are often correlated, so the assumption is "naive" — yet the classifier still works surprisingly well in many domains (e.g. text).',
    trapExplanations: {
      A: 'Bayes theorem itself is exact; only the likelihood factorization is approximate.',
      C: 'Gaussian naive Bayes handles continuous features directly (per-class mean/variance).',
      D: 'It uses the prior — the naive part is the independence assumption, not the prior.',
    },
    concepts: ['naive-bayes', 'independence'],
    difficulty: 1,
    tags: ['conceptual'],
  },
  {
    id: 'nb-003',
    mode: 'gate-mcq',
    prompt: 'In categorical naive Bayes without smoothing, a test sample has the value x₂ = 3, but class 0 never saw x₂ = 3 during training. What happens to P(x|C = 0)?',
    options: [
      'It is small but positive — rare values still contribute a little evidence',
      'It is exactly 0 — one unseen value annihilates the whole class likelihood, so class 0 is never predicted for this sample',
      'It stays 1 because unseen values carry no information',
      'It equals P(x₂ = 3) from the test set, which is the correct estimate',
    ],
    answer: 'B',
    explanation: 'P(x₂=3|C=0) = count(3)/n₀ = 0, and the naive likelihood is a PRODUCT, so P(x|C=0) = 0 — one zero factor kills the class. Laplace smoothing (count+α)/(n₀+αV) fixes this by keeping every value strictly positive.',
    trapExplanations: {
      A: 'Without smoothing the count is exactly 0, not small — the product is exactly 0.',
      C: 'Unseen values carry information: they are evidence AGAINST the class under a product model.',
      D: 'Test-set statistics are unavailable in supervised classification; smoothing estimates it from the training counts.',
    },
    concepts: ['naive-bayes', 'laplace-smoothing', 'zero-probability'],
    difficulty: 2,
    tags: ['trap'],
  },
  {
    id: 'nb-004',
    mode: 'gate-mcq',
    prompt: 'Two classes have equal likelihoods for a test sample: P(x|C₀) = P(x|C₁). The training set has 90% class 0 and 10% class 1. What does the Bayes-optimal prediction say?',
    options: [
      'Tie — predict either class with equal confidence',
      'Class 0 — the prior P(C₀) = 0.9 dominates when likelihoods are equal',
      'Class 1 — the rare class must be upweighted to balance the data',
      'Cannot decide without the evidence P(x)',
    ],
    answer: 'B',
    explanation: 'P(C|x) ∝ P(x|C)P(C). Equal likelihoods leave only the prior: P(C₀|x) : P(C₁|x) = 0.9 : 0.1, so the posterior favors class 0 nine to one. Ignoring the prior is a classic mistake — the prior is part of Bayes theorem.',
    trapExplanations: {
      A: 'Equal likelihoods do NOT mean equal posteriors — the priors break the tie.',
      C: 'There is no balancing needed at prediction time; the prior reflects the true class frequency.',
      D: 'P(x) is the same for both classes and cannot decide between them.',
    },
    concepts: ['naive-bayes', 'prior', 'bayes-theorem'],
    difficulty: 2,
    tags: ['trap', 'conceptual'],
  },
  {
    // Feature×class table problem: Laplace-smoothed likelihood from counts.
    // HAND-VERIFIED: class 1 has 9 values of x₂=0 and 1 of x₂=1 (n₁ = 10); V = 2;
    // with α=1 → (9+1)/(10+2) = 10/12 = 5/6.
    id: 'nb-005',
    mode: 'nat',
    prompt: 'A categorical naive Bayes model uses one feature x₂ with values {0, 1}. The training count table is: class 0 — x₂=0: 4, x₂=1: 6; class 1 — x₂=0: 9, x₂=1: 1. With Laplace smoothing α = 1, what is P(x₂ = 0 | C = 1)? (answer as a decimal)',
    answer: 5 / 6,
    tolerance: 0.01,
    explanation: 'Class 1 has n₁ = 9 + 1 = 10 samples and the feature has V = 2 distinct values. Laplace smoothing gives P(x₂=0|C=1) = (count + α)/(n₁ + α·V) = (9 + 1)/(10 + 2) = 10/12 = 5/6 ≈ 0.8333. Without smoothing it would be 9/10 = 0.9.',
    concepts: ['naive-bayes', 'laplace-smoothing', 'feature-class-table'],
    difficulty: 2,
    tags: ['numerical', 'matrix', 'formula'],
  },
  {
    id: 'nb-006',
    mode: 'visual',
    prompt: 'In the simulator, set 2 classes, correlation ρ = 0.9 and move the query point to (2.4, 2). Compare the naive decision boundary (Decision Regions panel) with the blue true-generative boundary (scatter) and the data. Which statement matches what you see?',
    options: [
      'The two boundaries coincide exactly — the independence assumption makes no difference at any correlation',
      'The naive boundary is roughly a vertical line at x₁ ≈ 1.5 and barely moves with ρ, while the true generative boundary tilts with ρ — the naive model cannot see the within-class correlation',
      'The true boundary is vertical and the naive boundary tilts — correlation makes the naive model MORE accurate',
      'Both boundaries are circles because Gaussian densities are symmetric',
    ],
    answer: 'B',
    explanation: 'Naive Bayes fits per-feature variances only (a diagonal covariance), so it treats x₂ as independent of x₁: its boundary stays near the perpendicular bisector x₁ ≈ 1.5 no matter what ρ is. The true generative model uses the full covariance, so its boundary rotates as the class ellipses tilt — exactly the "boundary vs true generative" contrast.',
    trapExplanations: {
      A: 'The naive-vs-true divergence grows with ρ — that is the point of the correlation slider.',
      C: 'The opposite: correlation makes the naive boundary WRONG (it cannot use the correlated structure).',
      D: 'Gaussian densities give elliptical contours, and the boundaries here are curves/lines, not circles.',
    },
    concepts: ['naive-bayes', 'independence', 'decision-boundary', 'visualization'],
    difficulty: 3,
    tags: ['visual'],
  },
  {
    // +1: log-space underflow with concrete numbers.
    id: 'nb-007',
    mode: 'conceptual-mcq',
    prompt: 'A naive Bayes model has d = 400 binary features, and for some class every feature likelihood is 0.01. Raw P(x|C) = 0.01⁴⁰⁰. Why is it essential to work in log space?',
    options: [
      'log space makes the computation 400 times faster',
      '0.01⁴⁰⁰ = 10⁻⁸⁰⁰, far below the double-precision minimum ≈ 10⁻³⁰⁸, so the raw product underflows to 0 and every class ties at zero — but Σ log P(x_j|C) = 400·log(0.01) = −1842 is perfectly finite and keeps classes distinguishable',
      'The raw product overflows to infinity, which breaks normalization',
      'Bayes theorem is only valid in log space',
    ],
    answer: 'B',
    explanation: 'Multiplication of many small probabilities underflows: 0.01⁴⁰⁰ = 10⁻⁸⁰⁰ < 10⁻³⁰⁸. The log identity converts the product into the finite sum Σ log P(x_j|C) = −1842, and since log is monotone the argmax — and hence the prediction — is unchanged. Normalization uses the logsumexp trick.',
    trapExplanations: {
      A: 'Speed is not the reason; representability (avoiding underflow) is.',
      C: 'Products of probabilities underflow (→0), they do not overflow (→∞).',
      D: 'Bayes theorem is exact in any space; log space is a numerical convenience.',
    },
    concepts: ['naive-bayes', 'log-space', 'underflow', 'numerical-stability'],
    difficulty: 3,
    tags: ['conceptual', 'numerical', 'trap'],
  },
];
