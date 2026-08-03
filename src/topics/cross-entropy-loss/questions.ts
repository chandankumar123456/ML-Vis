// src/topics/cross-entropy-loss/questions.ts
import type { Question } from '../../engine/types';

export const ceQuestions: Question[] = [
  {
    // Hand-verified: CE = −(0.7·ln 0.2 + 0.3·ln 0.8). ln 0.2 = −1.60944, ×0.7 = −1.12661;
    // ln 0.8 = −0.22314, ×0.3 = −0.06694. CE = 1.12661 + 0.06694 = 1.19355 ≈ 1.194.
    id: 'ce-nat-001',
    mode: 'nat',
    prompt: 'For p = [0.7, 0.3] and q = [0.2, 0.8], compute the cross-entropy CE(p, q) = −Σ pᵢ·log qᵢ using NATURAL log. Enter your answer to 3 decimals.',
    answer: 1.194,
    tolerance: 0.005,
    explanation: 'CE(p,q) = −(0.7·ln 0.2 + 0.3·ln 0.8) = −(0.7·(−1.6094) + 0.3·(−0.2231)) = −(−1.1266 − 0.0669) = 1.1936. Note this is NOT CE(q,p): CE(q,p) = −(0.3·ln 0.7 + 0.7·ln 0.2) = 0.1070 + 1.1266 = 1.2336 — cross-entropy is asymmetric. All logs natural (nats).',
    concepts: ['cross-entropy', 'information-theory'],
    difficulty: 3,
    tags: ['numerical', 'formula'],
  },
  {
    // Hand-verified: KL = 0.5·ln(0.5/0.8) + 0.5·ln(0.5/0.2) = 0.5·ln 0.625 + 0.5·ln 2.5
    // = 0.5·(−0.4700) + 0.5·(0.9163) = −0.2350 + 0.4581 = 0.2231 ≈ 0.223.
    id: 'ce-nat-002',
    mode: 'nat',
    prompt: 'For p = [0.5, 0.5] and q = [0.8, 0.2], compute the KL divergence KL(p‖q) = Σ pᵢ·log(pᵢ/qᵢ) using NATURAL log. Enter your answer to 3 decimals.',
    answer: 0.223,
    tolerance: 0.005,
    explanation: 'KL = 0.5·ln(0.5/0.8) + 0.5·ln(0.5/0.2) = 0.5·ln(0.625) + 0.5·ln(2.5) = 0.5·(−0.4700) + 0.5·(0.9163) = 0.2231. The identity CE = H + KL checks: H(p) = ln 2 = 0.6931 and CE(p,q) = −(0.5·ln 0.8 + 0.5·ln 0.2) = 0.9163, so CE − H = 0.9163 − 0.6931 = 0.2232 ✓.',
    concepts: ['kl-divergence', 'information-theory'],
    difficulty: 3,
    tags: ['numerical', 'formula'],
  },
  {
    // Hand-verified trivially: θ̂ = h/n = 7/10 = 0.7 (MLE of a Bernoulli bias).
    id: 'ce-nat-003',
    mode: 'nat',
    prompt: 'A coin is flipped 10 times and lands heads 7 times. Under the Bernoulli model, what is the maximum-likelihood estimate θ̂ of the coin\'s heads probability? Enter to 2 decimals.',
    answer: 0.7,
    tolerance: 0.005,
    explanation: 'The log-likelihood is log L(θ) = 7·log θ + 3·log(1−θ). Setting d/dθ = 7/θ − 3/(1−θ) = 0 gives 7(1−θ) = 3θ ⇒ θ = 7/10 = 0.7 — the empirical frequency. The same θ minimizes the per-sample NLL = CE(empirical, model).',
    concepts: ['mle', 'bernoulli-likelihood'],
    difficulty: 2,
    tags: ['numerical', 'formula'],
  },
  {
    // Hand-verified: KL(p‖q) = 0.8·ln(0.8/0.6) + 0.2·ln(0.2/0.4) = 0.8·ln(4/3) + 0.2·ln 0.5
    // = 0.8·(0.2877) + 0.2·(−0.6931) = 0.2301 − 0.1386 = 0.0915 ≈ 0.092.
    // Contrast KL(q‖p) = 0.6·ln(0.6/0.8) + 0.4·ln(0.4/0.2) = 0.6·(−0.2877) + 0.4·(0.6931)
    // = −0.1726 + 0.2773 = 0.1046 ≈ 0.105 — asymmetric.
    id: 'ce-nat-004',
    mode: 'nat',
    prompt: 'For p = [0.8, 0.2] and q = [0.6, 0.4], compute KL(p‖q) = Σ pᵢ·log(pᵢ/qᵢ) using NATURAL log. Enter your answer to 3 decimals. (The swapped KL(q‖p) is different — asymmetry.)',
    answer: 0.092,
    tolerance: 0.005,
    explanation: 'KL(p‖q) = 0.8·ln(0.8/0.6) + 0.2·ln(0.2/0.4) = 0.8·ln(4/3) + 0.2·ln(0.5) = 0.8·(0.2877) + 0.2·(−0.6931) = 0.0915. The swapped direction gives KL(q‖p) = 0.6·ln(0.6/0.8) + 0.4·ln(0.4/0.2) = 0.1046 — NOT equal. KL is not a metric (no symmetry, no triangle inequality); only KL ≥ 0 with equality iff p = q is guaranteed.',
    concepts: ['kl-divergence', 'information-theory'],
    difficulty: 3,
    tags: ['numerical', 'trap'],
  },
  {
    id: 'ce-con-001',
    mode: 'conceptual-mcq',
    prompt: 'Why do we maximize the LOG-likelihood instead of the raw likelihood?',
    options: [
      'The log turns the product of per-sample probabilities into a sum: numerically stable (products underflow), and the sum is concave/differentiable — same argmax',
      'Because log-likelihood is the only way to get a point estimate',
      'Because likelihood is undefined for discrete data',
      'Because maximizing log-likelihood is easier to compute by hand for every model',
    ],
    answer: 'A',
    explanation: 'Likelihoods are products of n probabilities — for even n = 100 they underflow to 0 in floating point. The log is monotone (same argmax) and maps products to sums of per-sample terms; the negative log-likelihood is exactly the cross-entropy loss we minimize in classification.',
    trapExplanations: {
      B: 'Likelihood itself can be maximized in principle; the log is a numerical and analytical convenience, not a necessity.',
      C: 'Likelihood is a probability of the data — defined for discrete outcomes too.',
      D: 'The benefit is principled (concavity, stability, CE connection), not convenience.',
    },
    concepts: ['mle', 'log-likelihood', 'cross-entropy'],
    difficulty: 2,
    tags: ['conceptual'],
  },
  {
    id: 'ce-matrix-001',
    mode: 'matrix',
    prompt: 'Softmax + CE: for a 3-class sample with one-hot label y = [0, 1, 0] and softmax output s = [0.2, 0.6, 0.2], the gradient ∂CE/∂z = s − y. Which vector is the gradient?',
    options: [
      '[0.2, −0.4, 0.2]',
      '[−0.8, 0.4, −0.8]',
      '[0.8, −0.4, 0.8]',
      '[0.2, 0.6, 0.2]',
    ],
    answer: 'A',
    explanation: '∂CE/∂z_j = s_j − y_j componentwise: [0.2−0, 0.6−1, 0.2−0] = [0.2, −0.4, 0.2]. The true class (j = 2) gets a negative gradient (probability should rise), the other classes positive (probabilities should fall), and Σ(s_j − y_j) = 1 − 1 = 0 — gradient components sum to zero, consistent with the softmax\'s Σs = 1 constraint.',
    trapExplanations: {
      B: 'That is −(s − y)·something wrong: it misplaces the −1 onto class 1 — the one-hot subtraction must be componentwise (s_j − y_j), not s − y transposed.',
      C: 'That vector has Σ = 1.2, but gradients from a normalized output must sum to 0.',
      D: 'That is just the softmax output s itself — the gradient is s MINUS y, not s.',
    },
    concepts: ['softmax-regression', 'cross-entropy', 'matrix-operations'],
    difficulty: 4,
    tags: ['matrix', 'numerical'],
  },
  {
    id: 'ce-visual-001',
    mode: 'visual',
    prompt: 'In the simulation, set Facet = Cross-Entropy, p₀ = 0.7, and scrub q₀ from 0.05 to 0.95. Watch the CE loss surface (blue), the entropy floor H(p) (green) and the KL gap. Which statement best describes what you observe?',
    options: [
      'CE(p,q) always sits at or above the green floor H(p); the gap is KL(p‖q), and it shrinks to zero exactly when q₀ reaches p₀ = 0.7',
      'CE(p,q) goes below H(p) for small q₀, so entropy is an upper bound, not a floor',
      'The red −log q₀ penalty curve is flat because the penalty only depends on p, not on q',
      'CE(p,q) is symmetric around q₀ = 0.5, so the minimum is always at the balanced distribution',
    ],
    answer: 'A',
    explanation: 'CE = H + KL with KL ≥ 0: the blue curve never dips below the green floor, and they touch exactly at q₀ = p₀ = 0.7 (the minimum of the loss surface, by strict convexity in q₀). The vertical gap between the amber marker and the green line IS the KL divergence.',
    trapExplanations: {
      B: 'KL ≥ 0 (Gibbs inequality) forces CE ≥ H always — the floor is never crossed.',
      C: 'The red curve −log q₀ depends only on q₀ and spikes to ∞ as q₀ → 0: the penalty for a confident-wrong prediction.',
      D: 'The minimum is at q = p (0.7), not at the balanced 0.5 — asymmetry of the loss surface.',
    },
    concepts: ['cross-entropy', 'kl-divergence', 'loss-surfaces'],
    difficulty: 2,
    tags: ['visual'],
  },
  {
    id: 'ce-trap-001',
    mode: 'gate-mcq',
    prompt: 'GATE-style statement: "For binary classification with a sigmoid output, using squared error (MSE) is just as good as cross-entropy loss." Which is the correct assessment?',
    options: [
      'False — with sigmoid outputs, MSE is non-convex in the weights (saturated regions give tiny gradients) and lacks the probabilistic NLL interpretation; CE is convex in the logits and is the NLL',
      'True — any loss that is minimized at perfect prediction is equivalent for training',
      'True, but only because MSE is bounded between 0 and 1',
      'False — MSE is convex in the weights while CE is not, so MSE is strictly better',
    ],
    answer: 'A',
    explanation: 'With σ(z) saturating, the MSE gradient vanishes in the saturated region (σ′ → 0) — the classic slow-learning / non-convex issue. CE is the negative log-likelihood: convex in the logits, gradient (ŷ − y)x never vanishes from saturation, and it gives calibrated probabilities. This is why CE is the default for classification.',
    trapExplanations: {
      B: 'Both are minimized at perfect prediction, but the optimization landscape and statistical meaning differ profoundly.',
      C: 'Boundedness is irrelevant — a bounded loss can still be non-convex (it is).',
      D: 'The direction is wrong: CE is the convex NLL; MSE with sigmoid is the non-convex one.',
    },
    concepts: ['cross-entropy', 'mse', 'logistic-regression'],
    difficulty: 3,
    tags: ['trap', 'conceptual'],
  },
  {
    id: 'ce-gate-001',
    mode: 'gate-mcq',
    prompt: 'Which statement about cross-entropy is TRUE?',
    options: [
      'CE(p,q) = H(p) + KL(p‖q) ≥ H(p), with equality exactly when q = p',
      'CE(p,q) = H(q) always, because entropy is symmetric',
      'CE(p,q) is symmetric: CE(p,q) = CE(q,p) for all p, q',
      'CE(p,q) = 0 whenever p and q have the same support',
    ],
    answer: 'A',
    explanation: 'CE = H(p) + KL(p‖q) is the defining decomposition; KL ≥ 0 (Gibbs) makes CE ≥ H(p) with equality iff q = p — the basis of the MLE ⟺ min-CE equivalence. CE is asymmetric, equals H(p) (not 0) at q = p, and depends on the probability masses, not just the support.',
    trapExplanations: {
      B: 'CE involves two distributions; H(q) is a property of q alone and has no such identity.',
      C: 'CE(p,q) ≠ CE(q,p) in general — asymmetry is a defining feature.',
      D: 'Equal support is not enough: CE(p,p) = H(p) > 0 for any non-degenerate p.',
    },
    concepts: ['cross-entropy', 'kl-divergence', 'entropy'],
    difficulty: 2,
    tags: ['trap', 'conceptual'],
  },
];
