// src/topics/cross-entropy-loss/formulas.ts
import type { Formula } from '../../engine/types';

// All logarithms are NATURAL (base e) — quantities are measured in NATS. The
// log-base mistake (bits vs nats) is covered in mistakes.ts; converting nats to
// bits divides by ln 2 ≈ 0.693 (1 nat = 1/ln 2 ≈ 1.44 bits).
export const ceFormulas: Formula[] = [
  {
    id: 'entropy',
    latex: 'H(p) = -\\sum_i p_i \\log p_i',
    symbols: [
      { symbol: 'p_i', meaning: 'probability of outcome i (Σ p_i = 1)', dimensions: 'probability, p_i ∈ (0,1]' },
      { symbol: '\\log', meaning: 'natural log — entropy is measured in NATS (bits = nats/ln 2)', dimensions: 'nats' },
      { symbol: 'H(p)', meaning: 'entropy of p — the expected surprise E_p[−log p] of sampling from p', dimensions: 'nats' },
    ],
    assumptions: ['p is a probability distribution (non-negative, sums to 1)', '0·log 0 = 0 by convention (limit x·ln x as x → 0⁺)'],
    failureCases: ['p_i = 0 contributes nothing (0·log 0 → 0)', 'confusing log base 2 (bits) with base e (nats) — a 1/ln 2 ≈ 1.44× factor'],
    derivesFrom: [],
    connections: ['Cross-entropy', 'KL divergence', 'Information theory'],
    whyWorks: 'H(p) = E_p[−log p] averages the information −log p over the distribution itself. For a binary p = [p, 1−p], H = −p log p − (1−p) log(1−p) peaks at p = 1/2 (maximum uncertainty) and is 0 at p ∈ {0, 1} (no uncertainty). Entropy is a property of ONE distribution.',
  },
  {
    id: 'cross-entropy',
    latex: 'CE(p,q) = -\\sum_i p_i \\log q_i',
    symbols: [
      { symbol: 'p', meaning: 'true / observed distribution (one-hot y in classification)', dimensions: 'probability vector' },
      { symbol: 'q', meaning: 'predicted distribution (model output ŷ)', dimensions: 'probability vector' },
      { symbol: '\\log q_i', meaning: 'coding cost of outcome i under the predicted distribution — the penalty for a confident-wrong prediction', dimensions: 'nats' },
    ],
    assumptions: ['q must have full support over p for CE to be finite', 'p and q are distributions over the same outcomes'],
    failureCases: ['q_i = 0 with p_i > 0 → −p_i·log 0 = +∞ (the log(0) blow-up)', 'confident wrong prediction: q_true-class → 0 → penalty −log q_true → ∞'],
    derivesFrom: ['entropy'],
    connections: ['KL divergence', 'Logistic regression loss', 'Softmax regression loss', 'Maximum likelihood'],
    whyWorks: 'CE is the expected coding cost of encoding samples from p using q: E_p[−log q]. In binary classification it is the per-sample NLL −[y log ŷ + (1−y) log(1−ŷ)]. CE = H(p) + KL(p‖q) ≥ H(p) — minimized exactly when q = p.',
  },
  {
    id: 'kl-divergence',
    latex: 'KL(p\\|q) = \\sum_i p_i \\log \\frac{p_i}{q_i}',
    symbols: [
      { symbol: 'p', meaning: 'reference distribution', dimensions: 'probability vector' },
      { symbol: 'q', meaning: 'approximating distribution', dimensions: 'probability vector' },
      { symbol: '\\log \\frac{p_i}{q_i}', meaning: 'per-outcome "surprise excess" of q relative to p', dimensions: 'nats' },
    ],
    assumptions: ['q_i = 0 with p_i > 0 is disallowed (KL = ∞ there — q must cover p)'],
    failureCases: ['KL(p‖q) = ∞ when q assigns zero probability where p has mass', 'KL is NOT symmetric: KL(p‖q) ≠ KL(q‖p) in general — not a metric (also no triangle inequality)'],
    derivesFrom: ['cross-entropy', 'entropy'],
    connections: ['Gibbs inequality', 'Cross-entropy', 'Maximum likelihood'],
    whyWorks: 'KL(p‖q) = CE(p,q) − H(p) is the extra expected coding cost of using q instead of the optimal p. By Gibbs inequality (Jensen on −log), KL ≥ 0 with equality iff p = q — the fundamental reason CE is minimized at q = p and MLE and min-CE coincide.',
  },
  {
    id: 'ce-h-kl',
    latex: 'CE(p,q) = H(p) + KL(p\\|q) \\;\\Rightarrow\\; CE(p,q) \\ge H(p)',
    symbols: [
      { symbol: 'H(p)', meaning: 'entropy floor — the best possible CE, achieved at q = p', dimensions: 'nats' },
      { symbol: 'KL(p\\|q)', meaning: 'the "distance" term — non-negative, 0 iff q = p', dimensions: 'nats' },
    ],
    assumptions: ['q covers p (else KL = ∞)'],
    failureCases: ['KL ≥ 0 fails only through numerical error when q ≈ p (subtraction of nearly equal floats)'],
    derivesFrom: ['cross-entropy', 'kl-divergence'],
    connections: ['Gibbs inequality', 'Bregman divergence'],
    whyWorks: 'Rewrite CE = −Σ p_i log q_i = −Σ p_i [log p_i + log(q_i/p_i)] = H(p) + Σ p_i log(p_i/q_i) = H(p) + KL(p‖q). Since KL ≥ 0, the CE loss curve always sits at or above the entropy floor, touching it exactly at q = p — the picture the loss-curve view draws.',
  },
  {
    id: 'bernoulli-likelihood',
    latex: 'L(\\theta) = \\prod_{i=1}^{n} \\theta^{y_i}(1-\\theta)^{1-y_i} = \\theta^{h}(1-\\theta)^{n-h}',
    symbols: [
      { symbol: 'y_i', meaning: 'flip i outcome: 1 (heads) or 0 (tails)', dimensions: 'Bernoulli {0,1}' },
      { symbol: '\\theta', meaning: 'coin bias — probability of heads', dimensions: 'θ ∈ (0,1)' },
      { symbol: 'h = \\sum_i y_i', meaning: 'observed heads count — a sufficient statistic', dimensions: 'count' },
      { symbol: 'L(\\theta)', meaning: 'likelihood of the observed SEQUENCE as a function of θ', dimensions: 'probability of the data' },
    ],
    assumptions: ['flips are i.i.d. Bernoulli(θ)', 'the binomial coefficient C(n,h) is omitted — it counts sequences, is θ-independent, and cancels in the argmax'],
    failureCases: ['θ ∈ {0,1} with h interior → 0^h or 0^(n−h) degeneracies (log 0 in log-form)'],
    derivesFrom: [],
    connections: ['Log-likelihood', 'Maximum likelihood', 'Binomial distribution'],
    whyWorks: 'Independence turns the joint probability of the flip sequence into a PRODUCT (the plan\'s "MLE derivation: product → log → sum"). Only the counts (h, n) survive — the empirical frequency is sufficient for the Bernoulli model.',
  },
  {
    id: 'log-likelihood',
    latex: '\\log L(\\theta) = h\\log\\theta + (n-h)\\log(1-\\theta), \\qquad \\hat\\theta_{MLE} = \\frac{h}{n}',
    symbols: [
      { symbol: '\\log L(\\theta)', meaning: 'log-likelihood — a sum of per-flip terms (product → log → sum)', dimensions: 'nats' },
      { symbol: '\\hat\\theta_{MLE}', meaning: 'the MLE of the bias — the empirical heads frequency', dimensions: 'probability' },
    ],
    assumptions: ['θ ∈ (0,1) so both log terms are defined'],
    failureCases: ['maximizing the raw product L directly underflows for large n — always work with log L'],
    derivesFrom: ['bernoulli-likelihood'],
    connections: ['Cross-entropy', 'Maximum likelihood', 'Newton-Raphson'],
    whyWorks: 'Setting d/dθ [h log θ + (n−h) log(1−θ)] = h/θ − (n−h)/(1−θ) = 0 gives h(1−θ) = (n−h)θ ⇒ θ̂ = h/n. The log turns the product into a concave SUM, so the stationary point is the unique global maximum.',
  },
  {
    id: 'nll-is-ce',
    latex: '-\\frac{1}{n}\\log L(\\theta) = CE\\big(\\hat p,\\ q_\\theta\\big) = -\\Big[\\hat p \\log\\theta + (1-\\hat p)\\log(1-\\theta)\\Big], \\quad \\hat p = \\frac{h}{n}',
    symbols: [
      { symbol: '\\hat p', meaning: 'empirical distribution [h/n, 1−h/n]', dimensions: 'probability vector' },
      { symbol: 'q_\\theta', meaning: 'model distribution [θ, 1−θ]', dimensions: 'probability vector' },
      { symbol: '-\\frac{1}{n}\\log L', meaning: 'per-sample negative log-likelihood — the loss being minimized', dimensions: 'nats/sample' },
    ],
    assumptions: ['Bernoulli likelihood without the θ-independent binomial coefficient'],
    failureCases: ['adding the binomial coefficient would offset CE by the θ-independent constant log C(n,h)/n, breaking the exact equality'],
    derivesFrom: ['log-likelihood', 'cross-entropy'],
    connections: ['Logistic regression', 'Softmax regression', 'MLE'],
    whyWorks: 'Expand: −(h/n) log θ − ((n−h)/n) log(1−θ) = −Σ p̂_i log q_i = CE(p̂, q_θ). Minimizing the negative log-likelihood is EXACTLY minimizing the cross-entropy between the empirical and the model distribution — the MLE ⟺ min-CE bridge, and the hidden gate link to logistic regression.',
  },
  {
    id: 'softmax-ce-gradient',
    latex: '\\frac{\\partial CE}{\\partial z_j} = s_j - y_j, \\qquad s = \\mathrm{softmax}(z), \\ y \\text{ one-hot}',
    symbols: [
      { symbol: 'z_j', meaning: 'logit for class j', dimensions: 'real' },
      { symbol: 's_j', meaning: 'softmax output — predicted probability of class j', dimensions: 'probability, Σ s = 1' },
      { symbol: 'y_j', meaning: 'one-hot true label (1 for the true class, 0 else)', dimensions: '{0,1}' },
    ],
    assumptions: ['CE loss with a softmax output layer and one-hot targets'],
    failureCases: ['applying the sigmoid-gradient formula (ŷ−y)x to a softmax layer — the softmax Jacobian s_k(δ_kj − s_j) must be used'],
    derivesFrom: ['cross-entropy'],
    connections: ['Softmax regression', 'Backpropagation', 'Logistic regression'],
    whyWorks: 'Chain rule: ∂CE/∂s_j = −y_j/s_j and ∂s_k/∂z_j = s_k(δ_kj − s_j). Summing −Σ_k (y_k/s_k)·s_k(δ_kj − s_j) = −y_j + s_j Σ_k y_k = s_j − y_j (using Σ y_k = 1). The logistic-regression gradient ∂L/∂w = Σ(ŷ − y)x is the two-class special case of this.',
  },
];
