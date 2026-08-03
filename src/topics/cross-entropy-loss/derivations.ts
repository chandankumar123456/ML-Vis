// src/topics/cross-entropy-loss/derivations.ts
import type { Derivation } from '../../engine/types';

export const ceDerivations: Derivation[] = [
  {
    id: 'mle-to-nll-to-ce',
    title: 'From MLE to Negative Log-Likelihood to Cross-Entropy (the plan\'s "product → log → sum")',
    steps: [
      {
        latex: 'L(\\theta) = \\prod_{i=1}^{n} \\theta^{y_i}(1-\\theta)^{1-y_i}',
        justification: 'The likelihood of the observed flip sequence under bias θ. Independence of flips makes the joint probability a PRODUCT — the "product" step. (The binomial coefficient C(n,h) is a θ-independent constant; it cancels in the argmax and is omitted.)',
      },
      {
        latex: '\\log L(\\theta) = \\sum_{i=1}^{n} \\Big[ y_i \\log\\theta + (1-y_i)\\log(1-\\theta) \\Big] = h\\log\\theta + (n-h)\\log(1-\\theta)',
        justification: 'The "log → sum" step: the product becomes a sum of per-sample terms. Each y_i ∈ {0,1} contributes one of the two log terms, so the sum collapses to counts: h heads and n−h tails.',
      },
      {
        latex: '\\text{maximize } L(\\theta) \\;\\Longleftrightarrow\\; \\text{maximize } \\log L(\\theta) \\;\\Longleftrightarrow\\; \\text{minimize } -\\log L(\\theta)',
        justification: 'log is strictly increasing and −log reverses the order, so the argmax is unchanged. Working with the negative log-likelihood (NLL) is numerically stable (no product underflow) and differentiable.',
      },
      {
        latex: '-\\frac{1}{n}\\log L(\\theta) = -\\Big[ \\frac{h}{n}\\log\\theta + \\frac{n-h}{n}\\log(1-\\theta)\\Big] = CE\\big(\\hat p,\\ q_\\theta\\big), \\quad \\hat p = \\Big[\\frac{h}{n},\\ \\frac{n-h}{n}\\Big]',
        justification: 'Expanding per sample: −(h/n)·log θ − ((n−h)/n)·log(1−θ) is exactly −Σ p̂ᵢ log qᵢ with p̂ the empirical distribution and q_θ = [θ, 1−θ]. Per-sample NLL IS the cross-entropy of empirical vs model.',
      },
      {
        latex: '\\frac{d}{d\\theta}\\log L = \\frac{h}{\\theta} - \\frac{n-h}{1-\\theta} = 0 \\;\\Rightarrow\\; h(1-\\theta) = (n-h)\\theta \\;\\Rightarrow\\; \\hat\\theta_{MLE} = \\frac{h}{n}',
        justification: 'Setting the derivative to zero. The log-likelihood is strictly concave in θ (d²/dθ² = −h/θ² − (n−h)/(1−θ)² < 0), so this stationary point is the unique global maximum — the MLE is the empirical frequency, and the same θ̂ minimizes the cross-entropy.',
      },
    ],
    derivedFrom: ['mle-to-nll-to-ce'],
  },
  {
    id: 'ce-h-kl',
    title: 'Decomposing Cross-Entropy: CE(p,q) = H(p) + KL(p‖q)',
    steps: [
      {
        latex: 'CE(p,q) = -\\sum_i p_i \\log q_i',
        justification: 'Definition of cross-entropy — the expected coding cost of using q to encode samples from p.',
      },
      {
        latex: 'CE(p,q) = -\\sum_i p_i \\Big[ \\log p_i + \\log\\frac{q_i}{p_i} \\Big]',
        justification: 'Insert the identity q_i = p_i·(q_i/p_i) inside the log — the log of a product splits into a sum.',
      },
      {
        latex: 'CE(p,q) = \\underbrace{-\\sum_i p_i \\log p_i}_{H(p)} + \\underbrace{\\sum_i p_i \\log\\frac{p_i}{q_i}}_{KL(p\\|q)}',
        justification: 'Distribute: the first term is exactly H(p); the second is exactly the KL divergence. Note the sign flip: −Σ p_i·log(q_i/p_i) = +Σ p_i·log(p_i/q_i).',
      },
      {
        latex: 'KL(p\\|q) = -\\sum_i p_i \\log\\frac{q_i}{p_i} = -E_p\\Big[\\log\\frac{q}{p}\\Big] \\ge -\\log E_p\\Big[\\frac{q}{p}\\Big] = -\\log\\sum_i q_i = 0',
        justification: 'Gibbs inequality via Jensen: −log is convex, so E[−log X] ≥ −log E[X]. The expectation of q/p under p is Σ q_i = 1, giving KL ≥ 0 with equality iff q = p (log is strictly concave).',
      },
      {
        latex: 'CE(p,q) = H(p) + KL(p\\|q) \\ge H(p), \\qquad CE(p,p) = H(p)',
        justification: 'Combining the decomposition with KL ≥ 0: the CE loss is always at least the entropy floor H(p), and it touches the floor exactly when the predicted distribution equals the true one. This is why minimizing CE pushes q → p — and why MLE and min-CE coincide.',
      },
    ],
    derivedFrom: ['ce-h-kl'],
  },
  {
    id: 'softmax-ce-gradient',
    title: 'The Softmax + CE Gradient: ∂CE/∂z = s − y (the famous result)',
    steps: [
      {
        latex: 'CE = -\\sum_k y_k \\log s_k, \\qquad s_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}',
        justification: 'CE loss with a softmax output layer and one-hot target y (Σ y = 1).',
      },
      {
        latex: '\\frac{\\partial CE}{\\partial s_k} = -\\frac{y_k}{s_k}',
        justification: 'Only the k = true-class term of the sum survives differentiation — no chain rule needed yet.',
      },
      {
        latex: '\\frac{\\partial s_k}{\\partial z_j} = s_k\\left(\\delta_{kj} - s_j\\right)',
        justification: 'The softmax Jacobian: ∂s_k/∂z_j = s_k(δ_kj − s_j). This is the multi-class replacement for the scalar sigmoid derivative σ′(z) = σ(z)(1−σ(z)).',
      },
      {
        latex: '\\frac{\\partial CE}{\\partial z_j} = \\sum_k \\frac{\\partial CE}{\\partial s_k}\\frac{\\partial s_k}{\\partial z_j} = \\sum_k \\left(-\\frac{y_k}{s_k}\\right) s_k \\left(\\delta_{kj} - s_j\\right)',
        justification: 'Chain rule. The s_k factors cancel, leaving −Σ_k y_k(δ_kj − s_j).',
      },
      {
        latex: '\\frac{\\partial CE}{\\partial z_j} = -y_j + s_j\\sum_k y_k = s_j - y_j, \\qquad \\nabla_z CE = s - y',
        justification: 'Σ_k y_k(δ_kj − s_j) = y_j − s_j·Σ_k y_k = y_j − s_j (one-hot sums to 1). The gradient is the softmax output minus the one-hot target — the logistic-regression gradient (ŷ − y)x is the two-class special case.',
      },
    ],
    derivedFrom: ['softmax-ce-gradient'],
  },
];
