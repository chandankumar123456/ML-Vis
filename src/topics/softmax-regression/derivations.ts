// src/topics/softmax-regression/derivations.ts
import type { Derivation } from '../../engine/types';

export const softmaxDerivations: Derivation[] = [
  {
    id: 'softmax-gradient',
    title: 'The Softmax Gradient: ∂L/∂w_k = Σ(ŷ_k − 1{y=k})x via the Chain Rule',
    steps: [
      {
        latex: 'L = -\\sum_{i} \\log \\hat{y}_{i, y_i} = -\\sum_i \\left( z_{i,y_i} - \\log \\sum_j e^{z_{ij}} \\right)',
        justification: 'Write the CE with the log-sum-exp trick: log ŷ_{i,y_i} = z_{i,y_i} − log Σ_j e^{z_{ij}}. This single form makes the chain rule clean (and it is the stable way to compute it).',
      },
      {
        latex: '\\frac{\\partial}{\\partial z_{ik}} \\log \\sum_j e^{z_{ij}} = \\frac{e^{z_{ik}}}{\\sum_j e^{z_{ij}}} = \\hat{y}_{ik}',
        justification: 'Derivative of log-sum-exp: the derivative of log f is f′/f, and ∂/∂z_{ik} of Σ_j e^{z_{ij}} is e^{z_{ik}}. The ratio is exactly the softmax of the k-th logit.',
      },
      {
        latex: '\\frac{\\partial L}{\\partial z_{ik}} = -1\\{y_i = k\\} + \\hat{y}_{ik}',
        justification: 'Differentiate the LSE form: ∂/∂z_{ik} of z_{i,y_i} is 1 iff y_i = k (indicator), and ∂/∂z_{ik} of the log-denominator is ŷ_{ik} by the previous step. The minus sign gives ŷ_{ik} − 1{y_i = k}.',
      },
      {
        latex: '\\frac{\\partial L}{\\partial w_k} = \\sum_{i} \\frac{\\partial L}{\\partial z_{ik}} \\cdot \\frac{\\partial z_{ik}}{\\partial w_k} = \\sum_i \\left( \\hat{y}_{ik} - 1\\{y_i = k\\} \\right) x_i',
        justification: 'Chain rule along z_{ik} = w_k·x_i + b_k, whose derivative w.r.t. w_k is x_i. Hand-check one entry: the residual ŷ_{ik} − 1{y_i=k} is positive when the model under-predicts class k, so the step moves w_k toward owning that point.',
      },
      {
        latex: '\\frac{\\partial L}{\\partial b_k} = \\sum_i \\left( \\hat{y}_{ik} - 1\\{y_i = k\\} \\right)',
        justification: 'Same chain rule with ∂z_{ik}/∂b_k = 1. Each class k has its OWN bias gradient — biases are not shared (per-class b_k).',
      },
      {
        latex: 'W \\leftarrow W - \\eta \\cdot \\frac{1}{n}\\left( \\hat{Y} - Y_{\\text{one-hot}} \\right)^T X',
        justification: 'The matrix form of the update: (Ŷ − Y_onehot)ᵀX is (K × d), the sum of (ŷ_i − onehot_i)·x_i over points. Each epoch subtracts η times this gradient — exactly what gdEpochStd computes on standardized features.',
      },
    ],
    derivedFrom: ['softmax-gradient'],
  },
  {
    id: 'log-sum-exp-stability',
    title: 'Log-Sum-Exp Stability: Why the Max-Shift Is Mandatory',
    steps: [
      {
        latex: '\\hat{y}_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}',
        justification: 'Textbook softmax. Mathematically perfect — numerically fragile when any logit is large (e.g. z = [1000, 1001, 1002]: e^{1002} ≈ 10⁴³⁵ overflows the double range 1.8×10³⁰⁸ → Infinity).',
      },
      {
        latex: '\\hat{y}_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}} = \\frac{e^{z_k - m}}{\\sum_j e^{z_j - m}}, \\qquad m = \\max_j z_j',
        justification: 'Multiply numerator and denominator by e^{−m}: the value is IDENTICAL for any m (shift invariance), but with m = max z_j every exponent is z_k − m ≤ 0, so exp(z_k − m) ∈ (0, 1] and the sum is in [1, K] — no overflow, no underflow to zero.',
      },
      {
        latex: '\\log \\sum_j e^{z_j} = m + \\log \\sum_j e^{z_j - m}',
        justification: 'The same shift stabilizes the log-denominator (log-sum-exp): with m = max, the inner sum is between 1 and K, so its log is O(1) and never −∞ or +∞. This is the form CE uses internally.',
      },
      {
        latex: '\\text{stable: } e^{z_k - m} \\in (0, 1], \\qquad \\text{naive: } e^{z_k} \\in [10^{-308}, \\infty)',
        justification: 'The contrast: naive exp spans 600+ orders of magnitude (over/underflow); the shifted version is bounded in (0, 1]. This is the canonical "softmax trap" — the topic simulates it in the failure demo "huge logits".',
      },
    ],
    derivedFrom: ['softmax-stable'],
  },
];
