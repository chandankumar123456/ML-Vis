// src/topics/logistic-regression/derivations.ts
import type { Derivation } from '../../engine/types';

export const logisticDerivations: Derivation[] = [
  {
    id: 'ce-from-mle',
    title: 'Cross-Entropy Loss from Maximum Likelihood',
    steps: [
      {
        latex: 'p(y_i = 1 \\mid x_i) = \\hat{y}_i = \\sigma(w \\cdot x_i + b)',
        justification: 'The model: each point is Bernoulli with success probability given by the sigmoid of its log-odds.',
      },
      {
        latex: 'P(y_i \\mid x_i) = \\hat{y}_i^{\\,y_i}\\, (1 - \\hat{y}_i)^{\\,1 - y_i}',
        justification: 'One compact factor covers both labels: yᵢ = 1 picks ŷᵢ, yᵢ = 0 picks (1−ŷᵢ).',
      },
      {
        latex: 'L(w, b) = \\prod_{i=1}^{n} \\hat{y}_i^{\\,y_i}\\, (1 - \\hat{y}_i)^{\\,1 - y_i}',
        justification: 'Independence (i.i.d. draws) → the likelihood is the product of the per-point probabilities.',
      },
      {
        latex: '\\ell(w, b) = \\ln L = \\sum_{i=1}^{n} \\Big[ y_i \\ln \\hat{y}_i + (1 - y_i) \\ln(1 - \\hat{y}_i) \\Big]',
        justification: 'The log converts the product to a sum (strictly monotone, so the maximizer is unchanged). Logs of probabilities are also numerically friendlier than the product.',
      },
      {
        latex: '\\arg\\max \\ell \\;\\Longleftrightarrow\\; \\arg\\min -\\ell = \\arg\\min \\frac{1}{n}\\sum_{i=1}^{n} \\Big[ -y_i \\ln \\hat{y}_i - (1 - y_i) \\ln(1 - \\hat{y}_i) \\Big]',
        justification: 'Maximizing likelihood ≡ minimizing the negative log-likelihood ≡ minimizing CROSS-ENTROPY (the 1/n is a harmless rescale). This is the MLE connection: logistic regression’s loss is not arbitrary — it IS maximum likelihood.',
      },
      {
        latex: '\\text{CE}(p, q) = H(p) + \\mathrm{KL}(p \\| q)',
        justification: 'Information-theory view: minimizing CE is equivalent to minimizing the KL divergence between the empirical label distribution and the model — the loss measures how many extra bits the model needs on average.',
      },
    ],
    derivedFrom: ['ce-from-mle'],
  },
  {
    id: 'ce-gradient-chain-rule',
    title: 'The (ŷ − y)x Gradient via the Chain Rule',
    steps: [
      {
        latex: 'z_i = w \\cdot x_i + b, \\qquad \\hat{y}_i = \\sigma(z_i), \\qquad L = \\frac{1}{n}\\sum_i \\Big[ -y_i \\ln \\hat{y}_i - (1 - y_i) \\ln(1 - \\hat{y}_i) \\Big]',
        justification: 'Set up the three layers of the computation graph: score → sigmoid → loss.',
      },
      {
        latex: '\\sigma\'(z) = \\frac{e^{-z}}{(1 + e^{-z})^2} = \\sigma(z)\\big(1 - \\sigma(z)\\big)',
        justification: 'Differentiate the sigmoid: d/dz of 1/(1+e^−z). The result is the famous "sigmoid squeeze" — its maximum is 1/4 at z = 0.',
      },
      {
        latex: '\\frac{\\partial L_i}{\\partial z_i} = -y_i \\frac{\\sigma\'(z_i)}{\\sigma(z_i)} + (1 - y_i)\\frac{\\sigma\'(z_i)}{1 - \\sigma(z_i)}',
        justification: 'Chain rule through the loss: d(−ln ŷ)/dz = −σ′/σ and d(−ln(1−ŷ))/dz = σ′/(1−σ) (the inner minus signs cancel).',
      },
      {
        latex: '\\frac{\\partial L_i}{\\partial z_i} = \\sigma\'(z_i)\\left( \\frac{\\hat{y}_i - y_i}{\\hat{y}_i(1 - \\hat{y}_i)} \\right)',
        justification: 'Combine over a common denominator ŷ(1−ŷ): −y/ŷ + (1−y)/(1−ŷ) = (ŷ−y)/(ŷ(1−ŷ)).',
      },
      {
        latex: '\\frac{\\partial L_i}{\\partial z_i} = \\sigma(z_i)(1 - \\sigma(z_i)) \\cdot \\frac{\\hat{y}_i - y_i}{\\hat{y}_i(1 - \\hat{y}_i)} = \\hat{y}_i - y_i',
        justification: 'The KEY cancellation: σ′ = ŷ(1−ŷ) cancels the denominator exactly, leaving the residual ŷᵢ − yᵢ. This cancellation is WHY cross-entropy is the natural partner of the sigmoid — squared error would leave the σ′ factor behind.',
      },
      {
        latex: '\\frac{\\partial L}{\\partial w_j} = \\sum_i \\frac{\\partial L_i}{\\partial z_i} \\frac{\\partial z_i}{\\partial w_j} = \\frac{1}{n}\\sum_i (\\hat{y}_i - y_i)\\, x_{ij}, \\qquad \\frac{\\partial L}{\\partial b} = \\frac{1}{n}\\sum_i (\\hat{y}_i - y_i)',
        justification: 'Chain rule through zᵢ = w·xᵢ + b: ∂zᵢ/∂wⱼ = xᵢⱼ and ∂zᵢ/∂b = 1. Vectorized: ∂L/∂w = (1/n)Xᵀ(ŷ−y) — the famous (ŷ−y)x result, one residual-weighted feature sum per component.',
      },
    ],
    derivedFrom: ['ce-gradient-chain-rule'],
  },
  {
    id: 'decision-boundary-level-set',
    title: 'The Boundary Is the p = 0.5 Level Set (a Line)',
    steps: [
      {
        latex: 'p = \\sigma(z) = 0.5',
        justification: 'The default decision rule: class 1 whenever the model assigns probability above 0.5.',
      },
      {
        latex: '\\sigma(z) = 0.5 \\;\\Longleftrightarrow\\; \\frac{1}{1 + e^{-z}} = \\frac{1}{2} \\;\\Longleftrightarrow\\; e^{-z} = 1 \\;\\Longleftrightarrow\\; z = 0',
        justification: 'σ is strictly monotone, so the 0.5-contour is exactly the z = 0 set — no calculus needed, just invert the sigmoid.',
      },
      {
        latex: 'z = 0 \\;\\Longleftrightarrow\\; w \\cdot x + b = 0 \\;\\Longleftrightarrow\\; x_2 = -\\frac{w_1}{w_2} x_1 - \\frac{b}{w_2}',
        justification: 'Since z is affine in x, its zero set is a straight line (hyperplane in d-D). The line’s slope is −w₁/w₂ and its intercept −b/w₂ — the weights literally ARE the geometry.',
      },
      {
        latex: 'p > 0.5 \\;\\Longleftrightarrow\\; z > 0 \\;\\Longleftrightarrow\\; w \\cdot x + b > 0',
        justification: 'Everything on one side of the line is class 1, everything on the other side class 0 — the probability heat (blue→red) is the smooth shading of this hard partition.',
      },
    ],
    derivedFrom: ['decision-boundary'],
  },
];
