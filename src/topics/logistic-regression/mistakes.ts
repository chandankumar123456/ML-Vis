// src/topics/logistic-regression/mistakes.ts
import type { Mistake } from '../../engine/types';

export const logisticMistakes: Mistake[] = [
  {
    id: 'mse-for-classification',
    pattern: 'Using mean squared error as the loss for logistic regression (because it worked for linear regression)',
    example: 'L = \\frac{1}{n}\\sum_i (\\hat{y}_i - y_i)^2 \\quad\\text{instead of}\\quad L = -\\frac{1}{n}\\sum_i \\big[ y_i \\log \\hat{y}_i + (1-y_i) \\log(1-\\hat{y}_i) \\big]',
    whyWrong: 'Two independent strikes: (1) MSE + sigmoid is NON-CONVEX in w (the sigmoid’s curvature creates local optima), while CE is convex with the same model. (2) MSE’s gradient carries an extra σ′(z) = σ(1−σ) factor that → 0 at the tails — the model learns slowest exactly when it is confidently wrong. CE is the negative log-likelihood, so its minimizer is the MLE.',
    gateTrap: true,
    relatedConcept: 'cross-entropy',
  },
  {
    id: 'forgetting-residual-factor',
    pattern: 'Writing the logistic gradient without the (ŷ − y) residual factor',
    example: '\\frac{\\partial L}{\\partial w} = \\sum_i y_i x_i \\quad\\text{or}\\quad \\sum_i \\hat{y}_i x_i \\quad\\text{instead of}\\quad \\sum_i (\\hat{y}_i - y_i)\\, x_i',
    whyWrong: 'The chain rule produces exactly ∂L/∂w = Σ(ŷᵢ − yᵢ)xᵢ (the sigmoid’s own derivative cancels the 1/(ŷ(1−ŷ)) factors). Dropping the residual changes both the direction AND the magnitude of every update — the classic "missing (ŷ−y)" GATE trap.',
    gateTrap: true,
    relatedConcept: 'ce-gradient',
  },
  {
    id: 'threshold-05-only',
    pattern: 'Believing "predict class 1 iff p ≥ 0.5" is the only correct decision rule',
    example: '\\hat{y} = \\mathbb{1}[p \\ge 0.5] \\quad\\text{— one threshold among many}',
    whyWrong: 'The sigmoid’s output is a PROBABILITY; the threshold is a decision-rule choice, not part of the model. Raising it (say 0.7) trades false positives for false negatives and vice versa — useful under asymmetric costs. The p = 0.5 contour is just the most common default, and "p = 0.5 → class 1" is an arbitrary tie-break.',
    gateTrap: true,
    relatedConcept: 'decision-boundary',
  },
  {
    id: 'assuming-separability',
    pattern: 'Assuming a logistic-regression decision boundary always separates the training data perfectly',
    whyWrong: 'The boundary is LINEAR — it is the p = 0.5 level set of an affine score. On overlapping clusters, or non-linear structures (ring data), no straight line can split the classes: the model still fits (it places the best linear compromise) and misclassifies points near the boundary. "Accuracy 1" is a property of SEPARABLE data, not of logistic regression.',
    gateTrap: false,
    relatedConcept: 'decision-boundary',
  },
  {
    id: 'sign-of-gradient',
    pattern: 'Getting the sign of the gradient update backwards (adding instead of subtracting)',
    example: '\\theta \\leftarrow \\theta + \\eta \\sum_i (\\hat{y}_i - y_i) x_i \\quad\\text{instead of}\\quad \\theta \\leftarrow \\theta - \\eta \\sum_i (\\hat{y}_i - y_i) x_i',
    whyWrong: 'Gradient DESCENT moves opposite the gradient. With the wrong sign the loss increases every epoch — and on separable data it can look "productive" for a while because the boundary still wanders. The direction check: a point with y = 1 and ŷ < 1 has positive residual, so w should grow to raise z; only the minus-sign rule does that.',
    gateTrap: true,
    relatedConcept: 'ce-gradient',
  },
];
