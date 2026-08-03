// src/topics/cross-entropy-loss/comparisons.ts
import type { Comparison } from '../../engine/types';

export const ceComparisons: Comparison[] = [
  {
    id: 'ce-vs-mse',
    title: 'Cross-Entropy vs MSE for Classification',
    topics: ['cross-entropy-loss', 'multiple-linear-regression'],
    axes: [
      {
        axis: 'Loss definition',
        entries: [
          { topic: 'cross-entropy-loss', value: 'CE(p,q) = −Σ pᵢ log qᵢ = −[y log ŷ + (1−y) log(1−ŷ)] (binary)' },
          { topic: 'multiple-linear-regression', value: 'MSE = (1/n)Σ(y − ŷ)² — squared residual' },
        ],
      },
      {
        axis: 'Statistical origin',
        entries: [
          { topic: 'cross-entropy-loss', value: 'Negative log-likelihood (Bernoulli/Categorical) — maximum likelihood estimation' },
          { topic: 'multiple-linear-regression', value: 'Minimizing squared error = Gaussian MLE — least squares' },
        ],
      },
      {
        axis: 'With sigmoid output',
        entries: [
          { topic: 'cross-entropy-loss', value: 'Convex in the logits; gradient (ŷ − y)x never vanishes from saturation' },
          { topic: 'multiple-linear-regression', value: 'MSE+sigmoid is NON-convex; σ′(z) → 0 in saturated regions → vanishing gradients, slow learning' },
        ],
      },
      {
        axis: 'Penalty for confident-wrong',
        entries: [
          { topic: 'cross-entropy-loss', value: '−log q → ∞ as the wrong class probability → 0 — unbounded, strongly penalizes overconfidence' },
          { topic: 'multiple-linear-regression', value: 'Quadratic — bounded penalty for wrong-but-confident predictions in (0,1)' },
        ],
      },
    ],
    notes: [
      'The classic GATE contrast: both are convex in their natural parameterization (linear predictor), but CE is the NLL for classification and gives calibrated probabilities; MSE with sigmoid is non-convex.',
      'For regression, MSE is the right NLL (Gaussian); for classification, CE is the right NLL (Bernoulli/Categorical). Using MSE "because regression uses it" ignores the likelihood model.',
    ],
  },
  {
    id: 'ce-vs-hinge',
    title: 'Cross-Entropy vs Hinge Loss (SVM)',
    topics: ['cross-entropy-loss', 'logistic-regression'],
    axes: [
      {
        axis: 'Definition',
        entries: [
          { topic: 'cross-entropy-loss', value: 'CE = −[y log σ(z) + (1−y) log(1−σ(z))], σ(z) = sigmoid — smooth everywhere' },
          { topic: 'logistic-regression', value: 'Hinge = max(0, 1 − y·z) — piecewise linear, zero gradient past the margin' },
        ],
      },
      {
        axis: 'Output semantics',
        entries: [
          { topic: 'cross-entropy-loss', value: 'Produces calibrated probabilities (proper scoring rule) — σ(z) ∈ (0,1)' },
          { topic: 'logistic-regression', value: 'Uncalibrated margin score z; only the sign matters (a "distance", not a probability)' },
        ],
      },
      {
        axis: 'Gradient behavior',
        entries: [
          { topic: 'cross-entropy-loss', value: 'Nonzero gradient everywhere — every sample pushes the boundary' },
          { topic: 'logistic-regression', value: 'Sparse: samples beyond the margin contribute exactly 0 — the SVM support-vector property' },
        ],
      },
      {
        axis: 'Robustness',
        entries: [
          { topic: 'cross-entropy-loss', value: 'Sensitive to mislabeled outliers near the boundary (log grows unbounded)' },
          { topic: 'logistic-regression', value: 'Hinge is bounded by margin violation — more robust to outliers close to the boundary' },
        ],
      },
    ],
    notes: [
      'Both are convex surrogates of 0-1 loss with an interpretable boundary; CE is smooth and probabilistic, hinge is piecewise-linear and sparse-gradient.',
      'GATE: SVM hinge is a margin-based loss that ignores already-correct samples; logistic CE weights all samples — the two-views-of-the-boundary contrast.',
    ],
  },
  {
    id: 'ce-vs-01',
    title: 'Cross-Entropy vs 0-1 Loss',
    topics: ['cross-entropy-loss', 'logistic-regression'],
    axes: [
      {
        axis: 'Definition',
        entries: [
          { topic: 'cross-entropy-loss', value: 'CE = −Σ pᵢ log qᵢ — smooth, differentiable, convex in the logits' },
          { topic: 'logistic-regression', value: '0-1 loss = 1[ŷ ≠ y] — counts errors, flat everywhere, discontinuous at the boundary' },
        ],
      },
      {
        axis: 'Optimization',
        entries: [
          { topic: 'cross-entropy-loss', value: 'Gradient descent works directly (gradient (ŷ − y)x)' },
          { topic: 'logistic-regression', value: 'Non-differentiable, gradient is 0 almost everywhere — NP-hard to optimize directly; needs a convex surrogate' },
        ],
      },
      {
        axis: 'Confidence information',
        entries: [
          { topic: 'cross-entropy-loss', value: 'Penalizes overconfident mistakes (−log q → ∞) — rewards well-calibrated probabilities' },
          { topic: 'logistic-regression', value: 'All errors cost the same 1; no notion of confidence or margin' },
        ],
      },
      {
        axis: 'Consistency',
        entries: [
          { topic: 'cross-entropy-loss', value: 'A strictly proper scoring rule — minimizing CE recovers the true conditional probabilities' },
          { topic: 'logistic-regression', value: 'Only the error rate is minimized — any decision rule achieving the Bayes error is optimal' },
        ],
      },
    ],
    notes: [
      'CE (and hinge) are convex SURROGATES of the 0-1 loss: they are upper bounds that are easy to optimize, and minimizing them drives 0-1 loss down.',
      'GATE: "why not just minimize the misclassification count?" — 0-1 is discontinuous with zero gradient almost everywhere; classification losses in practice are smoothed surrogates.',
    ],
  },
];
