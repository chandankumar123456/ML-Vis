// src/topics/logistic-regression/comparisons.ts
import type { Comparison } from '../../engine/types';

export const logisticComparisons: Comparison[] = [
  {
    id: 'lr-vs-perceptron',
    title: 'Logistic Regression vs Perceptron (smooth probabilistic vs hard threshold)',
    topics: ['logistic-regression', 'perceptron'],
    axes: [
      {
        axis: 'Output',
        entries: [
          { topic: 'logistic-regression', value: 'Probability p ∈ (0,1) — calibrated confidence, thresholded only at decision time' },
          { topic: 'perceptron', value: 'Hard class ±1 — no confidence, no probability' },
        ],
      },
      {
        axis: 'Loss',
        entries: [
          { topic: 'logistic-regression', value: 'Cross-entropy (convex, smooth, differentiable everywhere) — MLE' },
          { topic: 'perceptron', value: 'Perceptron loss max(0, −y·(w·x+b)) — piecewise linear, not differentiable at the margin' },
        ],
      },
      {
        axis: 'Convergence',
        entries: [
          { topic: 'logistic-regression', value: 'GD converges to the unique optimum (CE convex); no separability needed' },
          { topic: 'perceptron', value: 'Perceptron converges ONLY on linearly separable data (and to SOME separator, data-order dependent)' },
        ],
      },
      {
        axis: 'Boundary',
        entries: [
          { topic: 'logistic-regression', value: 'Linear p = 0.5 level set, positioned by the data distribution (not just by the points near it)' },
          { topic: 'perceptron', value: 'Any valid separating line — the algorithm stops at the first one it finds' },
        ],
      },
    ],
    notes: [
      'Logistic regression is the probabilistic, well-behaved upgrade of the perceptron: same linear score, but a smooth convex loss that yields probabilities.',
      'The perceptron’s inability to handle non-separable data is logistic regression’s main selling point — CE is defined for any data.',
    ],
  },
  {
    id: 'lr-vs-svm',
    title: 'Logistic Regression vs SVM (probability vs margin)',
    topics: ['logistic-regression', 'svm'],
    axes: [
      {
        axis: 'Objective',
        entries: [
          { topic: 'logistic-regression', value: 'min CE — penalizes ALL misclassified points, scaled by residual (ŷ−y); no hard margin concept' },
          { topic: 'svm', value: 'min ‖w‖² + C·Σξᵢ (hinge) — only points INSIDE/on the margin matter (support vectors)' },
        ],
      },
      {
        axis: 'Output',
        entries: [
          { topic: 'logistic-regression', value: 'Calibrated probability p — needs no extra step' },
          { topic: 'svm', value: 'Signed distance to the hyperplane — probabilities need Platt scaling' },
        ],
      },
      {
        axis: 'Solution',
        entries: [
          { topic: 'logistic-regression', value: 'Dense: every training point contributes to the gradient' },
          { topic: 'svm', value: 'Sparse: only support vectors determine the boundary' },
        ],
      },
      {
        axis: 'Boundary',
        entries: [
          { topic: 'logistic-regression', value: 'Linear (kernelized versions exist); positioned by the whole distribution' },
          { topic: 'svm', value: 'Maximum-margin hyperplane — the widest separator, most robust to small perturbations' },
        ],
      },
    ],
    notes: [
      'On separable data both find a linear boundary; SVM’s is the max-margin one, logistic’s is the MLE one — they differ and both are defensible.',
      'GATE contrast: hinge loss has no probabilistic interpretation; CE has no sparsity. Choose SVM for margin robustness, logistic for calibrated probabilities.',
    ],
  },
  {
    id: 'lr-vs-naive-bayes',
    title: 'Logistic Regression vs Naive Bayes (discriminative vs generative)',
    topics: ['logistic-regression', 'naive-bayes'],
    axes: [
      {
        axis: 'Modeling direction',
        entries: [
          { topic: 'logistic-regression', value: 'DISCRIMINATIVE: models P(y | x) directly — no assumption about how x is generated' },
          { topic: 'naive-bayes', value: 'GENERATIVE: models P(x | y) and P(y), then P(y|x) ∝ P(x|y)P(y) (Bayes rule)' },
        ],
      },
      {
        axis: 'Assumptions',
        entries: [
          { topic: 'logistic-regression', value: 'Linearity of the log-odds (the score is affine in x)' },
          { topic: 'naive-bayes', value: 'Conditional independence of features given the class (the "naive" assumption)' },
        ],
      },
      {
        axis: 'Data efficiency',
        entries: [
          { topic: 'logistic-regression', value: 'Needs more data to reach its (lower) asymptotic error' },
          { topic: 'naive-bayes', value: 'Converges faster with few samples (stronger inductive bias)' },
        ],
      },
      {
        axis: 'Boundary',
        entries: [
          { topic: 'logistic-regression', value: 'Linear (by construction)' },
          { topic: 'naive-bayes', value: 'Quadratic in general, but linear when the classes share a covariance (the classic equivalence)' },
        ],
      },
    ],
    notes: [
      'With Gaussian features sharing a covariance matrix, naive Bayes produces a LINEAR boundary — structurally the same family as logistic regression, but estimated differently.',
      'Asymptotically (n → ∞) the discriminative model wins or ties; with tiny samples the generative prior helps. This tradeoff is a favourite GATE discussion.',
    ],
  },
  {
    id: 'lr-vs-linear-regression',
    title: 'Logistic Regression vs Linear Regression (why not OLS on 0/1 labels)',
    topics: ['logistic-regression', 'multiple-linear-regression'],
    axes: [
      {
        axis: 'Target',
        entries: [
          { topic: 'logistic-regression', value: 'P(y = 1 | x) ∈ (0,1) — a probability, via the sigmoid link' },
          { topic: 'multiple-linear-regression', value: 'y ∈ R — an unbounded real value, predicted directly as w·x + b' },
        ],
      },
      {
        axis: 'Loss',
        entries: [
          { topic: 'logistic-regression', value: 'Cross-entropy (convex; the MLE for Bernoulli labels)' },
          { topic: 'multiple-linear-regression', value: 'MSE (the MLE for Gaussian noise)' },
        ],
      },
      {
        axis: 'Out-of-range predictions',
        entries: [
          { topic: 'logistic-regression', value: 'Impossible by construction — σ maps to (0,1)' },
          { topic: 'multiple-linear-regression', value: 'Easy: ŷ can be < 0 or > 1 for classification-style targets (a linear fit to 0/1 labels routinely over/undershoots)' },
        ],
      },
      {
        axis: 'Interpretation',
        entries: [
          { topic: 'logistic-regression', value: 'Weights act on the log-odds: e^wⱼ is the odds multiplier per unit of xⱼ' },
          { topic: 'multiple-linear-regression', value: 'Weights act additively on the raw target' },
        ],
      },
    ],
    notes: [
      'Fitting OLS to 0/1 labels is the classic misuse: the model produces "probabilities" outside [0,1], is dominated by confident outliers, and the squared-error geometry mismatches the Bernoulli likelihood.',
      'Linear regression is a regression tool; logistic regression is its classification counterpart — both linear in parameters, different links and losses.',
    ],
  },
];
