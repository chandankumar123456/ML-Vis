// src/topics/svm-hard-margin/comparisons.ts
import type { Comparison } from '../../engine/types';

export const svmComparisons: Comparison[] = [
  {
    id: 'svm-vs-perceptron',
    title: 'SVM vs Perceptron (max-margin vs any-separator)',
    topics: ['svm-hard-margin', 'perceptron'],
    axes: [
      {
        axis: 'Objective',
        entries: [
          { topic: 'svm-hard-margin', value: 'Maximize the margin (minimize ½‖w‖²) — a unique optimum; 2 SVs decide it on the default seed' },
          { topic: 'perceptron', value: 'Any separating hyperplane — the answer depends on initialization and update order' },
        ],
      },
      {
        axis: 'Optimality',
        entries: [
          { topic: 'svm-hard-margin', value: 'Unique max-margin separator, provably most robust to perturbation' },
          { topic: 'perceptron', value: 'Converges (separable data) but to an arbitrary separator, possibly hugging a support vector' },
        ],
      },
      {
        axis: 'Representation',
        entries: [
          { topic: 'svm-hard-margin', value: 'w = Σαᵢyᵢxᵢ over SUPPORT VECTORS only (αᵢ = 0 elsewhere)' },
          { topic: 'perceptron', value: 'w accumulates every misclassified point during training' },
        ],
      },
      {
        axis: 'Convergence guarantee',
        entries: [
          { topic: 'svm-hard-margin', value: 'Exact in finite time (the 2D geometric solver enumerates candidate directions)' },
          { topic: 'perceptron', value: 'Perceptron theorem: finite steps IF separable — but the bound depends on the margin (larger margin → faster convergence)' },
        ],
      },
    ],
    notes: [
      'The perceptron margin-robustness argument is exactly why SVM exists: the perceptron is content with any separator, SVM picks the safest one.',
      'Both require separability for the hard versions; soft-margin SVM and the pocket algorithm are their respective relaxations.',
    ],
  },
  {
    id: 'svm-vs-logistic',
    title: 'SVM vs Logistic Regression (geometric margin vs probabilistic score)',
    topics: ['svm-hard-margin', 'logistic-regression'],
    axes: [
      {
        axis: 'Loss / objective',
        entries: [
          { topic: 'svm-hard-margin', value: 'Hinge-like via ½‖w‖² + hard constraints — geometry first, no probabilities' },
          { topic: 'logistic-regression', value: 'Cross-entropy of p = σ(w·x+b) — calibrated probabilities, every point contributes' },
        ],
      },
      {
        axis: 'Data usage',
        entries: [
          { topic: 'svm-hard-margin', value: 'ONLY support vectors shape the boundary (2 of 24 on the default seed)' },
          { topic: 'logistic-regression', value: 'Every point contributes a gradient term — no point is ever "irrelevant"' },
        ],
      },
      {
        axis: 'Output',
        entries: [
          { topic: 'svm-hard-margin', value: 'Class + sign of w·x+b; no calibrated probability (Platt scaling is an add-on)' },
          { topic: 'logistic-regression', value: 'σ(w·x+b) ∈ (0,1) — a probability that doubles as confidence' },
        ],
      },
      {
        axis: 'Boundary at saturation',
        entries: [
          { topic: 'svm-hard-margin', value: 'Stops exactly at the max margin — further data changes nothing (SVs fixed)' },
          { topic: 'logistic-regression', value: 'Keeps pushing probabilities toward 0/1 as epochs continue; boundary drifts with the score scale' },
        ],
      },
    ],
    notes: [
      'Both produce a linear boundary in 2D, and both can be written as score > 0 — the visible difference is WHERE the boundary sits: SVM maximizes the gap, logistic balances the odds.',
      'The soft-margin SVM is closer to logistic than the hard margin is: slack variables behave like a hinge loss.',
    ],
  },
  {
    id: 'svm-vs-lda',
    title: 'SVM vs LDA (distribution-free max-margin vs Gaussian-Bayes)',
    topics: ['svm-hard-margin', 'lda'],
    axes: [
      {
        axis: 'Model assumption',
        entries: [
          { topic: 'svm-hard-margin', value: 'None beyond separability — the boundary is chosen purely by the extreme points' },
          { topic: 'lda', value: 'Both classes are Gaussian with a SHARED covariance; the boundary is then Bayes-optimal' },
        ],
      },
      {
        axis: 'Data used',
        entries: [
          { topic: 'svm-hard-margin', value: 'Support vectors only — the extremes (default seed: 2 points)' },
          { topic: 'lda', value: 'Every point: class means and the pooled covariance matrix' },
        ],
      },
      {
        axis: 'Boundary placement',
        entries: [
          { topic: 'svm-hard-margin', value: 'Maximizes the gap between classes (margin 1.276 on the default seed)' },
          { topic: 'lda', value: 'Placed at equal Mahalanobis distance — optimal ONLY when the Gaussian assumption holds' },
        ],
      },
      {
        axis: 'Robustness',
        entries: [
          { topic: 'svm-hard-margin', value: 'Sensitive to outliers near the boundary (a single outlier can collapse the margin — see failures)' },
          { topic: 'lda', value: 'Means and covariance are smoothed over all points — more robust to one outlier, wrong model otherwise' },
        ],
      },
    ],
    notes: [
      'The classic trade: LDA is the right answer when the Gaussian assumption is true (it is Bayes-optimal), SVM wins when it is not — SVM makes no distributional bet.',
      'Both are linear classifiers; the difference is WHICH statistics of the data they trust (means+spread vs extremes).',
    ],
  },
];
