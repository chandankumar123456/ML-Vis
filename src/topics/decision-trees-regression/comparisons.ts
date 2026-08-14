// src/topics/decision-trees-regression/comparisons.ts
import type { Comparison } from '../../engine/types';

export const dtrComparisons: Comparison[] = [
  {
    id: 'dtr-vs-linear',
    title: 'Regression tree vs linear regression',
    topics: ['decision-trees-regression', 'simple-linear-regression'],
    axes: [
      {
        axis: 'Fitted function',
        entries: [
          { topic: 'decision-trees-regression', value: 'Piecewise-constant step function — flat inside each region, jumps at split thresholds' },
          { topic: 'simple-linear-regression', value: 'A single line ŷ = wx + b — smooth, one slope everywhere' },
        ],
      },
      {
        axis: 'Extrapolation behavior',
        entries: [
          { topic: 'decision-trees-regression', value: 'Constant beyond the observed range (flat tail — a guess, not a trend)' },
          { topic: 'simple-linear-regression', value: 'Line continues with the fitted slope — an explicit trend assumption' },
        ],
      },
      {
        axis: 'Interpretability of the model',
        entries: [
          { topic: 'decision-trees-regression', value: 'Rules "x < 0.42 → 1.1" — legible as if/then statements, depth limits readability' },
          { topic: 'simple-linear-regression', value: 'Slope = change in y per unit x — a single global number' },
        ],
      },
      {
        axis: 'When it wins',
        entries: [
          { topic: 'decision-trees-regression', value: 'Nonlinear, non-smooth, or threshold-structured targets' },
          { topic: 'simple-linear-regression', value: 'Genuinely linear trends with noise' },
        ],
      },
    ],
    notes: [
      'Both are "mean" models in disguise: linear regression predicts the conditional mean under a linearity assumption; a regression tree predicts the conditional mean of each region. The tree trades smoothness for flexibility.',
      'The tree needs many steps to approximate a smooth curve (bias at low depth), while a line can never represent a step at all (bias at any depth).',
    ],
  },
  {
    id: 'dtr-vs-knn',
    title: 'Regression tree vs k-nearest neighbors',
    topics: ['decision-trees-regression', 'knn'],
    axes: [
      {
        axis: 'How the prediction region is found',
        entries: [
          { topic: 'decision-trees-regression', value: 'Axis-aligned rectangles fixed at TRAINING time by greedy SSE splits' },
          { topic: 'knn', value: 'Adaptive balls around the query point — the region changes with every query' },
        ],
      },
      {
        axis: 'Resolution of the fit',
        entries: [
          { topic: 'decision-trees-regression', value: 'Fixed after training: resolution is set by depth/min-leaf, uniform-ish across x' },
          { topic: 'knn', value: 'Adaptive: dense data → small neighborhoods (fine detail); sparse data → large neighborhoods (smooth)' },
        ],
      },
      {
        axis: 'Training / prediction cost',
        entries: [
          { topic: 'decision-trees-regression', value: 'Training = greedy splits; prediction = O(depth) threshold walks' },
          { topic: 'knn', value: 'Training = store the data; prediction = O(n·d) distance scan' },
        ],
      },
      {
        axis: 'Smoothness of predictions',
        entries: [
          { topic: 'decision-trees-regression', value: 'Discontinuous steps at thresholds' },
          { topic: 'knn', value: 'Continuous (averaging over moving neighborhoods) but with kinks' },
        ],
      },
    ],
    notes: [
      'Both are non-parametric local-constant models; the tree commits to a static partition, kNN defers partitioning until query time.',
      'The tree\'s axis-aligned cuts are the limitation the plan\'s failure list captures: kNN bends its regions to the data, the tree cannot rotate its boundaries.',
    ],
  },
  {
    id: 'dtr-vs-svm-regression',
    title: 'Regression tree vs support vector regression (SVR)',
    topics: ['decision-trees-regression', 'svm-soft-margin'],
    axes: [
      {
        axis: 'Loss function',
        entries: [
          { topic: 'decision-trees-regression', value: 'Squared error (SSE) — every residual counts, outliers squared' },
          { topic: 'svm-soft-margin', value: 'ε-insensitive tube — residuals smaller than ε cost nothing' },
        ],
      },
      {
        axis: 'Prediction surface',
        entries: [
          { topic: 'decision-trees-regression', value: 'Step function (piecewise constant, discontinuous)' },
          { topic: 'svm-soft-margin', value: 'Kernel-dependent smooth surface (linear / RBF / polynomial)' },
        ],
      },
      {
        axis: 'Robustness to outliers',
        entries: [
          { topic: 'decision-trees-regression', value: 'Weak — one extreme y shifts an entire leaf mean' },
          { topic: 'svm-soft-margin', value: 'Strong — outliers inside the ε-tube cost zero, beyond it cost linearly (not squared)' },
        ],
      },
      {
        axis: 'Scalability of the model description',
        entries: [
          { topic: 'decision-trees-regression', value: 'Compact rule set (depth × thresholds) — interpretable' },
          { topic: 'svm-soft-margin', value: 'Support vectors only — compact but opaque' },
        ],
      },
    ],
    notes: [
      'SVR (regression flavor of the SVM family) buys smoothness and outlier tolerance with a convex-but-opaque kernel model; the tree buys interpretable if/then rules with squared-error fragility.',
      'The tree and the ε-insensitive SVM sit at opposite ends of the interpretability / smoothness spectrum for the same 1-D regression task.',
    ],
  },
];