// src/topics/naive-bayes/comparisons.ts
import type { Comparison } from '../../engine/types';

export const nbComparisons: Comparison[] = [
  {
    id: 'nb-cmp1',
    title: 'Naive Bayes vs Logistic Regression: generative vs discriminative',
    topics: ['naive-bayes', 'logistic-regression'],
    axes: [
      {
        axis: 'What is modeled',
        entries: [
          { topic: 'naive-bayes', value: 'Generative: fits the joint P(x, C) = P(C)·P(x|C), then derives P(C|x) by Bayes' },
          { topic: 'logistic-regression', value: 'Discriminative: fits P(C|x) directly without modeling the feature distribution' },
        ],
      },
      {
        axis: 'Parameter estimate',
        entries: [
          { topic: 'naive-bayes', value: 'Closed form from per-class statistics (means, variances, counts)' },
          { topic: 'logistic-regression', value: 'Iterative (gradient descent on cross-entropy)' },
        ],
      },
      {
        axis: 'Feature assumptions',
        entries: [
          { topic: 'naive-bayes', value: 'Conditional independence of features given the class (the naive assumption)' },
          { topic: 'logistic-regression', value: 'None explicitly; learns feature weights' },
        ],
      },
      {
        axis: 'Small-sample behavior',
        entries: [
          { topic: 'naive-bayes', value: 'Can estimate from very few samples (few statistics per class)' },
          { topic: 'logistic-regression', value: 'Needs enough samples to fit the weight vector reliably' },
        ],
      },
    ],
    notes: [
      'The classic trade-off: NB converges to its (restrictive) model faster; logistic converges to the true conditional distribution given enough data.',
      'On the simulator\u2019s correlated data, logistic would fit the tilt in the boundary; naive Bayes cannot because it is structurally diagonal-covariance.',
    ],
  },
  {
    id: 'nb-cmp2',
    title: 'Naive Bayes vs K-Nearest Neighbors: parametric vs instance-based',
    topics: ['naive-bayes', 'knn'],
    axes: [
      {
        axis: 'Training',
        entries: [
          { topic: 'naive-bayes', value: 'Parametric: fits a fixed number of parameters (per-class means/variances/counts), then discards the data' },
          { topic: 'knn', value: 'Instance-based: stores every training point; there is no model to fit' },
        ],
      },
      {
        axis: 'Prediction cost',
        entries: [
          { topic: 'naive-bayes', value: 'O(d) per query — a product of likelihoods' },
          { topic: 'knn', value: 'O(N·d) per query — distance to every stored point' },
        ],
      },
      {
        axis: 'Decision surface',
        entries: [
          { topic: 'naive-bayes', value: 'Smooth, defined by the class densities (quadratic for Gaussian NB)' },
          { topic: 'knn', value: 'Piecewise constant Voronoi-like regions' },
        ],
      },
      {
        axis: 'Interpretability',
        entries: [
          { topic: 'naive-bayes', value: 'Explicit probabilities and per-feature evidence' },
          { topic: 'knn', value: 'No probability model; hard to explain why a neighbor wins' },
        ],
      },
    ],
    notes: [
      'Both are simple baselines; NB generalizes better on small data, KNN needs no distributional assumption.',
      'KNN is non-parametric and can represent arbitrary boundaries, but degrades as dimensionality grows (curse of dimensionality).',
    ],
  },
  {
    id: 'nb-cmp3',
    title: 'Naive Bayes vs Decision Trees: probability model vs recursive partitioning',
    topics: ['naive-bayes', 'decision-trees'],
    axes: [
      {
        axis: 'Feature interactions',
        entries: [
          { topic: 'naive-bayes', value: 'Assumes independence — correlated features are double-counted' },
          { topic: 'decision-trees', value: 'Captures interactions by splitting on different features at different depths' },
        ],
      },
      {
        axis: 'Output',
        entries: [
          { topic: 'naive-bayes', value: 'Calibrated class probabilities from Bayes theorem' },
          { topic: 'decision-trees', value: 'Class proportions in leaves (coarse probabilities, often miscalibrated)' },
        ],
      },
      {
        axis: 'Data efficiency',
        entries: [
          { topic: 'naive-bayes', value: 'Works with few samples; smoothing handles sparse counts' },
          { topic: 'decision-trees', value: 'Needs enough samples per leaf to split reliably; deep trees overfit' },
        ],
      },
      {
        axis: 'Decision surface',
        entries: [
          { topic: 'naive-bayes', value: 'Smooth curves defined by densities' },
          { topic: 'decision-trees', value: 'Axis-aligned rectangular regions' },
        ],
      },
    ],
    notes: [
      'The simulator\u2019s true boundary is tilted (correlated features): a tree approximates it with axis-aligned steps, NB approximates it with a vertical line — neither is exact, but for different reasons.',
      'Trees are the go-to when features interact and you want interpretable if-then rules.',
    ],
  },
];
