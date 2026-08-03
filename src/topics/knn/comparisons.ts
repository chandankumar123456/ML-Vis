// src/topics/knn/comparisons.ts
import type { Comparison } from '../../engine/types';

export const knnComparisons: Comparison[] = [
  {
    id: 'knn-vs-naive-bayes',
    title: 'k-NN vs Naive Bayes (lazy instance-based vs eager parametric)',
    topics: ['knn', 'naive-bayes'],
    axes: [
      {
        axis: 'Learning style',
        entries: [
          { topic: 'knn', value: 'Lazy — stores instances, no parameters learned (training is O(1))' },
          { topic: 'naive-bayes', value: 'Eager — learns P(C) and P(x|C) from data in one pass' },
        ],
      },
      {
        axis: 'Decision boundary',
        entries: [
          { topic: 'knn', value: 'Arbitrary, data-driven (Voronoi cells); can fit any shape with enough data' },
          { topic: 'naive-bayes', value: 'Depends on the likelihood model; Gaussian NB gives quadratic boundaries' },
        ],
      },
      {
        axis: 'Assumptions',
        entries: [
          { topic: 'knn', value: 'Distance is meaningful (needs scaling); no distributional assumption' },
          { topic: 'naive-bayes', value: 'Conditional independence of features given class' },
        ],
      },
      {
        axis: 'Prediction cost',
        entries: [
          { topic: 'knn', value: 'O(n·d) per query — slows with dataset size' },
          { topic: 'naive-bayes', value: 'O(d) per query — parameter lookups only' },
        ],
      },
    ],
    notes: [
      'Both are conceptually simple and easy to implement — a common GATE comparison pair.',
      'NB is fast at inference but pays for a (possibly wrong) independence assumption; k-NN assumes nothing but pays O(n·d) per query.',
    ],
  },
  {
    id: 'knn-vs-decision-tree',
    title: 'k-NN vs Decision Tree (boundary shape)',
    topics: ['knn', 'decision-tree'],
    axes: [
      {
        axis: 'Boundary geometry',
        entries: [
          { topic: 'knn', value: 'Free-form Voronoi cells following the local data density' },
          { topic: 'decision-tree', value: 'Axis-aligned rectangles — each split is on ONE feature' },
        ],
      },
      {
        axis: 'Feature scaling',
        entries: [
          { topic: 'knn', value: 'Critical — distances mix feature units' },
          { topic: 'decision-tree', value: 'Unnecessary — splits compare one feature to one threshold' },
        ],
      },
      {
        axis: 'k / depth = complexity dial',
        entries: [
          { topic: 'knn', value: 'k small → overfit; k large → smooth (bias–variance via k)' },
          { topic: 'decision-tree', value: 'Depth small → underfit; depth large → overfit (pruning needed)' },
        ],
      },
    ],
    notes: [
      'k-NN captures diagonal/non-axis-aligned patterns naturally; a decision tree needs many splits to approximate a 45° boundary.',
      'Trees are the classic "feature scaling irrelevant" contrast to k-NN\'s "scaling critical".',
    ],
  },
  {
    id: 'knn-vs-svm',
    title: 'k-NN vs SVM (boundary smoothness and margin)',
    topics: ['knn', 'svm'],
    axes: [
      {
        axis: 'Boundary',
        entries: [
          { topic: 'knn', value: 'Piecewise-linear Voronoi cells; roughness controlled by k' },
          { topic: 'svm', value: 'Max-margin hyperplane (linear) or kernel surface; explicitly smooth/regularized' },
        ],
      },
      {
        axis: 'Representation',
        entries: [
          { topic: 'knn', value: 'Stores ALL training points — n memory, O(n·d) query' },
          { topic: 'svm', value: 'Keeps only support vectors — the points on the margin' },
        ],
      },
      {
        axis: 'Guarantee',
        entries: [
          { topic: 'knn', value: 'No margin notion; a point right on the boundary flips easily' },
          { topic: 'svm', value: 'Maximizes the margin → most robust to small perturbations' },
        ],
      },
    ],
    notes: [
      'The simulation\'s k-sweep shows exactly why SVM\'s margin objective exists: k=1\'s boundary is fragile to noise, while a margin-maximizing line stays stable.',
      'Both are distance-ish (SVM kernels are inner products), but SVM keeps only support vectors while k-NN keeps everything.',
    ],
  },
];
