// src/topics/knn/failures.ts
import type { FailureDemo } from '../../engine/types';

// Failure demos mirror the Wave-1 convention: each demo carries its OWN params
// (so the failure reproduces deterministically) + narration + whyItBreaks.
export const knnFailureDemos: FailureDemo[] = [
  {
    id: 'knn-fail-high-dim',
    title: 'Curse of Dimensionality: "nearest" stops meaning near',
    scenario: 'curse-of-dimensionality',
    params: { k: 5, nPerClass: 12, margin: 1.0, metric: 'euclidean', seed: 42 },
    narration: 'As the number of features d grows, distances concentrate: in the d-dimensional unit cube the inner (1−2ε)-cube holds only (1−2ε)^d of the volume, and the ratio of farthest to nearest neighbor distance tends to 1. Every point looks equidistant, so the vote collapses toward the class prior — adding noise features makes this visible: the boundary roughens for any k while k=1 train error stays 0.',
    whyItBreaks: 'k-NN is a distance-based method: it needs distances to DISCRIMINATE. Volume concentration kills the discrimination (max/min distance → 1), and distance terms from irrelevant features drown the informative ones. Feature selection / PCA is the fix — the model cannot learn which dimensions matter.',
  },
  {
    id: 'knn-fail-imbalance',
    title: 'Class Imbalance: the majority class steals the vote',
    scenario: 'class-imbalance',
    params: { k: 7, points: '[[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[2,0,0],[0,2,0],[-2,0,0],[0,-2,0],[3,0,0],[-3,0,0],[-1,-1,1],[1,1,1],[-1,1,1],[1,-1,1]]', queryX: 0, queryY: 0, metric: 'euclidean', seed: 42 },
    narration: 'With 11 class-0 points against 4 class-1 points, the k=7 neighborhood around the query is majority-heavy even in genuinely minority territory: the minority class sits inside a sea of majority Voronoi cells and gets few correct votes.',
    whyItBreaks: 'k-NN assumes neighborhoods are balanced representatives. With n_major ≫ n_minor the expected majority count inside any radius is larger, so large k systematically under-predicts the minority class. Distance-weighted voting, stratification or a threshold shift is the fix.',
  },
  {
    id: 'knn-fail-noisy-feature',
    title: 'Noisy / irrelevant features poison the distance',
    scenario: 'noisy-features',
    params: { k: 5, nPerClass: 12, margin: 1.0, metric: 'euclidean', seed: 42 },
    narration: 'A label-independent feature contributes as large an L2 term as the informative features, so the k nearest neighbors are selected partly by pure noise. Small k makes this catastrophic (one noise-flipped neighbor decides the class); large k averages noise away but blurs the real boundary too.',
    whyItBreaks: 'Distances do not know which axis carries signal — every feature counts equally. The informative axes are diluted by noise axes, so the neighbor relation is partially random. Feature selection, PCA, or a learned (weighted) metric is the fix.',
  },
  {
    id: 'knn-fail-large-n',
    title: 'Large n: every query scans the whole dataset (O(n·d))',
    scenario: 'large-n-inference',
    params: { k: 5, nPerClass: 15, margin: 1.0, metric: 'euclidean', seed: 42 },
    narration: 'k-NN stores every training point, so a deployment with millions of instances pays O(n·d) per query — exactly backwards from a trained model, which pays the training cost once and then predicts cheaply. kd-tree/ball-tree indexes help only in low dimensions (they degrade toward linear scan as d grows).',
    whyItBreaks: 'The lazy-learner trade-off: training is O(1), inference is O(n·d). There is no learned summary to compress away the dataset. Data condensation (keep prototypes), approximate nearest-neighbour (LSH) or switching to an eager model is the fix.',
  },
];
