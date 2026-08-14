// src/topics/hierarchical-clustering/comparisons.ts
// Measured anchors cited below:
//   default run (n 12, single, blobs 2, seed 42): cut@1.2 → 2 clusters
//     (the two blobs); the height gap 0.896 → 1.795 is the natural cut.
import type { Comparison } from '../../engine/types';

export const hierarchicalComparisons: Comparison[] = [
  {
    id: 'hc-vs-kmeans',
    title: 'Hierarchical Clustering vs k-means (nested tree vs flat partition)',
    topics: ['hierarchical-clustering', 'kmeans'],
    axes: [
      {
        axis: 'Number of clusters k',
        entries: [
          { topic: 'hierarchical-clustering', value: 'Not needed upfront — the dendrogram encodes EVERY k; pick k by cutting at a height (cut@1.2 on the default run gives the 2 blobs; cut at 0 gives 12)' },
          { topic: 'kmeans', value: 'Required before running; every k needs a fresh run and initialization, with elbow/silhouette used afterwards to justify the choice' },
        ],
      },
      {
        axis: 'Structure produced',
        entries: [
          { topic: 'hierarchical-clustering', value: 'A nested hierarchy (tree) of partitions — clusters at every scale, directly comparable across cut heights' },
          { topic: 'kmeans', value: 'A single flat partition into k disjoint convex (Voronoi) cells' },
        ],
      },
      {
        axis: 'Shape assumptions',
        entries: [
          { topic: 'hierarchical-clustering', value: 'Linkage-dependent: single chains arbitrarily, complete/ward prefer compact spherical clusters' },
          { topic: 'kmeans', value: 'Assumes roughly spherical, equal-sized clusters (minimizes SSE around centroids)' },
        ],
      },
      {
        axis: 'Objective',
        entries: [
          { topic: 'hierarchical-clustering', value: 'Greedy local merges under a linkage criterion (Ward is the greedy version of the k-means SSE objective)' },
          { topic: 'kmeans', value: 'Global SSE = ΣΣ‖x−μ‖² minimized by alternating assignment/update (local optima, init-dependent)' },
        ],
      },
    ],
    notes: [
      'The classic GATE contrast: k-means needs k and gives one answer; hierarchical needs no k and gives a family of answers. The nested structure (plan: "no k needed, nested structure") is hierarchical\'s defining advantage.',
      'Ward linkage and k-means share the SSE objective — the Ward tree is a greedy bottom-up version of k-means; a k-means run on the same data is an upper bound on the SSE any cut of the Ward tree achieves.',
    ],
  },
  {
    id: 'hc-vs-dbscan',
    title: 'Hierarchical Clustering vs DBSCAN (shape flexibility vs density)',
    topics: ['hierarchical-clustering', 'dbscan'],
    axes: [
      {
        axis: 'Cluster shapes',
        entries: [
          { topic: 'hierarchical-clustering', value: 'Depends on linkage: single linkage CAN find elongated/concave shapes (chains) but is fragile; complete/ward prefer compact blobs' },
          { topic: 'dbscan', value: 'Finds ARBITRARY shapes natively — clusters are density-connected regions; no shape assumption at all' },
        ],
      },
      {
        axis: 'Parameters',
        entries: [
          { topic: 'hierarchical-clustering', value: 'Linkage criterion + cut height (no notion of noise vs core points)' },
          { topic: 'dbscan', value: 'ε (neighborhood radius) and minPts (density threshold) — plus an explicit NOISE label for points in sparse regions' },
        ],
      },
      {
        axis: 'Outliers',
        entries: [
          { topic: 'hierarchical-clustering', value: 'Every point belongs to some cluster at every cut — outliers become singleton clusters that merge late (measured in the failures demo)' },
          { topic: 'dbscan', value: 'Outliers are labeled NOISE and excluded — density, not connectivity, decides membership' },
        ],
      },
      {
        axis: 'Result structure',
        entries: [
          { topic: 'hierarchical-clustering', value: 'A tree — one run, every scale' },
          { topic: 'dbscan', value: 'A single partition for the chosen (ε, minPts)' },
        ],
      },
    ],
    notes: [
      'DBSCAN is the "arbitrary shapes + noise" answer to the plan\'s question; hierarchical answers "how many clusters, and how are they nested?". DBSCAN has no nested structure and hierarchical has no noise concept.',
      'Single linkage on the two-half-moons shape behaves like a poor-man\'s DBSCAN (both exploit connectivity), but without a noise label it will still attach every outlier.',
    ],
  },
  {
    id: 'hc-vs-gmm',
    title: 'Hierarchical Clustering vs Gaussian Mixture Models (hard tree vs soft probabilities)',
    topics: ['hierarchical-clustering', 'gmm'],
    axes: [
      {
        axis: 'Membership type',
        entries: [
          { topic: 'hierarchical-clustering', value: 'HARD — every point belongs to exactly one cluster at every cut; merges are deterministic and discrete' },
          { topic: 'gmm', value: 'SOFT — every point carries a probability of belonging to each Gaussian component (responsibilities), refined by EM' },
        ],
      },
      {
        axis: 'Assumptions',
        entries: [
          { topic: 'hierarchical-clustering', value: 'Only a distance/linkage choice — no distributional model of the data' },
          { topic: 'gmm', value: 'The data is a mixture of Gaussians (parameters μ_k, Σ_k, π_k) — a full generative model' },
        ],
      },
      {
        axis: 'Output',
        entries: [
          { topic: 'hierarchical-clustering', value: 'A dendrogram — cluster structure at all scales, but no uncertainty estimates' },
          { topic: 'gmm', value: 'Component parameters + per-point probabilities — uncertainty and density estimates, but k must be chosen (like k-means) and EM can stick in local optima' },
        ],
      },
    ],
    notes: [
      'The plan\'s "hard vs soft assignment" axis: hierarchical commits each point at each cut; GMM hedges with probabilities. Both are unsupervised; GMM additionally models the data distribution (usable for density estimation and generative sampling).',
      'The Ward-SSE and GMM objectives are related: both measure fit around centers, but GMM allows elliptical components (Σ_k) and soft memberships, so it handles overlap and unequal spreads that a hard hierarchical cut smooths over.',
    ],
  },
];
