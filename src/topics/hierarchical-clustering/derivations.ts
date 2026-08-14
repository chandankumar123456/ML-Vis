// src/topics/hierarchical-clustering/derivations.ts
// Measured anchors cited below (default run — n 12, single, blobs 2, seed 42):
//   merge heights [0.241 … 1.795], copheneticCorr = 0.901.
//   Hand-computed dataset A: heights [0.5, 1, 2], copheneticCorr ≈ 0.8985.
//   Ward (n 10, blobs 2, seed 42): SSE-increase heights, non-decreasing.
import type { Derivation } from '../../engine/types';

export const hierarchicalDerivations: Derivation[] = [
  {
    id: 'hc-single-min-pairwise',
    title: 'Single Linkage = Min Pairwise Distance, and the Min-Update',
    steps: [
      {
        latex: 'd_{\\text{single}}(A, B) = \\min_{a \\in A,\\, b \\in B} d(a, b)',
        justification: 'Definition: the cost of merging A and B is the distance of their CLOSEST pair of points — the "nearest neighbors" rule. On dataset A (p0…p3 collinear at 0, 1, 3, 3.5) the closest pair overall is (p2, p3) at 0.5, so the first merge joins them.',
      },
      {
        latex: '\\text{after merging } A \\cup B: \\quad d_{\\text{single}}(A \\cup B, C) = \\min_{a \\in A \\cup B,\\, c \\in C} d(a, c)',
        justification: 'The new cluster A∪B inherits every member of both. Its distance to any third cluster C is the closest pair across the union — which is exactly the minimum over the two sub-problems.',
      },
      {
        latex: '\\min_{a \\in A \\cup B,\\, c \\in C} d(a, c) = \\min\\big(\\min_{a \\in A,\\, c \\in C} d(a, c), \\; \\min_{b \\in B,\\, c \\in C} d(b, c)\\big) = \\min\\big(d_{\\text{single}}(A, C), \\, d_{\\text{single}}(B, C)\\big)',
        justification: 'The minimum over a union equals the minimum of the two minima (min distributes over set union). This is the Lance-Williams update for single linkage with (α_A, α_B, β, γ) = (½, ½, 0, −½): D(A∪B, C) = min(D(A,C), D(B,C)). On dataset A, after merging (p2, p3) at 0.5, the distance from {p2, p3} to p0 is min(3, 3.5) = 3 and to p1 is min(2, 2.5) = 2 — so the next merge is (p0, p1) at 1.',
      },
      {
        latex: 'd_{\\text{single}}(A, B) \\le \\max(d_{\\text{single}}(A, C), d_{\\text{single}}(B, C)) \\;\\Rightarrow\\; \\text{monotone heights}',
        justification: 'The reducibility inequality — merging never creates a cluster closer to a third cluster than the closest child was. Hence single-linkage merge heights are non-decreasing and the dendrogram grows upward (measured: [0.5, 1, 2] on dataset A, strictly increasing).',
      },
    ],
    derivedFrom: ['hc-linkage-single'],
  },
  {
    id: 'hc-complete-max-pairwise',
    title: 'Complete Linkage = Max Pairwise Distance, and the Max-Update',
    steps: [
      {
        latex: 'd_{\\text{complete}}(A, B) = \\max_{a \\in A,\\, b \\in B} d(a, b)',
        justification: 'Definition: the cost of merging A and B is the distance of their FARTHEST pair — the "farthest neighbors" / diameter rule. The mirror image of single linkage: min becomes max.',
      },
      {
        latex: 'd_{\\text{complete}}(A \\cup B, C) = \\max_{a \\in A \\cup B,\\, c \\in C} d(a, c) = \\max\\big(d_{\\text{complete}}(A, C), \\, d_{\\text{complete}}(B, C)\\big)',
        justification: 'Exactly as in the single-linkage derivation, the max over a union is the max of the two maxima — the Lance-Williams update with γ = +½: D(A∪B, C) = max(D(A,C), D(B,C)). On dataset B, d({p2,p3}, p1) = max(1.044, 1.217) = 1.217 exceeds d(p0,p1) = 1.118, so complete linkage merges (p0, p1) FIRST while single linkage chains p1 onto the tight pair — the measurable structure difference.',
      },
      {
        latex: '\\text{complete heights} \\ge \\text{single heights (on the same data, same merge list)}',
        justification: 'For ANY pair of clusters max ≥ min, so every complete-linkage merge cost is at least the single-linkage cost of the same pair. When the merge ORDER differs too (dataset B), the final heights diverge even more: 2.220 vs 1.118 (measured).',
      },
    ],
    derivedFrom: ['hc-linkage-complete', 'hc-linkage-single'],
  },
  {
    id: 'hc-ward-sse-increase',
    title: "Ward's Merge Cost = The SSE Increase (Variance, Not Distance)",
    steps: [
      {
        latex: 'SSE(C) = \\sum_{x \\in C} \\|x - \\mu_C\\|^2, \\quad \\mu_C = \\frac{1}{|C|} \\sum_{x \\in C} x',
        justification: 'The within-cluster sum of squares of a single cluster — the same quantity k-means minimizes over all clusters. Ward\'s method merges the pair whose SSE increase is smallest.',
      },
      {
        latex: '\\Delta SSE(A, B) = SSE(A \\cup B) - SSE(A) - SSE(B)',
        justification: 'The increase is the new cluster\'s SSE minus the SSE the two clusters already contributed. Everything else in the dataset is untouched, so this is the TOTAL change in the global objective.',
      },
      {
        latex: '\\sum_{x \\in A \\cup B} \\|x - \\mu_{A \\cup B}\\|^2 = \\sum_{x \\in A} \\|x - \\mu_A\\|^2 + \\sum_{x \\in B} \\|x - \\mu_B\\|^2 + \\frac{|A||B|}{|A| + |B|} \\|\\mu_A - \\mu_B\\|^2',
        justification: 'The standard two-cluster decomposition: the pooled variance is the sum of the within-variances plus a between-term. Expand each sum via ‖x − μ‖² = ‖x − μ_A‖² + ‖μ_A − μ_{A∪B}‖² + 2(x − μ_A)·(μ_A − μ_{A∪B}); the linear terms telescope to zero, leaving exactly the between-cluster term.',
      },
      {
        latex: '\\Delta SSE(A, B) = \\frac{|A| \\cdot |B|}{|A| + |B|} \\cdot \\|\\mu_A - \\mu_B\\|^2',
        justification: 'Subtract SSE(A) + SSE(B) from both sides: the within terms cancel and only the size-weighted squared centroid distance remains — the closed form used by the module. NOTE THE UNITS: a squared quantity, NOT a distance. Ward dendrogram heights live on an "SSE increase" axis; comparing them to single/complete heights is the plan\'s trap mistake.',
      },
    ],
    derivedFrom: ['hc-linkage-ward'],
  },
  {
    id: 'hc-cophenetic-distance',
    title: 'Cophenetic Distance = Lowest-Common-Ancestor Merge Height',
    steps: [
      {
        latex: 'c_{ij} = h(m^*) \\quad \\text{where } m^* \\text{ is the FIRST merge containing both } i \\text{ and } j',
        justification: 'In the merge tree each pair of leaves has exactly one lowest common ancestor — the merge whose subtree first contains both points. Its height h(m*) is the cophenetic distance: the level at which the tree joins them.',
      },
      {
        latex: 'c_{ij} \\ge \\max(d_{\\text{path constraints}}) \\;\\Rightarrow\\; \\text{for every merge } m, \\text{ all pairs inside } m \\text{ have } c \\le h(m)',
        justification: 'Once i and j are co-members (from merge m* on), every later merge also contains them, so their cophenetic distance is fixed at the EARLIEST height — c is a "first joining time", like the coalescence time in population genetics.',
      },
      {
        latex: 'r = \\operatorname{corr}\\big(\\{d_{ij}\\}_{i<j}, \\{c_{ij}\\}_{i<j}\\big)',
        justification: 'Collect the n(n−1)/2 off-diagonal (d, c) pairs and take their Pearson correlation. On dataset A the pairs are (d, c) = (1,1), (3,2), (3.5,2), (2,2), (2.5,2), (0.5,0.5) → r ≈ 0.8985 (hand-computed). The default run measures r = 0.901. r ≈ 1 means the dendrogram preserves the distance ordering.',
      },
    ],
    derivedFrom: ['hc-cophenetic', 'hc-cophenetic-corr'],
  },
];
