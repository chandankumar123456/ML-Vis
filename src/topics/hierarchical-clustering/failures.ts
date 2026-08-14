// src/topics/hierarchical-clustering/failures.ts
// Every numeric claim below is measured by running the module on the demo params:
//   - F1 chaining: six collinear points 0..5, single linkage → five merges at
//     height 1.000 each (the cluster grows one point at a time; no structure).
//   - F3 outlier: a (5,5) point next to a tight blob → blob merges at
//     0.224–0.361, the outlier stays a singleton until the final merge at
//     6.727 (cut below 6.727 isolates it as its own cluster).
import type { FailureDemo } from '../../engine/types';

export const hierarchicalFailureDemos: FailureDemo[] = [
  {
    id: 'hc-fail-chaining',
    title: 'Single linkage chains: elongated "snake" clusters hide the true structure',
    scenario: 'chaining',
    params: {
      n: 6, linkage: 'single', blobs: '2', seed: 42,
      points: '[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]]',
    },
    narration: 'Six points lie on a line with uniform spacing 1.0. Single linkage merges the nearest pair (p0,p1) at 1.0, then the growing cluster absorbs p2 at 1.0, then p3, p4, p5 — all five merges at exactly height 1.000. The dendrogram is a staircase: the tree reveals NO structure, because every merge is equally close. There are no "two clusters" here to find — the cut slider can only trade chain length for chain length.',
    whyItBreaks: 'Single linkage defines distance as the closest pair, so a linear chain of points is merged one element at a time: each new point is closer to the growing cluster than any gap inside it. The result is the plan\'s classic "chaining" failure. Mitigations: switch to complete or Ward linkage (which force compactness), or standardize the features first — chains also appear when one axis dominates the distances.',
  },
  {
    id: 'hc-fail-complexity',
    title: 'O(n³) agglomerative cost: the naive implementation does not scale',
    scenario: 'complexity',
    params: { n: 20, linkage: 'single', blobs: '2', seed: 42 },
    narration: 'Every merge re-evaluates every remaining pair of clusters from the raw point sets: O(n²) distances × O(n) merges = O(n³) total. At n = 20 the module evaluates 190 pair distances on the first merge, 171 on the second, … — 1,330 evaluations over the 19 merges, each scanning up to |A|·|B| point pairs. That is why validateParams caps n at 20: the demo stays instant, but the cost grows cubically — n = 200 would be ~1,000× slower (≈ O(n³) = 8,000,000× the distance scans at n = 20).',
    whyItBreaks: 'The distance matrix alone is O(n²) memory and the naive linkage recomputation is O(n³) time — both explode on large data. Production implementations use Lance-Williams updates (O(1) per pair, O(n² log n) with a heap for nearest-neighbor search) — the formula the derivations state. The honest fix here is the n cap; the conceptual fix is the update formula.',
  },
  {
    id: 'hc-fail-outliers',
    title: 'Noisy outliers become singleton clusters that merge very late',
    scenario: 'outliers',
    params: {
      n: 10, linkage: 'single', blobs: '2', seed: 42,
      points: '[[0,0],[0.5,0],[-0.3,0.4],[0.2,-0.3],[-0.4,-0.1],[0,0.5],[-0.2,0.2],[0.4,-0.2],[-0.1,-0.4],[5,5]]',
    },
    narration: 'Nine points form a tight blob around the origin while one outlier sits at (5,5) — at distance ≈ 6.73 from every blob point. The blob merges internally first (heights 0.224–0.361), and the outlier stays a SINGLETON cluster until the very last merge, which joins it at height 6.727. Cutting below 6.727 (e.g. cutHeight 2.0) leaves the outlier as its own one-point cluster: an honest singleton, but one that distorts the dendrogram\'s height axis — the axis auto-ranges to 6.727, compressing all the real structure into the bottom third.',
    whyItBreaks: 'Hierarchical clustering has NO noise label (contrast DBSCAN): every point must belong somewhere at every cut, so an outlier survives as a singleton that merges last. Its merge height inflates the axis and can mislead cut choices (a gap at 7.1 looks like a great cut — it only separates the blob from the outlier). Mitigations: pre-filter obvious outliers, use a robust distance/standardization, or pick a method with an explicit noise class.',
  },
];
