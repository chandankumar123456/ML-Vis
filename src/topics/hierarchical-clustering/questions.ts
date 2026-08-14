// src/topics/hierarchical-clustering/questions.ts
// Measured anchors cited below:
//   dataset A (4 collinear points 0,1,3,3.5): single-linkage merge order
//     (p2,p3)@0.5 → (p0,p1)@1 → final@2; copheneticCorr ≈ 0.8985.
//   default run (n 12, single, blobs 2, seed 42): heights [0.241 … 1.795],
//     copheneticCorr = 0.901, cut@1.2 → 2 clusters.
//   dataset B: single final height 1.118, complete final height 2.220.
import type { Question } from '../../engine/types';

export const hierarchicalQuestions: Question[] = [
  {
    id: 'hc-001',
    mode: 'nat',
    prompt: 'Four points lie on a line at positions p0=0, p1=1, p2=3, p3=3.5. Using SINGLE linkage, what is the height of the THIRD (final) merge? (Enter the number.)',
    answer: 2,
    tolerance: 0.01,
    explanation: 'Distances: d(p2,p3)=0.5 (nearest pair → first merge at 0.5, forming {p2,p3}), d(p0,p1)=1 (next nearest → second merge at 1, forming {p0,p1}). The final merge joins the two clusters at min over all cross pairs = min(d(p0,p2)=3, d(p0,p3)=3.5, d(p1,p2)=2, d(p1,p3)=2.5) = 2. Merge heights: [0.5, 1, 2] — the answer is 2.',
    concepts: ['hierarchical clustering', 'single linkage', 'merge order'],
    difficulty: 2,
    tags: ['numerical', 'formula'],
  },
  {
    id: 'hc-002',
    mode: 'conceptual-mcq',
    prompt: 'Why does SINGLE linkage produce "chaining" — elongated, snake-like clusters?',
    options: [
      'Because it merges based on the CLOSEST pair of members, a chain of points can be absorbed one at a time even when the clusters as wholes are far apart',
      'Because it computes distances from cluster centroids, which drift along the data',
      'Because it only ever merges the two points that are farthest apart, pulling the cluster into a line',
      'Because it uses the mean pairwise distance, which averages out the shape of each cluster',
    ],
    answer: 'A',
    explanation: 'Single linkage only looks at the nearest neighbor pair. Two dense blobs linked by a thin line of intermediate points get absorbed point-by-point: each intermediate point is closer to the growing cluster than the next blob is to it, so the merge heights stay low and the tree "chains". This is measurable: the failure demo chains six collinear points at five merges all near height 1, while complete linkage on the same data would require the full 5.0 diameter before merging the ends.',
    trapExplanations: {
      B: 'Single linkage never uses centroids — centroid-based merging is Ward\'s / k-means behavior, and it does NOT chain this way.',
      C: 'Farthest-pair merging is COMPLETE linkage; it produces compact clusters and actively resists chaining.',
      D: 'Mean pairwise distance is AVERAGE linkage (UPGMA) — a compromise that chains less than single linkage.',
    },
    concepts: ['hierarchical clustering', 'single linkage', 'chaining'],
    difficulty: 3,
    tags: ['conceptual', 'trap'],
  },
  {
    id: 'hc-003',
    mode: 'visual',
    prompt: 'Run the default simulation (n 12, single linkage, 2 blobs, seed 42) and scrub to the final snapshot. The merge heights end at 1.795. Why is cutting at cutHeight = 1.2 a sensible choice, and what does it produce?',
    options: [
      'Because the largest gap in the heights is between 0.896 and 1.795 — cutting inside that gap at 1.2 separates the two Gaussian blobs (2 clusters)',
      'Because 1.2 is roughly the median merge height, and cutting at the median always reveals the natural clusters',
      'Because the cophenetic correlation is 0.901, so a cut at 1.2 must give the best score',
      'Because single linkage always has exactly two natural clusters, regardless of the heights',
    ],
    answer: 'A',
    explanation: 'The measured heights are [0.241, 0.266, 0.353, 0.543, 0.594, 0.596, 0.693, 0.733, 0.881, 0.896, 1.795]. The within-blob merges are all below 0.896; the final blob-bridging merge (1.795) is far above. Cutting at 1.2 sits inside the 0.896→1.795 gap, keeping the ten within-blob merges and dropping the inter-blob one — exactly 2 clusters (the final snapshot colors them). The dendrogram shows a horizontal line at 1.2 crossing two vertical branches.',
    trapExplanations: {
      B: 'The median merge height is 0.596 — cutting there yields 6 clusters, not the natural 2. A median cut is not a structure-aware cut.',
      C: 'The cophenetic correlation measures how well the tree preserves distances — it does not select a cut height or score a cut.',
      D: 'The number of clusters comes from the cut height, not from the linkage; single linkage on 3 blobs has three natural clusters.',
    },
    concepts: ['dendrogram', 'cut height', 'visual reading'],
    difficulty: 2,
    tags: ['visual'],
  },
  {
    id: 'hc-004',
    mode: 'gate-mcq',
    prompt: "In Ward's method the merge cost is ΔSSE = (|A|·|B|)/(|A|+|B|) · ‖μ_A − μ_B‖². Which statement is TRUE?",
    options: [
      "Ward's merge heights are in SQUARED units (an SSE increase), so they are not directly comparable to single/complete linkage heights in the same dendrogram axis",
      "Ward's method minimizes the maximum pairwise distance between the two clusters",
      "Ward's method is identical to average linkage when the clusters have equal sizes",
      "Ward's method guarantees finding the globally optimal k-means clustering",
    ],
    answer: 'A',
    explanation: 'The closed form is a squared-centroid-distance term: ΔSSE is a variance quantity with units feature². The dendrogram\'s height axis for Ward is therefore an "SSE increase" axis. Comparing Ward heights to single/complete heights (both raw distances) is unit confusion — the classic GATE trap. On the module\'s ward run the heights are non-decreasing but numerically much larger than the distance-linkage heights on the same data.',
    trapExplanations: {
      B: 'Max-pairwise is COMPLETE linkage, not Ward — Ward merges on the minimum SSE increase.',
      C: 'Ward and average linkage coincide only in special cases (equal sizes AND balanced geometries); in general the size weighting |A||B|/(|A|+|B|) differs from the pair-count weighting of UPGMA.',
      D: 'Ward is a GREEDY hierarchical heuristic — it provably minimizes the SSE increase per single merge, not the global k-means objective. k-means can find a better partition than any cut of the Ward tree.',
    },
    concepts: ['ward', 'variance', 'trap'],
    difficulty: 3,
    tags: ['trap', 'conceptual'],
  },
  {
    id: 'hc-005',
    mode: 'conceptual-mcq',
    prompt: 'A key advantage of hierarchical clustering over k-means is that it does NOT require the user to pre-specify k. What does hierarchical clustering provide instead?',
    options: [
      'A nested hierarchy of partitions (the dendrogram) — any number of clusters can be read off by cutting at any height',
      'A single optimal partition chosen automatically by the linkage criterion',
      'Cluster shapes that are guaranteed to be convex and spherical',
      'A deterministic guarantee of the global optimum over all k',
    ],
    answer: 'A',
    explanation: 'The dendrogram encodes every partition at once: each horizontal cut gives a different k, from n singletons (cut at 0) to 1 cluster (cut at max height). k-means must be re-run for every k with a new initialization, and the "right" k is chosen afterwards by heuristics (elbow/silhouette). Hierarchical clustering pays the k-choice forward into a cut-height choice — a nested (multiscale) view the flat partition of k-means lacks. This is the plan\'s "no k needed, nested structure" comparison.',
    trapExplanations: {
      B: 'No linkage criterion picks a single "best" k automatically — the tree is k-agnostic; the cut is the user\'s choice.',
      C: 'Hierarchical clustering makes NO shape guarantee (single linkage chains, complete linkage resists chains) — spherical guarantees are k-means\' assumption.',
      D: 'Agglomerative merging is greedy and local; it makes no global optimality claim for any k.',
    },
    concepts: ['hierarchical vs k-means', 'indirect'],
    difficulty: 2,
    tags: ['indirect', 'conceptual'],
  },
  {
    id: 'hc-006',
    mode: 'nat',
    prompt: 'On dataset A the single-linkage tree has merge heights [0.5, 1, 2]. What is the cophenetic distance of the pair (p0, p2)? (Enter the number.)',
    answer: 2,
    tolerance: 0.01,
    explanation: 'The cophenetic distance is the height of the FIRST merge that puts p0 and p2 in the same cluster. p0 is merged with p1 at height 1 (forming {p0,p1}); p2 is merged with p3 at 0.5 (forming {p2,p3}); the two groups join only at the final merge, height 2. So c(p0,p2) = 2 — the same for every cross-pair (p0,p3), (p1,p2), (p1,p3). Only pairs inside the first two merges have lower cophenetic values (0.5 and 1).',
    concepts: ['cophenetic distance', 'dendrogram'],
    difficulty: 2,
    tags: ['numerical', 'visual'],
  },
];
