// src/topics/hierarchical-clustering/mistakes.ts
// Measured anchors cited below:
//   dataset B: single final height 1.118 (chains p1 onto the tight pair),
//     complete final height 2.220 (merges the far pair first).
//   dataset A: heights [0.5, 1, 2] — the second merge (p0,p1)@1 is NOT at
//     height 0.5 or 2; misreading the axis is measurable on this tree.
import type { Mistake } from '../../engine/types';

export const hierarchicalMistakes: Mistake[] = [
  {
    id: 'hc-single-vs-complete',
    pattern: 'Confusing single and complete linkage — assuming they give the same tree, or that "single" just means "fewer merges"',
    example: 'd_{\\text{single}}(A,B) = \\min_{a,b} d(a,b) \\;\\neq\\; d_{\\text{complete}}(A,B) = \\max_{a,b} d(a,b)',
    whyWrong: 'min vs max is a structural difference, not a scale difference. On dataset B the two criteria pick a DIFFERENT second merge: single linkage chains p1 onto the tight pair (m2 = (p1, {p2,p3}) at 1.044) because p1 is only 1.044 from p2, while complete linkage merges (p0, p1) at 1.118 because p1\'s FARTHEST distance to the right cluster (1.217) exceeds it. The final heights diverge to 1.118 (single) vs 2.220 (complete). Assuming they agree on real data is the plan\'s #1 linkage mistake.',
    gateTrap: true,
    relatedConcept: 'hc-linkage-single',
  },
  {
    id: 'hc-dendrogram-height',
    pattern: 'Reading the dendrogram height wrong — e.g. thinking a merge\'s height is the distance between two specific points, or reading the axis upside down',
    example: 'h(\\text{merge of } A, B) = d_{\\text{linkage}}(A, B) \\;\\neq\\; \\min_{a \\in A, b \\in B} d(a,b) \\;\\text{in general}',
    whyWrong: 'The height of a merge is the LINKAGE value of the two clusters — for single linkage that is the closest-pair distance, but for complete/average/ward it is a different aggregate. On dataset A the final merge is at height 2 even though the closest cross-pair distance is 2 (they coincide here only because min and the final merge align); on dataset B the final single-linkage merge is at 1.118, which is NOT the distance between p0 and any single point (d(p0,p3) = 2.220). Reading heights as point-to-point distances is wrong whenever the tree has more than two leaves, and Ward heights are squared units on top of that.',
    gateTrap: true,
    relatedConcept: 'hc-linkage-complete',
  },
  {
    id: 'hc-merges-irreversible',
    pattern: 'Thinking agglomerative merges are reversible — that undoing a bad early merge is possible after the tree is built',
    example: '\\text{merge}(A, B) \\Rightarrow \\text{split}(A \\cup B) \\;\\text{recovers } A, B \\;\\text{exactly} \\quad (\\text{FALSE})',
    whyWrong: 'Agglomerative clustering is GREEDY: each merge is final, and later merges are decided with the merged cluster as a unit. Once {p0, p1} exists, every subsequent distance is computed against the PAIR — information about which member was responsible is lost. Splitting a dendrogram node (a cut) recovers the partition AT that height, but not the original point sets of earlier hypothetical trees: dataset A cut at 1.5 gives {{p0,p1}, {p2,p3}} — not the same as re-running with a different first merge. The correct way to explore "what if the first merge differed" is re-running with another linkage, not un-merging.',
    gateTrap: true,
    relatedConcept: 'hc-cut',
  },
  {
    id: 'hc-cophenetic-means-quality',
    pattern: 'Treating the cophenetic correlation as an accuracy score or a cluster-count selector',
    example: 'r_{\\text{cophenetic}} = 0.901 \\;\\Rightarrow\\; \\text{"the tree is 90.1\\% correct"} \\quad (\\text{FALSE})',
    whyWrong: 'The cophenetic correlation measures how faithfully the merge heights reproduce the ORDER of the pairwise distances — a validity index for the TREE, not a prediction accuracy and not a criterion for choosing a cut height. A perfectly distorted but distance-consistent tree can score high while cutting badly. On the default run r = 0.901 with a clearly two-blob structure; the cut choice comes from the height GAP (0.896 → 1.795), not from the correlation value.',
    gateTrap: true,
    relatedConcept: 'hc-cophenetic-corr',
  },
];
