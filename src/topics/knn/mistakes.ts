// src/topics/knn/mistakes.ts
import type { Mistake } from '../../engine/types';

export const knnMistakes: Mistake[] = [
  {
    id: 'knn-no-scaling',
    pattern: 'Running k-NN on raw features with wildly different scales (e.g. height in cm vs income in lakh)',
    example: 'd = \\sqrt{(x_1 - q_1)^2 + (x_2 - q_2)^2} \\;\\Rightarrow\\; \\text{large-scale feature dominates}',
    whyWrong: 'Euclidean distance sums squared differences, so the feature with the bigger numeric range contributes the bulk of every distance — the classifier effectively ignores the others. The fix is feature standardization (z-scoring) before computing distances; without it the "nearest" neighbors are nearest only in the dominant feature.',
    gateTrap: true,
    relatedConcept: 'knn-euclidean',
  },
  {
    id: 'knn-odd-k-myth',
    pattern: 'Believing odd k always avoids ties (true only for exactly 2 classes), and ignoring tie-break rules for 3+ classes',
    example: '\\text{3 classes, } k = 5: \\; (2, 2, 1) \\; \\text{is a tie} \\; \\Rightarrow \\; \\text{odd k does NOT save you}',
    whyWrong: 'With two classes odd k guarantees a strict majority, but with three or more classes a tie (e.g. 2–2–1 at k=5) is possible at ANY k. A deterministic tie-break (nearest-of-tied, then lower class index) must be defined — and even-k ties in the 2-class case need one too.',
    gateTrap: true,
    relatedConcept: 'knn-majority-vote',
  },
  {
    id: 'knn-training-phase',
    pattern: 'Thinking k-NN has a training phase that learns parameters from the data',
    example: '\\text{lazy learner: store } (x_i, y_i) \\text{ — "training" is O(1); every query pays } O(n \\cdot d)',
    whyWrong: 'k-NN is a lazy (instance-based) learner: it stores the training set and does no fitting. The O(n·d) cost is paid at PREDICTION time (scanning all points), not training. Confusing this reverses the complexity story — training is trivial, inference is expensive.',
    gateTrap: true,
    relatedConcept: 'knn-complexity',
  },
  {
    id: 'knn-k1-best',
    pattern: 'Choosing k by minimizing TRAIN error (which always picks k=1)',
    example: 'E_{\\text{train}}(k{=}1) = 0 \\;\\text{(each point is its own neighbor)}\\; \\Rightarrow \\; \\text{k=1 "wins"}',
    whyWrong: 'k=1 scores 0 train error by construction (self-neighbor), so train error is useless for selecting k — it always says "k=1". The honest curve is validation/leave-one-out error: it starts high (≈0.42 at k=1 on the default seed) and drops quickly as k grows, bottoming out around 0.21 at intermediate k. On a small dataset the LOO estimate is noisy — it oscillates between ≈0.21 and ≈0.38 on this seed rather than descending monotonically — and once k passes half the dataset the vote saturates toward the class majority: past that point increasing k stops improving the honest error (on the default seed it even creeps back up at k=19–20). Train error cannot select k; LOO at least identifies the working range.',
    gateTrap: true,
    relatedConcept: 'knn-loo-error',
  },
  {
    id: 'knn-all-far',
    pattern: 'Thinking that in high dimensions "some point is always close"',
    whyWrong: 'The curse of dimensionality does the opposite: volume concentrates near the surface, pairwise distances concentrate, and the ratio of nearest-to-farthest distance approaches 1. Nearest neighbors in 100 dimensions are barely closer than random points — distance stops discriminating.',
    gateTrap: false,
    relatedConcept: 'knn-curse',
  },
];
