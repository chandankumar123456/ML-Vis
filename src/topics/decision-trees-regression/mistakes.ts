// src/topics/decision-trees-regression/mistakes.ts
// Common GATE traps — the classification/regression confusion is the big one.
import type { Mistake } from '../../engine/types';

export const dtrMistakes: Mistake[] = [
  {
    id: 'dtr-mistake-classification-metrics',
    pattern: 'Using classification metrics (accuracy, entropy, Gini) to evaluate a REGRESSION tree',
    example: '\\text{score} = \\text{accuracy}(\\text{round}(\\hat{y}), y)',
    whyWrong: 'A regression tree is trained by minimizing SSE over continuous targets. Scoring it by rounded-accuracy throws away every residual\'s magnitude — a prediction of 5.2 vs true 8.1 "matches" after rounding 5≈8? No: it is a 2.9-unit miss that accuracy marks as a hit. The honest score is MSE/SSE (or MAE); Gini/entropy belong to the classification tree that predicts class fractions, not constants.',
    gateTrap: true,
    relatedConcept: 'sse-node',
  },
  {
    id: 'dtr-mistake-leaf-linear',
    pattern: 'Assuming regression-tree leaves store a linear model of the features',
    example: '\\hat{y} = w_0 + w_1 x_1 + \\dots + w_d x_d \\quad \\text{(per leaf)}',
    whyWrong: 'CART regression leaves store ONE constant — the mean of the targets in the region (the minimizer of SSE). A per-leaf linear model is a different model class (model trees / M5); with it, the tree would not be a step function and the splits would not minimize region SSE the way CART does.',
    gateTrap: true,
    relatedConcept: 'leaf-mean',
  },
  {
    id: 'dtr-mistake-greedy-optimal',
    pattern: 'Believing CART\'s greedy split selection produces the globally optimal tree',
    example: '\\Delta SSE_{best} \\text{ at each node } \\Rightarrow \\text{ globally minimal } \\text{SSE}',
    whyWrong: 'CART picks the locally best split at each node (largest ΔSSE) and never looks ahead. A split that seems slightly worse now can unlock a much better partition one level deeper, so the greedy tree is NOT the minimal-SSE tree in general. This is why pruning exists: grow deep greedily, then collapse leaves that fail to pay for themselves.',
    gateTrap: true,
    relatedConcept: 'sse-reduction',
  },
  {
    id: 'dtr-mistake-mean-vs-median',
    pattern: 'Using the mean as the leaf value without considering outliers',
    example: '\\hat{y}_S = \\bar{y}_S \\text{ while } y \\text{ contains an extreme value}',
    whyWrong: 'The mean minimizes SSE but is not robust: a single extreme target drags the whole leaf constant toward it, inflating error for every other point in the region. The median (minimizer of MAE) is the robust alternative — but CART regression trees report the mean, so the failure shows up as shifted steps, not as a crash.',
    gateTrap: false,
    relatedConcept: 'leaf-mean',
  },
  {
    id: 'dtr-mistake-deeper-always-better',
    pattern: 'Thinking deeper trees are always more accurate because train error falls with depth',
    example: '\\text{trainError}(d_2) < \\text{trainError}(d_1) \\Rightarrow d_2 \\text{ better}',
    whyWrong: 'Train error falls monotonically as the tree adds splits (each split only reduces in-sample SSE), but test error is U-shaped: after the right depth, the tree starts memorizing noise — the jagged step function overfits. The loss-curve\'s two series (train vs test) diverging is the signature; depth should be chosen on the TEST curve (or by validation/pruning).',
    gateTrap: true,
    relatedConcept: 'train-test-error',
  },
];