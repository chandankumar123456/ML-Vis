// src/topics/softmax-regression/failures.ts
import type { FailureDemo } from '../../engine/types';

export const softmaxFailureDemos: FailureDemo[] = [
  {
    id: 'softmax-fail-imbalance',
    title: 'Class imbalance: the boundary is pushed toward the majority class',
    scenario: 'class-imbalance',
    params: { nPerClass: 30, margin: 1, learningRate: 0.1, epochs: 300, seed: 42 },
    narration: 'With overlapping clusters (margin = 1), the softmax minimizes mean CE over all points equally. A large majority class gets more gradient mass, so the model becomes over-confident in it and the boundary shifts toward the minority clusters — the minority regions shrink in the decision-boundary view and their points are misclassified more often.',
    whyItBreaks: 'Mean CE weights every point equally, so classes with few points contribute little to the gradient. The classifier is not class-balanced; the optimal CE decision boundary still favors the prior (majority) class in the overlap region. Fixes: class weights, resampling, or balanced sampling.',
  },
  {
    id: 'softmax-fail-huge-logits',
    title: 'Huge logits: exp overflows without the max-shift (log-sum-exp)',
    scenario: 'huge-logits',
    params: { nPerClass: 20, margin: 3, learningRate: 0.5, epochs: 500, seed: 42 },
    narration: 'When weights grow large (high η, many epochs), raw logits can exceed ~700, where e^{z} overflows double precision (max ≈ 1.8×10³⁰⁸). A naive softmax then returns NaN and the whole run dies. The stable max-shift softmax used here renormalizes the same probabilities with exponents ≤ 0, so the run keeps training instead of blowing up.',
    whyItBreaks: 'exp(z) for z ≳ 709 overflows the IEEE-754 double range. The max-shift form computes the IDENTICAL probabilities but never evaluates exp with a positive argument — the canonical reason log-sum-exp stability is mandatory in softmax implementations.',
  },
  {
    id: 'softmax-fail-correlated',
    title: 'Correlated classes: near-duplicate clusters make scores ambiguous',
    scenario: 'correlated-classes',
    params: { nPerClass: 20, margin: 0.75, learningRate: 0.1, epochs: 300, seed: 7 },
    narration: 'With two cluster centers very close together (margin = 0.75), their Gaussian clouds overlap so heavily that the linear softmax boundary cannot separate them: the model assigns nearly equal probability to both, accuracy plateaus well below 1, and the confusion between the pair dominates the misclassification count.',
    whyItBreaks: 'Classes are only linearly separable when their score functions can be ordered consistently; overlapping Gaussians put points with z_k ≈ z_j in the ambiguous band. Correlated (near-duplicate) classes violate the separability assumption — no linear decision boundary can fix the overlap.',
  },
];
