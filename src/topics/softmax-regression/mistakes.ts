// src/topics/softmax-regression/mistakes.ts
import type { Mistake } from '../../engine/types';

export const softmaxMistakes: Mistake[] = [
  {
    id: 'softmax-no-maxshift',
    pattern: 'Applying softmax without the max-shift (log-sum-exp stability)',
    example: '\\hat{y}_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}} \\;\\text{(naive)} \\quad\\Rightarrow\\quad e^{1000} = \\infty \\Rightarrow \\text{NaN}',
    whyWrong: 'For logits like [1000, 1001, 1002] the naive exponent overflows to Infinity and every probability becomes NaN. The max-shift form e^{z_k−m}/Σe^{z_j−m} is algebraically identical (shift invariance) but keeps every exponent ≤ 0, so it is always finite.',
    gateTrap: true,
    relatedConcept: 'softmax-stable',
  },
  {
    id: 'sigmoid-vs-softmax',
    pattern: 'Confusing sigmoid with softmax (or using K independent sigmoids for K classes)',
    example: '\\hat{y}_1 = \\sigma(z_1), \\hat{y}_2 = \\sigma(z_2) \\;\\text{(wrong for multiclass)} \\quad\\text{vs}\\quad \\hat{y}_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}',
    whyWrong: 'Two independent sigmoids need not sum to 1 — they are not a probability distribution. The 2-class softmax is σ(z₁ − z₀), a sigmoid of the DIFFERENCE, which automatically enforces ŷ₁ + ŷ₀ = 1. For K classes use one softmax over all K logits.',
    gateTrap: true,
    relatedConcept: 'softmax-sigmoid',
  },
  {
    id: 'gradient-missing-class-sum',
    pattern: 'Forgetting the sum over classes (or the indicator term) in the softmax gradient',
    example: '\\frac{\\partial L}{\\partial w_k} = \\sum_i \\hat{y}_{ik} x_i \\;\\text{(wrong — no indicator)} \\quad\\text{vs}\\quad \\sum_i (\\hat{y}_{ik} - 1\\{y_i = k\\}) x_i',
    whyWrong: 'Each class-k gradient needs the SOFTMAX ERROR ŷ_{ik} − 1{y_i=k}: the indicator subtracts the one-hot truth so that non-true classes are pushed away while the true class is pulled in. Dropping it makes every class gradient positive — the model never learns to specialize.',
    gateTrap: true,
    relatedConcept: 'softmax-gradient',
  },
  {
    id: 'softmax-as-confidence',
    pattern: 'Treating softmax probabilities as calibrated confidences (they are not)',
    whyWrong: 'Softmax outputs are guaranteed to be positive and sum to 1, but they are NOT calibrated probabilities of correctness — a confidently wrong model can assign 0.99 to the wrong class. Calibration needs temperature scaling or Platt scaling; argmax accuracy is what CE directly optimizes.',
    gateTrap: false,
    relatedConcept: 'softmax-def',
  },
  {
    id: 'ce-log-zero',
    pattern: 'Calling log(0) during CE on a confident wrong prediction',
    example: '\\log 0 = -\\infty \\;\\text{(in CE when } \\hat{y} \\text{ hits exactly 0)}',
    whyWrong: 'A softmax prediction is mathematically always > 0, but float rounding can produce exactly 0. The fix is clipping predictions to [1e-12, 1] before taking log — used throughout this topic\'s CE metric — so a confident mistake costs about 27.6 nats instead of crashing the run.',
    gateTrap: false,
    relatedConcept: 'categorical-ce',
  },
];
