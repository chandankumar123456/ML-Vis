// src/topics/svm-soft-margin/mistakes.ts
import type { Mistake } from '../../engine/types';

export const svmSoftMistakes: Mistake[] = [
  {
    id: 'c-direction',
    pattern: 'Thinking a larger C always gives better generalization (treating C like a regularization strength λ)',
    example: 'C \\uparrow \\;\\Rightarrow\\; \\text{"more regularization"} \\quad\\text{WRONG — actually } C \\uparrow \\;\\Rightarrow\\; \\lambda = 1/C \\downarrow',
    whyWrong: 'C is the price of slack: larger C means FEWER violations are tolerated — the boundary hugs the training data, variance goes up. Larger C is LESS regularization, the exact opposite of a larger λ in ridge/lasso. This is the single most common GATE trap for the soft margin.',
    gateTrap: true,
    relatedConcept: 'svm-c-lambda',
  },
  {
    id: 'slack-vs-hinge',
    pattern: 'Confusing the slack ξᵢ with the hinge loss, or forgetting ξᵢ ≥ 0',
    example: '\\xi_i \\;\\text{is just "how wrong" — it can be negative} \\quad\\text{WRONG — } \\xi_i = \\max(0, 1 - y_i f(x_i)) \\ge 0',
    whyWrong: 'ξᵢ IS the hinge loss max(0, 1 − y·f) — identical by construction (the simulation reports both metrics with the same value). And ξᵢ ≥ 0 is not optional: allowing negative slack makes every constraint trivially satisfiable and the optimum collapses to w = 0.',
    gateTrap: true,
    relatedConcept: 'svm-slack-constraints',
  },
  {
    id: 'label-convention',
    pattern: 'Using 0/1 labels instead of ±1 in the hinge loss',
    example: 'y \\in \\{0, 1\\}: \\max(0, 1 - y\\,f(x)) \\quad\\text{WRONG for SVM — labels must be } \\pm 1',
    whyWrong: 'The hinge is defined for y ∈ {+1, −1}. With y = 0 a correct point gets max(0, 1 − 0) = 1 — every class-0 point is "misclassified" by construction and the loss is meaningless. (0/1 labels are fine for cross-entropy; for hinge you must map 0 → −1.)',
    gateTrap: true,
    relatedConcept: 'svm-hinge-loss',
  },
  {
    id: 'box-bound',
    pattern: 'Writing the soft-margin dual without the upper bound αᵢ ≤ C',
    example: '\\max\\; \\sum_i \\alpha_i - \\tfrac12 \\sum_{i,j} \\alpha_i\\alpha_j y_i y_j x_i\\cdot x_j,\\; \\sum_i \\alpha_i y_i = 0,\\; \\alpha_i \\ge 0 \\quad\\text{— the BOX bound } \\alpha_i \\le C \\text{ is missing}',
    whyWrong: 'Without 0 ≤ αᵢ ≤ C the dual is the HARD-margin dual — it silently forbids slack, and on non-separable data the optimum diverges. The box bound is exactly where the primal C survives into the dual (stationarity C − αᵢ − μᵢ = 0 with μᵢ ≥ 0).',
    gateTrap: true,
    relatedConcept: 'svm-dual-box',
  },
  {
    id: 'hard-margin-zero',
    pattern: 'Believing hard margin and C = 0 are the same thing',
    example: 'C = 0 \\;=\\; \\text{hard margin} \\quad\\text{WRONG — } C = 0 \\Rightarrow w = 0 \\;\\text{(every violation free)}',
    whyWrong: 'Hard margin FORBIDS violations (C → ∞); C = 0 makes them free, so the optimum is w = 0, b = 0 — a constant model that classifies nothing. The two ends of the C axis are hard margin and degenerate, not each other.',
    gateTrap: true,
    relatedConcept: 'svm-soft-objective',
  },
];
