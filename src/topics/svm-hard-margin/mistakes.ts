// src/topics/svm-hard-margin/mistakes.ts
import type { Mistake } from '../../engine/types';

export const svmMistakes: Mistake[] = [
  {
    id: 'svm-functional-vs-geometric',
    pattern: 'Confusing the functional margin yᵢ(w·xᵢ+b) with the geometric margin yᵢ(w·xᵢ+b)/‖w‖',
    example: '\\text{functional } \\hat{\\gamma}_i = y_i (w \\cdot x_i + b) \\;\\neq\\; \\gamma_i = \\frac{y_i (w \\cdot x_i + b)}{\\|w\\|}',
    whyWrong: 'The functional margin is measured in arbitrary SCORE units — scale w by 2 and every functional margin doubles, while the actual geometry is unchanged. The geometric margin divides by ‖w‖ and is the true perpendicular distance (0.638 for the support vectors in the default run). Mixing the two makes the margin appear to depend on the arbitrary scale of w; the canonical constraint yᵢ(w·xᵢ+b) ≥ 1 exists precisely to fix that scale.',
    gateTrap: true,
    relatedConcept: 'svm-geometric-margin',
  },
  {
    id: 'svm-all-points-matter',
    pattern: 'Thinking every training point contributes to the max-margin boundary ("SVM uses all the data")',
    example: 'w = \\sum_{i=1}^{n} \\alpha_i y_i x_i \\quad \\text{but} \\quad \\alpha_i = 0 \\;\\text{for}\\; y_i (w \\cdot x_i + b) > 1',
    whyWrong: 'KKT complementary slackness forces αᵢ = 0 for every point strictly inside its region. On the default seed 22 of 24 points have αᵢ = 0 — the boundary is built from exactly the 2 support vectors (d9, d21). Unlike logistic regression (where every point contributes a gradient term) the hard-margin SVM keeps only the margin-touching points; interior points can be moved around without changing the separator at all.',
    gateTrap: true,
    relatedConcept: 'svm-kkt',
  },
  {
    id: 'svm-forgetting-b',
    pattern: 'Dropping the bias term b when writing or solving the hyperplane (using w·x = 0 instead of w·x + b = 0)',
    example: 'w \\cdot x = 0 \\quad \\text{(wrong — forces the boundary through the origin)}',
    whyWrong: 'Without b every boundary is forced through the origin, which is almost never the max-margin separator for real data. In the default run b = −0.230 shifts the boundary so it bisects the support-vector segment; setting b = 0 would tilt the whole geometry and collapse the margin. The bias also produces the dual condition Σαᵢyᵢ = 0 — it is not optional algebra.',
    gateTrap: true,
    relatedConcept: 'svm-hyperplane',
  },
  {
    id: 'svm-label-sign',
    pattern: 'Writing the canonical constraint with the wrong label convention — e.g. yᵢ(w·xᵢ+b) ≥ 1 with yᵢ ∈ {0, 1}',
    example: 'y_i \\in \\{0, 1\\} \\;\\Rightarrow\\; \\text{class 0 contributes } 0 \\cdot (w \\cdot x_i + b) \\ge 1 \\;\\text{— impossible}',
    whyWrong: 'The SVM constraint only makes sense with ±1 labels: yᵢ = −1 for class 0 makes yᵢ(w·xᵢ+b) ≥ 1 mean w·xᵢ+b ≤ −1 (the class-0 side), and yᵢ = +1 for class 1 means w·xᵢ+b ≥ 1. With 0/1 labels the class-0 constraint would read 0 ≥ 1 — nonsense. This topic maps class 0 → −1 and class 1 → +1 throughout the solver.',
    gateTrap: true,
    relatedConcept: 'svm-functional-margin',
  },
  {
    id: 'svm-arbitrary-scale',
    pattern: 'Thinking you can inflate the margin by scaling w (e.g. doubling w to double the "margin")',
    example: 'w \\to 2w \\;\\Rightarrow\\; \\frac{2}{\\|2w\\|} = \\frac{1}{\\|w\\|} \\;\\text{— the margin HALVES, it never grows}',
    whyWrong: 'The margin 2/‖w‖ is invariant to rescaling: if the constraint was yᵢ(w·xᵢ+b) ≥ 1, doubling w doubles every functional margin so the "canonical" scale moves with it, and the geometric band is unchanged. The correct reading: the scale of w is NOT free — the canonical constraints pin it down, and the genuine dial for the margin is the DATA (separation, outliers), not w. Measured: scaling the data by 2 keeps the boundary but halves ‖w‖ and doubles the margin, preserving margin·‖w‖ = 2.',
    gateTrap: true,
    relatedConcept: 'svm-margin',
  },
  {
    id: 'svm-perceptron-confusion',
    pattern: 'Believing any separating hyperplane is as good as the SVM\'s (perceptron mindset: "it separates, so it is correct")',
    example: '\\text{perceptron: } \\text{ANY separator with train error 0} \\;\\neq\\; \\text{SVM: the max-margin separator}',
    whyWrong: 'The perceptron converges to SOME separating hyperplane — which one depends on initialization and update order, and it can sit arbitrarily close to a support vector. The SVM is the unique separator maximizing the margin: on the default seed the max-margin line (margin 1.276) is the one that stays most robust to perturbations. Same training error, very different robustness — the margin is the tie-breaker the perceptron ignores.',
    gateTrap: false,
    relatedConcept: 'svm-primal',
  },
];
