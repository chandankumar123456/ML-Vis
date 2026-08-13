// src/topics/svm-soft-margin/failures.ts
import type { FailureDemo } from '../../engine/types';

export const svmSoftFailureDemos: FailureDemo[] = [
  {
    id: 'svm-soft-fail-c-huge-outlier',
    title: 'C → ∞ with an outlier: the box constraint αᵢ ≤ C caps a single point\'s influence',
    scenario: 'outlier-c-huge',
    params: { C: 1000, nPerClass: 10, margin: 1.5, spread: 0.5, outlier: true, outlierStrength: 3, seed: 42 },
    narration: 'The outlier toggle simulates one mislabelled point: a single class-0 point is pushed deep into class-1 territory (x = margin + 3·spread). At C = 1000 the objective is dominated by the slack term — C·Σξ = 3685.07 of the total 3685.48 (99.99%) — yet even at C = 1000 the boundary does NOT chase the outlier: rotating it to shrink that one point\'s slack creates more slack elsewhere, so the optimum stays at the max-margin line and pays the outlier\'s 3.685 slack. The box constraint αᵢ ≤ C caps single-point influence: the fitted line is identical to the outlier-free fit (margin 2.2119 vs 2.2118), so the model is not memorizing the mislabelled point — it is simply paying it.',
    whyItBreaks: 'The objective C·Σξᵢ makes a single outlier dominate the cost: at C = 1000 the outlier\'s slack is 99.99% of the objective. Yet the boundary still does not chase it, because rotating to shrink one point\'s slack creates more slack elsewhere — the optimum stays at the max-margin line and pays the slack. The textbook "large C overfits" failure only appears when the mislabelled point sits close to the boundary (where absorbing it costs little else) or when several labels flip; the box constraint αᵢ ≤ C is the guarantee that no single point can contribute more than C·‖xᵢ‖ to w.',
  },
  {
    id: 'svm-soft-fail-c-tiny',
    title: 'C → 0: slack is free, so the model gives up on classifying',
    scenario: 'c-tiny',
    params: { C: 0.01, nPerClass: 10, margin: 1.5, spread: 0.5, outlier: false, seed: 42 },
    narration: 'At C = 0.01 violations are nearly free, so the solver inflates the margin band enormously: margin = 6.80 vs the hard-margin 2.21 (≈ 3.1×) — the band is wider than the data span. 19 of 20 points pay slack; with slack this cheap the solver even tolerates 5 misclassified points, so the boundary no longer separates the clusters cleanly — the underfitting end of the C axis.',
    whyItBreaks: 'The ½‖w‖² term shrinks ‖w‖ toward 0 (margin → ∞) once C·Σξᵢ stops mattering. A huge margin with slack on nearly every point (19 of 20 here) means almost no training point is classified with confidence — the learned boundary is an ill-conditioned, high-bias approximation of the true one.',
  },
  {
    id: 'svm-soft-fail-label-noise',
    title: 'Label noise: a few flipped labels can dominate an otherwise clean dataset',
    scenario: 'label-noise',
    params: { C: 1, nPerClass: 10, margin: 1.5, spread: 0.5, outlier: true, outlierStrength: 3, seed: 42 },
    narration: 'The outlier toggle simulates one flipped label (a class-0 point sitting inside the class-1 cluster). At C = 1 the boundary is visibly distorted compared with the outlier-free fit — ‖Δw‖ ≈ 0.14 and the margin widens from 2.21 to 2.33 — yet the solver still pays slack on the outlier (Σξ = 3.70) rather than misclassifying a whole cluster: exactly 1 of 20 points ends up on the wrong side. A single flipped label among 20 clean points is enough to move the decision boundary noticeably, which is why robustness to label noise needs loss clipping or even lower C.',
    whyItBreaks: 'Hinge loss is linear in the violation, so a single label flip contributes up to C·(distance) to the objective — with many clean points only pulling through ½‖w‖², a few flipped labels can outweigh the entire margin term unless C is kept small or the loss is made bounded (e.g. truncated hinge).',
  },
];
