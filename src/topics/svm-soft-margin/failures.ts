// src/topics/svm-soft-margin/failures.ts
import type { FailureDemo } from '../../engine/types';

export const svmSoftFailureDemos: FailureDemo[] = [
  {
    id: 'svm-soft-fail-c-huge-outlier',
    title: 'C → ∞ with an outlier: the boundary chases the noise',
    scenario: 'outlier-c-huge',
    params: { C: 1000, nPerClass: 10, margin: 1.5, spread: 0.5, outlier: true, outlierStrength: 3, seed: 42 },
    narration: 'With the outlier toggle ON, a single class-0 point is pushed deep into class-1 territory. At C = 1000 every unit of slack is ruinously expensive, so the solver rotates and tightens the boundary to shrink that one point\'s slack — the margin band collapses (the margin metric drops far below its outlier-free value) and the boundary starts bending toward the noise. The model generalizes poorly: it is memorizing one mislabelled point instead of the two clean clusters.',
    whyItBreaks: 'The objective C·Σξᵢ makes a single outlier dominate: it can consume the entire budget that would otherwise be spent keeping the margin wide. The optimal C balances the margin term ½‖w‖² against the slack term; with C too large the balance breaks and the fitted hyperplane chases outliers (high variance, overfitting). This is the textbook "large C overfits" failure.',
  },
  {
    id: 'svm-soft-fail-c-tiny',
    title: 'C → 0: slack is free, so the model gives up on classifying',
    scenario: 'c-tiny',
    params: { C: 0.01, nPerClass: 10, margin: 1.5, spread: 0.5, outlier: false, seed: 42 },
    narration: 'At C = 0.01 violations are nearly free, so the solver inflates the margin band enormously (the margin metric grows far beyond the hard-margin value — the band is wider than the data). Every point ends up inside the band paying a tiny slack, and although the boundary still roughly separates the clusters, the model has effectively stopped committing to confident predictions: it is the underfitting end of the C axis.',
    whyItBreaks: 'The ½‖w‖² term shrinks ‖w‖ toward 0 (margin → ∞) once C·Σξᵢ stops mattering. A huge margin with slack on every point means no training point is classified with confidence — the learned boundary is an ill-conditioned, high-bias approximation of the true one.',
  },
  {
    id: 'svm-soft-fail-label-noise',
    title: 'Label noise: a few flipped labels can dominate an otherwise clean dataset',
    scenario: 'label-noise',
    params: { C: 30, nPerClass: 10, margin: 1.5, spread: 0.5, outlier: true, outlierStrength: 3, seed: 42 },
    narration: 'The outlier toggle simulates one flipped label (a class-0 point sitting inside the class-1 cluster). At a moderate C = 30 the boundary is visibly distorted compared with the outlier-free fit, yet the solver still pays slack on the outlier rather than misclassifying a whole cluster. A handful of flipped labels — far fewer than the 20 clean points — are enough to move the decision boundary noticeably, which is why robustness to label noise needs loss clipping or even lower C.',
    whyItBreaks: 'Hinge loss is linear in the violation, so a single label flip contributes up to C·(distance) to the objective — with many clean points only pulling through ½‖w‖², a few flipped labels can outweigh the entire margin term unless C is kept small or the loss is made bounded (e.g. truncated hinge).',
  },
];
