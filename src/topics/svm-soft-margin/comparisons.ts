// src/topics/svm-soft-margin/comparisons.ts
import type { Comparison } from '../../engine/types';

export const svmSoftComparisons: Comparison[] = [
  {
    id: 'soft-vs-hard-margin',
    title: 'Soft Margin vs Hard Margin (the C → ∞ limit)',
    topics: ['svm-soft-margin', 'svm-hard-margin'],
    axes: [
      {
        axis: 'Constraint',
        entries: [
          { topic: 'svm-soft-margin', value: 'y·f ≥ 1 − ξᵢ with ξᵢ ≥ 0 — violations allowed at cost C·Σξᵢ' },
          { topic: 'svm-hard-margin', value: 'y·f ≥ 1 strictly — no point may enter the band' },
        ],
      },
      {
        axis: 'Non-separable data',
        entries: [
          { topic: 'svm-soft-margin', value: 'Feasible for ANY finite dataset (with slack) — this is its defining advantage' },
          { topic: 'svm-hard-margin', value: 'Infeasible — no solution exists when the classes overlap' },
        ],
      },
      {
        axis: 'Objective',
        entries: [
          { topic: 'svm-soft-margin', value: 'min ½‖w‖² + C·Σξᵢ — slack term added' },
          { topic: 'svm-hard-margin', value: 'min ½‖w‖² — margin alone' },
        ],
      },
      {
        axis: 'Limit behaviour',
        entries: [
          { topic: 'svm-soft-margin', value: 'C → ∞ recovers the hard-margin solution on separable data (verified in the tests: margin within ~0.2%)' },
          { topic: 'svm-hard-margin', value: 'The reference point the soft margin approaches; on the default separable dataset margin ≈ 2' },
        ],
      },
      {
        axis: 'Support vectors',
        entries: [
          { topic: 'svm-soft-margin', value: 'Free SVs on the band (ξ = 0) PLUS bounded SVs inside/wrong side (ξ > 0, αᵢ = C)' },
          { topic: 'svm-hard-margin', value: 'Only points exactly on the band edges f = ±1' },
        ],
      },
    ],
    notes: [
      'Scrub the soft-margin run to large C: the margin band shrinks toward the hard-margin band and the slack lines vanish on separable data.',
      'GATE: the soft margin is what makes SVM usable on real (noisy, overlapping) data; the hard margin is the clean-ideal limit.',
    ],
  },
  {
    id: 'hinge-vs-logistic',
    title: 'Hinge Loss vs Logistic Loss (SVM vs Logistic Regression)',
    topics: ['svm-soft-margin', 'logistic-regression'],
    axes: [
      {
        axis: 'Loss at y·f',
        entries: [
          { topic: 'svm-soft-margin', value: 'max(0, 1 − y·f) — EXACTLY 0 once y·f ≥ 1 (flat region, zero gradient)' },
          { topic: 'logistic-regression', value: 'ln(1 + e^(−y·f)) — never exactly 0, nonzero gradient for every point' },
        ],
      },
      {
        axis: 'Which points matter',
        entries: [
          { topic: 'svm-soft-margin', value: 'Only the support vectors near the band — far correct points are ignored' },
          { topic: 'logistic-regression', value: 'All points pull on the boundary, though confident ones only weakly' },
        ],
      },
      {
        axis: 'Shape near 0',
        entries: [
          { topic: 'svm-soft-margin', value: 'Linear in y·f — subgradient is a step (not smooth at y·f = 1)' },
          { topic: 'logistic-regression', value: 'Smooth everywhere — gradient magnitude decays exponentially' },
        ],
      },
      {
        axis: 'Probabilistic output',
        entries: [
          { topic: 'svm-soft-margin', value: 'No natural probability — scores need post-hoc calibration (Platt scaling)' },
          { topic: 'logistic-regression', value: 'Native probabilities via the sigmoid — calibrated by construction' },
        ],
      },
    ],
    notes: [
      'Both are convex surrogates for the 0-1 loss; the hinge is the tightest linear upper bound, the logistic is the smooth surrogate.',
      'The flat hinge region is exactly why SVMs are sparse in the training set and logistic regression is not.',
    ],
  },
  {
    id: 'svm-c-vs-ridge-lambda',
    title: 'C vs Ridge λ (the Inverse-Regularization Duality)',
    topics: ['svm-soft-margin', 'ridge-regression'],
    axes: [
      {
        axis: 'Objective shape',
        entries: [
          { topic: 'svm-soft-margin', value: 'Σξᵢ + (1/C)·½‖w‖² — hinge slack + margin term' },
          { topic: 'ridge-regression', value: '(1/n)·‖y − Xθ‖² + λ·‖θ‖² — squared loss + penalty' },
        ],
      },
      {
        axis: 'Regularization dial',
        entries: [
          { topic: 'svm-soft-margin', value: 'C ↑ ⇒ LESS regularization (1/C ↓) — smaller margin, more overfit' },
          { topic: 'ridge-regression', value: 'λ ↑ ⇒ MORE regularization — stronger shrinkage, more underfit' },
        ],
      },
      {
        axis: 'End points',
        entries: [
          { topic: 'svm-soft-margin', value: 'C → ∞ hard margin (no slack); C → 0 degenerate w = 0' },
          { topic: 'ridge-regression', value: 'λ → 0 plain OLS; λ → ∞ coefficients → 0' },
        ],
      },
      {
        axis: 'What gets penalized',
        entries: [
          { topic: 'svm-soft-margin', value: 'Margin (inverse ‖w‖) trades against CLASSIFICATION slack' },
          { topic: 'ridge-regression', value: 'Coefficient magnitude trades against SQUARED regression error' },
        ],
      },
    ],
    notes: [
      'The direction-duality C ↔ 1/λ is the cleanest GATE summary: both are "loss + penalty" with the dial on the penalty, but SVM dials the loss weight while ridge dials the penalty weight.',
      'Never conflate the two directions: C ↑ = variance ↑ (SVM), λ ↑ = variance ↓ (ridge).',
    ],
  },
];
