// src/topics/logistic-regression/failures.ts
import type { FailureDemo } from '../../engine/types';

export const logisticFailureDemos: FailureDemo[] = [
  {
    id: 'logistic-fail-imbalance',
    title: 'Class imbalance: the boundary is pushed toward the minority class',
    scenario: 'class-imbalance',
    params: { nPerClass: 15, nClass1: 60, margin: 2, noise: 1, lr: 0.3, epochs: 300, init: 'zero', seed: 42 },
    narration: 'With 15 class-0 points against 60 class-1 points, the bias term learns the prior and the p = 0.5 level set shifts toward the minority (class 0) cluster: watch accClass0 fall while accClass1 stays high. The model is honestly calibrated (mean p ≈ fraction of positives) yet its geometric boundary is distorted — a linear bias term cannot hold the boundary mid-line when the classes are this lopsided.',
    whyItBreaks: 'The bias gradient ∂L/∂b = (1/n)Σ(ŷ−y) forces Σŷ ≈ Σy, so the average probability tracks the empirical prior. With the prior at 80% class 1, p = 0.5 corresponds to a log-odds of 0 — but the class-1 likelihood mass sits at +margin, so the z = 0 contour must slide toward −margin (the minority) to balance the counts. Mitigations: resample (oversample minority), reweight the loss, or move the decision threshold to the calibrated prior (p ≈ 0.8).',
  },
  {
    id: 'logistic-fail-nonlinear',
    title: 'Non-linear class structure: a linear boundary cannot carve a ring',
    scenario: 'non-linear',
    params: { nPerClass: 20, margin: 2, noise: 1, lr: 0.3, epochs: 300, init: 'random', seed: 42, nonLinear: true },
    narration: 'Class 1 is an annulus (ring) around class 0’s central blob — the decision boundary is fundamentally non-linear. Logistic regression still fits: the best straight line cuts through the ring, misclassifying roughly the ring’s left and right arcs (accuracy settles well below 1 no matter how long training runs). The probability heat shows a linear gradient across the ring — structurally incapable of the "hole" the data needs.',
    whyItBreaks: 'The model family is restricted to affine scores z = w·x + b, whose 0.5 level set is a straight line. A ring-shaped class boundary has no linear separator (the classes are not linearly separable, and no line even approximates the topology). The CE minimizer exists but its accuracy is bounded below 1 — the failure is the MODEL FAMILY, not the optimization. Fixes: feature engineering (radial features), kernels, or non-linear models.',
  },
  {
    id: 'logistic-fail-saturation',
    title: 'Saturated sigmoid: vanishing gradients stall learning',
    scenario: 'sigmoid-saturation',
    params: { nPerClass: 20, margin: 2, noise: 1, lr: 0.3, epochs: 300, init: 'random', initScale: 10, seed: 42 },
    narration: 'Initializing with large weights (±10) pushes every log-odds deep into the sigmoid’s tails, where σ′(z) ≈ 0. The gradient ∂L/∂w = Σ(ŷ−y)x is tiny (the residual ŷ−y ≈ 0 for saturated-but-wrong predictions only slowly changes), so the loss crawls for dozens of epochs — the run looks frozen even though lr is healthy. Small init (initScale 0.1) reaches the same optimum in a fraction of the epochs.',
    whyItBreaks: 'For a point deep in the wrong tail (y = 1, z = −15), ŷ = σ(−15) ≈ 3×10⁻⁷ and the CE gradient contribution (ŷ−y) ≈ −1 is actually LARGE — the deep-wrong points are fine. The stall comes from the MIXED regime: points near z = 0 have ŷ−y ≠ 0 but σ′ ≈ 1/4, while most points sit mid-tail where the update per epoch is small and the boundary drifts only slowly. Small, standardized init keeps z in the informative band (−3, 3) where σ′ is substantial. This is the same mechanism as vanishing gradients in deep nets.',
  },
];
