// src/topics/lda/comparisons.ts
import type { Comparison } from '../../engine/types';

export const ldaComparisons: Comparison[] = [
  {
    id: 'lda-vs-pca',
    title: 'LDA vs PCA (supervised vs unsupervised dimensionality reduction)',
    topics: ['lda', 'pca'],
    axes: [
      {
        axis: 'Uses labels?',
        entries: [
          { topic: 'lda', value: 'Yes — maximizes wᵀS_Bw/wᵀS_Ww from class means and per-class covariances (supervised)' },
          { topic: 'pca', value: 'No — maximizes wᵀΣw on the pooled covariance only (unsupervised; labels ignored)' },
        ],
      },
      {
        axis: 'Objective',
        entries: [
          { topic: 'lda', value: 'Maximize class separation: make projected means far apart while classes stay tight' },
          { topic: 'pca', value: 'Maximize total variance: keep directions with the most spread, regardless of classes' },
        ],
      },
      {
        axis: 'Max # directions',
        entries: [
          { topic: 'lda', value: 'C − 1 (with 2 classes: exactly ONE axis)' },
          { topic: 'pca', value: 'd (all features get a principal axis)' },
        ],
      },
      {
        axis: 'Resulting boundary / use',
        entries: [
          { topic: 'lda', value: 'A linear classifier on the projection (threshold on z = ŵᵀx) — classification + reduction in one model' },
          { topic: 'pca', value: 'A reconstructed low-dim subspace — denoising/reduction only; a classifier must be added after' },
        ],
      },
    ],
    notes: [
      'THE frequently-confused pair in GATE: if labels exist and separation is the goal, LDA; if you are just summarizing/denoising unlabeled structure, PCA.',
      'In the simulation, tilted shared covariance makes the two axes visibly differ: PCA follows the ellipse\'s long axis, LDA cuts across it after whitening — the sweep curve is the LDA objective, the variance profile would be PCA\'s.',
      'Both are linear projections of the same data, so it is tempting to conflate them — but LDA\'s rank-1 S_B (2 classes) versus PCA\'s d eigenvectors is an immediate structural tell.',
    ],
  },
  {
    id: 'lda-vs-logistic',
    title: 'LDA vs Logistic Regression (generative vs discriminative)',
    topics: ['lda', 'logistic-regression'],
    axes: [
      {
        axis: 'Model family',
        entries: [
          { topic: 'lda', value: 'GENERATIVE: models P(x|C) with Gaussians (shared covariance), derives P(C|x) via Bayes' },
          { topic: 'logistic-regression', value: 'DISCRIMINATIVE: models P(C|x) directly with a sigmoid — no distribution assumed for P(x)' },
        ],
      },
      {
        axis: 'Parameters estimated',
        entries: [
          { topic: 'lda', value: 'Class means μ_c, shared covariance Σ (via S_W), threshold τ — from the density fits' },
          { topic: 'logistic-regression', value: 'w and b — by maximizing conditional likelihood (gradient descent on cross-entropy)' },
        ],
      },
      {
        axis: 'Robustness to non-Gaussian data',
        entries: [
          { topic: 'lda', value: 'Brittle — wrong density assumption degrades the boundary (multimodal/heteroscedastic failures)' },
          { topic: 'logistic-regression', value: 'Robust — as a discriminative model it only needs the linear log-odds to be reasonable' },
        ],
      },
      {
        axis: 'Efficiency with few samples',
        entries: [
          { topic: 'lda', value: 'Closed form (analytic) — one matrix inverse, no iteration' },
          { topic: 'logistic-regression', value: 'Iterative optimizer (GD/Newton); needs epochs but no covariance inversion' },
        ],
      },
    ],
    notes: [
      'Both produce a LINEAR boundary in 2-D — in the simulation they draw near-identical lines on well-behaved Gaussian data, which is the GATE "when does it matter" crux.',
      'When the Gaussian/shared-Σ assumption holds, LDA is typically more sample-efficient (analytic, lower variance); when it fails, logistic is the safer bet (no density assumption to violate).',
      'The shared covariance IS the LDA Gaussian assumption — that is the axis the plan\'s "generative vs discriminative" framing rides on.',
    ],
  },
  {
    id: 'lda-vs-svm',
    title: 'LDA vs SVM (density assumption vs margin geometry)',
    topics: ['lda', 'svm-hard-margin'],
    axes: [
      {
        axis: 'Optimization target',
        entries: [
          { topic: 'lda', value: 'Fisher ratio J(w) = wᵀS_Bw/wᵀS_Ww — built from class means and pooled covariance' },
          { topic: 'svm-hard-margin', value: 'Margin: min ½‖w‖² s.t. yᵢ(w·xᵢ+b) ≥ 1 — built from the closest boundary points only' },
        ],
      },
      {
        axis: 'Data used for the boundary',
        entries: [
          { topic: 'lda', value: 'ALL points, through μ_c and S_W (mean + covariance statistics)' },
          { topic: 'svm-hard-margin', value: 'ONLY support vectors (points on the margin); the rest are discarded' },
        ],
      },
      {
        axis: 'Assumptions',
        entries: [
          { topic: 'lda', value: 'Gaussian classes with shared covariance — a generative assumption that can be wrong' },
          { topic: 'svm-hard-margin', value: 'Margin/separability geometry — distribution-free; needs no density model' },
        ],
      },
      {
        axis: 'Class imbalance / outliers',
        entries: [
          { topic: 'lda', value: 'Means and covariances are sensitive — one outlier tilts S_W (failure demo); a majority class distorts μ' },
          { topic: 'svm-hard-margin', value: 'Boundary is set by a few SVs — robust to bulk imbalance; but a single mislabeled point can hijack a hard margin (soft margin fixes it)' },
        ],
      },
    ],
    notes: [
      'Both are linear classifiers, so on the simulation\'s well-separated Gaussians their boundaries coincide in spirit — the differences show in the failure demos: LDA breaks on outliers/singular S_W, SVM on non-separability.',
      'LDA is parametric with a closed form (cheap, sample-efficient when its model is right); SVM is a convex margin optimizer (robust to density assumptions, but slower and margin-centric).',
      'A nice GATE contrast: LDA asks "how do the densities overlap?", SVM asks "how far apart are the nearest opposing points?".',
    ],
  },
];