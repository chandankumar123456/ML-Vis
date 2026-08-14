// src/topics/pca/comparisons.ts
import type { Comparison } from '../../engine/types';

export const pcaComparisons: Comparison[] = [
  {
    id: 'pca-vs-lda',
    title: 'PCA vs LDA (unsupervised variance vs supervised class separation)',
    topics: ['pca', 'lda'],
    axes: [
      {
        axis: 'Uses labels?',
        entries: [
          { topic: 'pca', value: 'No — maximizes uᵀΣu on the pooled covariance; a single unlabeled cloud (the eigenviewer\'s plain blue points)' },
          { topic: 'lda', value: 'Yes — maximizes the Fisher ratio wᵀS_Bw/wᵀS_Ww built from class means and per-class covariances' },
        ],
      },
      {
        axis: 'Objective',
        entries: [
          { topic: 'pca', value: 'Maximize total projected variance: PC1 = the direction of most spread (λ₁ = 1.9380 on the default run)' },
          { topic: 'lda', value: 'Maximize class separation: projected means far apart while classes stay tight' },
        ],
      },
      {
        axis: 'Max # directions',
        entries: [
          { topic: 'pca', value: 'd — every feature gets a principal axis (2 axes in the 2-D demo)' },
          { topic: 'lda', value: 'C − 1 — with 2 classes exactly ONE axis' },
        ],
      },
      {
        axis: 'Typical use',
        entries: [
          { topic: 'pca', value: 'Unsupervised reduction/denoising — reconstruct, compress, visualize; a classifier is added AFTER' },
          { topic: 'lda', value: 'Supervised reduction that doubles as a linear classifier on the projection' },
        ],
      },
    ],
    notes: [
      'THE frequently-confused pair in GATE: same-looking "project data onto a line" machinery, opposite objectives. PCA follows the data\'s long axis (variance), LDA cuts across it (class separation) — on labeled data the two directions visibly differ.',
      'PCA never sees yᵢ; the moment labels shape the objective it is no longer PCA — that is the plan\'s "PCA is not supervised" mistake.',
    ],
  },
  {
    id: 'pca-vs-feature-selection',
    title: 'PCA vs Feature Selection (extraction vs selection)',
    topics: ['pca', 'feature-selection'],
    axes: [
      {
        axis: 'What is kept',
        entries: [
          { topic: 'pca', value: 'EXTRACTION: new synthetic features z_k = v_k·(x − μ) — linear combinations of ALL original features (PC1 = 0.235·x + 0.972·y on the default run)' },
          { topic: 'feature-selection', value: 'SELECTION: a subset of the ORIGINAL features, unchanged (e.g. keep x, drop y)' },
        ],
      },
      {
        axis: 'Interpretability',
        entries: [
          { topic: 'pca', value: 'Loses feature semantics — a PC is a mix of every input dimension; units and meaning are blended' },
          { topic: 'feature-selection', value: 'Preserves semantics — the kept features are the original ones with their names and units' },
        ],
      },
      {
        axis: 'Variance retained',
        entries: [
          { topic: 'pca', value: 'Optimal per k: the top-k PCs keep the maximum possible variance (87.4% with k=1 on the default run)' },
          { topic: 'feature-selection', value: 'Sub-optimal in general — dropping correlated features can lose variance that a combination would have kept' },
        ],
      },
      {
        axis: 'When to prefer',
        entries: [
          { topic: 'pca', value: 'High-dimensional, correlated features; denoising/compression; when interpretability of directions is not needed' },
          { topic: 'feature-selection', value: 'When feature meaning matters (medical/credit), when you must cut cost per feature, or with sparse models (lasso)' },
        ],
      },
    ],
    notes: [
      'The practical test: does the downstream model need original-feature semantics? If yes, select; if no, PCA usually compresses better.',
      'They are complementary, not rivals — many pipelines select first (drop junk) then PCA (decorrelate the rest).',
    ],
  },
  {
    id: 'pca-vs-svd',
    title: 'PCA vs SVD (eigen-decomposition vs the numerically stable route)',
    topics: ['pca', 'pca-svd'],
    axes: [
      {
        axis: 'Computation',
        entries: [
          { topic: 'pca', value: 'Eigen-decomposition of the covariance (1/n)X_cᵀX_c — this topic uses the exact 2×2 closed form (no iteration)' },
          { topic: 'pca-svd', value: 'SVD of the centered data X_c = UΣVᵀ directly — the numerically stable route for d > 2' },
        ],
      },
      {
        axis: 'PCs / singular vectors',
        entries: [
          { topic: 'pca', value: 'PCs = eigenvectors of X_cᵀX_c; eigenvalues λ_k = the variance along PC k' },
          { topic: 'pca-svd', value: 'PCs = the RIGHT singular vectors V; λ_k = σ_k²/n (squared singular values scaled by 1/n)' },
        ],
      },
      {
        axis: 'Numerical behavior',
        entries: [
          { topic: 'pca', value: 'Forming X_cᵀX_c squares the condition number — loses precision for near-degenerate data (the λ₂ ≈ 0 warning case)' },
          { topic: 'pca-svd', value: 'Operates on X_c directly — more accurate for ill-conditioned data; the standard practice in numpy/scikit-learn' },
        ],
      },
      {
        axis: 'Results',
        entries: [
          { topic: 'pca', value: 'Identical in exact arithmetic — on the default run λ₁ = 1.9380, λ₂ = 0.2784, PC1 at 76.41°' },
          { topic: 'pca-svd', value: 'The same PCs, same explained-variance ratios, same reconstruction error — the two routes agree to machine precision' },
        ],
      },
    ],
    notes: [
      'Mathematically PCA IS the SVD of the centered data: X_cᵀX_c = V(Σ²/n)Vᵀ, so the eigen-solver and the SVD return the same directions (the pca-svd topic builds on exactly this identity).',
      'The difference is engineering: covariance squaring vs direct orthogonalization. For 2×2 this topic\'s closed form is exact either way.',
    ],
  },
];