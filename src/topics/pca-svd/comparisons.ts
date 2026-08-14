// src/topics/pca-svd/comparisons.ts
// Measured anchors cited below (n 40, corr 0.7, rotDeg 30, noise 0.15, seed 42):
//   σ₁ = 8.804484511776364, σ₂ = 3.3373516320652685, λ(XᵀX) = 77.51894751810988,
//   λ(cov) = 1.9379736879527472, V₁ = (0.2350, 0.9720) at 76.41°, ratio = 2.6382.
import type { Comparison } from '../../engine/types';

export const svdComparisons: Comparison[] = [
  {
    id: 'svd-vs-eigen',
    title: 'SVD vs Eigen-decomposition',
    topics: ['pca-svd', 'pca'],
    axes: [
      {
        axis: 'What it diagonalizes',
        entries: [
          { topic: 'pca-svd', value: 'X itself (any n × d matrix) — via its Gram matrix XᵀX = VΣ²Vᵀ' },
          { topic: 'pca', value: 'the square symmetric covariance Σ = (1/n)X_cᵀX_c' },
        ],
      },
      {
        axis: 'Square-matrix requirement',
        entries: [
          { topic: 'pca-svd', value: 'none — the SVD exists for EVERY real matrix, rectangular or not' },
          { topic: 'pca', value: 'yes — eigen-decomposition is only defined for square matrices (the covariance is square because it is d × d)' },
        ],
      },
      {
        axis: 'Output factors',
        entries: [
          { topic: 'pca-svd', value: 'U (n×n) left vectors, Σ diagonal σ_k = √λ_k, V (d×d) right vectors — two rotation frames + a stretch' },
          { topic: 'pca', value: 'V only: eigenvectors + eigenvalues λ_k (one frame, real scalars)' },
        ],
      },
      {
        axis: 'Numerical stability',
        entries: [
          { topic: 'pca-svd', value: 'condition σ₁/σ₂ — never forms the Gram matrix, so near-degeneracy is NOT squared' },
          { topic: 'pca', value: 'condition (σ₁/σ₂)² — forming XᵀX squares the condition number and loses digits' },
        ],
      },
    ],
    notes: [
      'The SVD is the generalization: eigen-decomposition of a symmetric matrix IS its SVD (U = V, σ_k = |λ_k|). For non-symmetric or rectangular matrices only the SVD exists.',
      'On this topic the two are linked exactly: the module\'s build-up step 3 shows XᵀX = VΣ²Vᵀ side-by-side with the SVD factors (same V, λ = σ²) — the relation animation the plan mandates.',
      'PCA is "eigen-decomposition of the covariance"; computing it via the SVD of the centered data is the numerically preferred route in practice.',
    ],
  },
  {
    id: 'svd-vs-pca',
    title: 'SVD vs PCA',
    topics: ['pca-svd', 'pca'],
    axes: [
      {
        axis: 'Mathematical object',
        entries: [
          { topic: 'pca-svd', value: 'factorizes the centered data matrix itself: X_c = UΣVᵀ' },
          { topic: 'pca', value: 'diagonalizes the covariance (1/n)X_cᵀX_c = V diag(λ) Vᵀ' },
        ],
      },
      {
        axis: 'Eigenvalue ↔ singular value',
        entries: [
          { topic: 'pca-svd', value: 'σ_k = √λ_k(XᵀX) = √(n·λ_k(cov)) — measured 8.8045 = √77.5189' },
          { topic: 'pca', value: 'λ_k(cov) = σ_k²/n — measured 1.9379736879527472 = 8.8045²/40' },
        ],
      },
      {
        axis: 'Directions',
        entries: [
          { topic: 'pca-svd', value: 'right singular vectors V ARE the PCs — V₁ = (0.2350, 0.9720) at 76.41° on the default run' },
          { topic: 'pca', value: 'eigenvectors of the covariance — the SAME V₁ at 76.41° (identical cloud, identical direction)' },
        ],
      },
      {
        axis: 'Scores',
        entries: [
          { topic: 'pca-svd', value: 'columns of UΣ (or U_kΣ_k) — the per-point coefficients in the principal frame' },
          { topic: 'pca', value: 'projections z_i = V·(x_i − μ) — numerically the same numbers' },
        ],
      },
      {
        axis: 'Low-rank reconstruction',
        entries: [
          { topic: 'pca-svd', value: 'X̂_k = U_kΣ_kV_kᵀ, error = σ_{k+1} (Eckart–Young, measured 3.3374 = σ₂)' },
          { topic: 'pca', value: 'μ + Σ_{j≤k} z_ij v_j, error = Σ_{j>k} λ_j (measured 0.2784 = λ₂)' },
        ],
      },
    ],
    notes: [
      'The two topics share byte-identical data synthesis — same n/corr/rotDeg/noise/seed produce the same cloud, so the comparison is apples-to-apples.',
      'PCA is the SVD of the centered data: same directions, same scores, same reconstruction — only the scaling (σ²/n = λ_cov) and the object being factorized differ.',
      'The mistake "PCA = SVD of the raw data" is caught by the shared centering convention: both topics factorize the CENTERED matrix.',
    ],
  },
  {
    id: 'svd-vs-normal-equation',
    title: 'SVD pseudoinverse vs the Normal Equation for least squares',
    topics: ['pca-svd', 'linear-regression'],
    axes: [
      {
        axis: 'Solution formula',
        entries: [
          { topic: 'pca-svd', value: 'β = VΣ⁺Uᵀy — invert each singular value separately (1/σ_j, 0 for zero ones)' },
          { topic: 'linear-regression', value: 'β = (XᵀX)⁻¹Xᵀy — invert the whole Gram matrix at once' },
        ],
      },
      {
        axis: 'Works when XᵀX is singular (σ₂ = 0)?',
        entries: [
          { topic: 'pca-svd', value: 'yes — 1/0 → 0 gives the minimum-norm solution (the collinear failure data: σ₂ = 0 yet X⁺ exists)' },
          { topic: 'linear-regression', value: 'no — (XᵀX)⁻¹ does not exist and the normal equation is unsolvable' },
        ],
      },
      {
        axis: 'Condition number',
        entries: [
          { topic: 'pca-svd', value: 'σ₁/σ₂ (2.64 on the default run) — no squaring, roughly twice the digits' },
          { topic: 'linear-regression', value: '(σ₁/σ₂)² (≈ 6.96 on the default run) — forming XᵀX squares the condition' },
        ],
      },
      {
        axis: 'Scope',
        entries: [
          { topic: 'pca-svd', value: 'general — rectangular, rank-deficient, and full-rank systems alike' },
          { topic: 'linear-regression', value: 'full-column-rank X only; also needs XᵀX invertible (well-conditioned)' },
        ],
      },
    ],
    notes: [
      'The normal equation is the textbook solution; the SVD pseudoinverse is the numerically safe one. On well-conditioned data they agree — the pseudoinverse becomes essential exactly when the normal equation breaks (rank deficiency, near-collinearity).',
      'The default-run condition numbers make the contrast concrete: the SVD works with 2.64, the normal equation with 6.96 — the squaring penalty grows as σ₂ → 0.',
      'This is the "indirect" use of the SVD: beyond compression and PCA, it is the standard back-end of robust least-squares solvers.',
    ],
  },
];