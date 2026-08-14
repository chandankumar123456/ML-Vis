// src/topics/pca-svd/derivations.ts
// Measured anchors (n 40, corr 0.7, rotDeg 30, noise 0.15, seed 42 — the default,
// ALL verified by running the module):
//   XᵀX = [[14.8047, 15.1645],[15.1645, 73.8521]], λ(XᵀX) = 77.51894751810988, 11.137915916048712
//   σ₁ = 8.804484511776364, σ₂ = 3.3373516320652685, V₁ = (0.2350, 0.9720) at 76.41°
//   u₁·u₂ = 3.9e-18, ‖X − UΣVᵀ‖∞ = 4.44e-16, errFroK1 = 3.33735163206527 = σ₂,
//   errMseK1 = 0.278447897901218 = σ₂²/n, errFroK2 = 1.28e-15.
import type { Derivation } from '../../engine/types';

export const svdDerivations: Derivation[] = [
  {
    id: 'svd-gram-eigen-link',
    title: 'XᵀX = VΣ²Vᵀ: the Gram Matrix Carries the Whole SVD',
    steps: [
      {
        latex: 'X = U \\Sigma V^T \\;\\Rightarrow\\; X^T = V \\Sigma^T U^T',
        justification: 'Start from the factorization. Transposing flips the order of the factors and transposes each one; U and V are orthonormal, Σ stays diagonal.',
      },
      {
        latex: 'X^T X = (V \\Sigma^T U^T)(U \\Sigma V^T) = V \\Sigma^T (U^T U) \\Sigma V^T = V \\Sigma^T \\Sigma V^T = V \\Sigma^2 V^T',
        justification: 'Multiply the two factors. UᵀU = I collapses the middle, and ΣᵀΣ = diag(σ₁², σ₂²) = Σ². So the Gram matrix of X equals the eigen-decomposition of a symmetric matrix with eigenvalues σ_k² and eigenvectors V.',
      },
      {
        latex: 'X^T X \\text{ symmetric } \\Rightarrow \\lambda_k(X^T X) = \\sigma_k^2, \\quad V = \\text{eigenvectors of } X^T X',
        justification: 'Because XᵀX is symmetric positive semi-definite, its eigen-decomposition is unique: the eigenvalues are the squared singular values and the eigenvectors ARE the right singular vectors. This is why the module solves the SVD through the exact 2×2 eigen closed form of the Gram matrix instead of any iterative solver.',
      },
      {
        latex: '\\sigma_k = \\sqrt{\\lambda_k(X^T X)} = \\sqrt{n \\cdot \\lambda_k\\big(\\tfrac{1}{n}X^T X\\big)}',
        justification: 'The last step ties SVD to PCA: the covariance (1/n)XᵀX has eigenvalues σ_k²/n. Measured on the default run: σ₁² = 77.51894751810988 = λ₁(XᵀX) to 1e-9, and σ₁²/n = 1.9379736879527472 = λ₁(covariance). The single build-up step "XᵀX eigen-decomposition → V, λ" animates exactly this identity side-by-side with the SVD factors (same V, λ = σ²).',
      },
    ],
    derivedFrom: ['svd-factorization'],
  },
  {
    id: 'svd-u-construction',
    title: 'u_k = Xv_k/σ_k: Building U Column-by-Column',
    steps: [
      {
        latex: '\\|X v_k\\|^2 = v_k^T X^T X v_k = v_k^T \\lambda_k v_k = \\lambda_k',
        justification: 'v_k is a unit eigenvector of XᵀX with eigenvalue λ_k (derivation svd-gram-eigen-link), so the squared norm of the projection vector Xv_k is the Rayleigh quotient v_kᵀXᵀXv_k = λ_k = σ_k².',
      },
      {
        latex: '\\|X v_k\\| = \\sigma_k \\;\\Rightarrow\\; u_k = \\frac{X v_k}{\\sigma_k} \\text{ is unit}',
        justification: 'Dividing the length-σ_k vector Xv_k by σ_k yields a vector of unit norm. Measured on the default run: ‖Xv₁‖ = 8.804484511776364 = σ₁ exactly.',
      },
      {
        latex: 'u_i^T u_j = \\frac{(X v_i)^T (X v_j)}{\\sigma_i \\sigma_j} = \\frac{v_i^T X^T X v_j}{\\sigma_i \\sigma_j} = \\frac{\\lambda_j \\, v_i^T v_j}{\\sigma_i \\sigma_j} = 0 \\quad (i \\neq j)',
        justification: 'Orthogonality of the u\'s follows from orthogonality of the v\'s: v_iᵀv_j = 0 for i ≠ j. So the column-by-column construction produces a genuinely orthonormal U — measured u₁·u₂ = 3.9e-18 on the default run, and the full reconstruction ‖X − UΣVᵀ‖∞ = 4.44e-16.',
      },
      {
        latex: '\\sigma_k = 0 \\;\\Rightarrow\\; u_k = \\frac{X v_k}{0} \\text{ is } 0/0 \\;\\Rightarrow\\; \\text{deterministic null-space completion}',
        justification: 'The one honest caveat: for a rank-deficient direction (σ_k ≈ 0, e.g. the collinear failure data with σ₂ = 0) the quotient is undefined. The module completes u₂ deterministically (Gram–Schmidt of the first standard-basis vector against u₁), flags rankDeficient, and the reconstruction stays exact — because σ₂ = 0, u₂ contributes nothing to X = UΣVᵀ anyway.',
      },
    ],
    derivedFrom: ['svd-gram-eigen-link', 'svd-factorization'],
  },
  {
    id: 'svd-eckart-young',
    title: 'Eckart–Young: the Best Rank-k Approximation is the Truncated SVD',
    steps: [
      {
        latex: 'X = \\sum_{j=1}^{d} \\sigma_j u_j v_j^T \\;\\Rightarrow\\; X - \\hat{X}_k = \\sum_{j > k} \\sigma_j u_j v_j^T',
        justification: 'The SVD expands X into rank-1 layers σ_j u_j v_jᵀ. Truncating to the top-k layers leaves exactly the layers j > k — a residual that is itself an SVD with singular values σ_{k+1} ≥ σ_{k+2} ≥ ….',
      },
      {
        latex: '\\|X - \\hat{X}_k\\|_2 = \\sigma_{k+1}, \\qquad \\|X - \\hat{X}_k\\|_F^2 = \\sum_{j > k} \\sigma_j^2',
        justification: 'The spectral norm of a matrix is its largest singular value — here σ_{k+1}. The Frobenius norm is the square root of the sum of squared singular values, so in 2D (only σ₂ dropped at k = 1) BOTH norms equal σ₂ exactly.',
      },
      {
        latex: '\\text{For any rank-}k \\; B: \\; \\|X - B\\|_2 \\ge \\sigma_{k+1} \\;\\Rightarrow\\; \\hat{X}_k \\text{ is optimal}',
        justification: 'Eckart–Young–Mirsky: no rank-k matrix can beat σ_{k+1} in the spectral norm, and the truncated SVD attains it — so the truncated SVD IS the best rank-k approximation. The rank slider animates exactly this: k = 1 gives the best single-direction picture, k = 2 reconstructs the plane.',
      },
      {
        latex: '\\text{MSE}_k = \\frac{1}{n}\\|X - \\hat{X}_k\\|_F^2 = \\frac{\\sigma_{k+1}^2}{n} \\;\\Rightarrow\\; \\text{measured: } \\frac{3.33735163206527^2}{40} = 0.278447897901218',
        justification: 'The module measures the residual from the ACTUAL reconstructed matrix X̂_k (not just the closed form) and reports the per-sample mean squared error σ_{k+1}²/n as the loss-curve metric. On the default run: errFroK1 = 3.33735163206527 = σ₂ (1e-12), errMseK1 = 0.278447897901218 = σ₂²/n = covλ₂, and at k = 2 the reconstruction is exact (errFro = 1.28e-15).',
      },
    ],
    derivedFrom: ['svd-eckart-young', 'svd-eckart-young-error'],
  },
];