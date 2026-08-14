// src/topics/pca-svd/formulas.ts
// Measured anchors (n 40, corr 0.7, rotDeg 30, noise 0.15, seed 42 — the default,
// ALL verified by running the module):
//   XᵀX = [[14.8047, 15.1645],[15.1645, 73.8521]], λ(XᵀX) = 77.51894751810988, 11.137915916048712
//   λ(cov) = 1.9379736879527472, 0.2784478979012178 (= σ²/n)
//   σ₁ = 8.804484511776364, σ₂ = 3.3373516320652685, ratio = 2.6381650729227606
//   V₁ = (0.2350, 0.9720) at 76.40663404901252°, V₂ = (0.9720, −0.2350) ⊥ V₁ (exact)
//   rank-0 baseline ‖X‖²_F/n = 2.216421585853965; errFroK1 = 3.33735163206527 = σ₂
//   errMseK1 = 0.278447897901218 = σ₂²/n; errFroK2 = 1.28e-15 (exact reconstruction)
import type { Formula } from '../../engine/types';

export const svdFormulas: Formula[] = [
  {
    id: 'svd-factorization',
    latex: 'X = U \\Sigma V^T, \\quad X \\in \\mathbb{R}^{n \\times d},\\; U \\in \\mathbb{R}^{n \\times n},\\; \\Sigma \\in \\mathbb{R}^{n \\times d},\\; V \\in \\mathbb{R}^{d \\times d}',
    symbols: [
      { symbol: 'X', meaning: 'the centered design matrix — every row is one data point, every column one feature (here n × 2)', dimensions: 'n × d' },
      { symbol: 'U', meaning: 'left singular vectors — orthonormal columns; UᵀU = I; column k holds the per-point coefficients of the k-th singular direction', dimensions: 'n × n' },
      { symbol: '\\Sigma', meaning: 'diagonal matrix of singular values σ₁ ≥ σ₂ ≥ 0 — the "energy" of each direction; only the first min(n,d) entries are nonzero', dimensions: 'n × d' },
      { symbol: 'V', meaning: 'right singular vectors — orthonormal; VᵀV = I; these are the PCA principal directions for the centered data', dimensions: 'd × d' },
    ],
    assumptions: ['X is a complete, finite matrix — a single NaN anywhere propagates through every factor (the missing-data failure)', 'U and V are orthonormal: the SVD is a rotation (Vᵀ), a stretch (Σ), and a rotation (U) of the data frame'],
    failureCases: ['Missing/NaN entries: X must be COMPLETE — the SVD has no missing-data counterpart (unlike EM for Gaussians); the module fails honestly via telemetry', 'Zero-variance data (X = 0): Σ is all zeros and the singular directions are undefined — the module throws (telemetry failure)', 'Reading the wrong factor: U is per-POINT coefficients, V is per-FEATURE directions — swapping them projects points along the wrong axes'],
    derivesFrom: [],
    derivationIds: ['svd-gram-eigen-link', 'svd-u-construction'],
    connections: ['svd-gram-eigen', 'svd-eckart-young', 'PCA: X_c = UΣVᵀ', 'Eigen-decomposition (symmetric case)'],
    whyWorks: 'Any real matrix factors into a rotation–stretch–rotation: Vᵀ rotates the feature frame onto the directions where the data spreads most, Σ stretches by the singular values, and U rotates the result back into data space. On the default run the factorization is exact to 4.4e-16 (‖X − UΣVᵀ‖∞ measured), with σ₁ = 8.8045 and σ₂ = 3.3374 — the two "energies" of the cloud.',
  },
  {
    id: 'svd-gram-eigen',
    latex: 'X^T X = V \\Sigma^2 V^T \\;\\Rightarrow\\; \\sigma_k = \\sqrt{\\lambda_k(X^T X)}',
    symbols: [
      { symbol: 'X^T X', meaning: 'the Gram matrix of the centered design matrix — the symmetric PSD matrix whose eigen-decomposition carries the ENTIRE SVD', dimensions: 'd × d' },
      { symbol: 'V \\Sigma^2 V^T', meaning: 'the eigen-decomposition of XᵀX: same V as the SVD, eigenvalues λ_k = σ_k² — the SVD and eigen links, exactly', dimensions: 'd × d' },
      { symbol: '\\sigma_k', meaning: 'k-th singular value = √λ_k(XᵀX) — the stretch of the data along direction k', dimensions: 'feature units' },
    ],
    assumptions: ['XᵀX is symmetric positive semi-definite (always true for a Gram matrix), so its eigenvalues are real and ≥ 0 — the square root is well-defined', 'The columns of V are sorted to match λ₁ ≥ λ₂, which forces σ₁ ≥ σ₂'],
    failureCases: ['Taking σ_k = λ_k directly (forgetting the square root): on the default run λ₁ = 77.52 but σ₁ = 8.80 — the error magnitude is the SQUARE root of the variance, a 8.8× overestimate', 'Using the eigenvalues of the COVARIANCE (1/n)XᵀX instead of the Gram matrix: λ_cov = σ²/n, so σ = √(n·λ_cov) — mixing the two conventions off by √n (on the default run, off by √40 ≈ 6.32)', 'Non-symmetric matrices: XᵀX is symmetric by construction, but a general A has no such eigen shortcut — the full SVD is required (the comparison vs eigen-decomposition)'],
    derivesFrom: ['svd-factorization'],
    derivationIds: ['svd-gram-eigen-link'],
    connections: ['svd-factorization', 'svd-u-construction', 'PCA: λ_cov = σ²/n', 'Eigen-decomposition'],
    whyWorks: 'Because V is orthonormal, XᵀX = (UΣVᵀ)ᵀ(UΣVᵀ) = VΣᵀUᵀUΣVᵀ = VΣ²Vᵀ — the symmetric eigen-decomposition of the Gram matrix with eigenvalues σ_k². This is the computational heart of the module: the exact 2×2 eigen closed form (λ₁,₂ = (a+c ± √((a−c)²+4b²))/2, θ = ½·atan2(2b, a−c)) produces V and σ² in one shot. Measured: σ₁² = 77.51894751810988 = λ₁(XᵀX) to 1e-9, and σ₁²/n = 1.9379736879527472 = λ₁(covariance) — the PCA link.',
  },
  {
    id: 'svd-u-construction',
    latex: 'u_k = \\frac{X v_k}{\\sigma_k}, \\quad \\|X v_k\\| = \\sqrt{v_k^T X^T X v_k} = \\sqrt{\\lambda_k} = \\sigma_k',
    symbols: [
      { symbol: 'u_k', meaning: 'k-th left singular vector — the unit vector of per-point coefficients for direction k (column k of U)', dimensions: 'n-vector' },
      { symbol: 'X v_k', meaning: 'the projection of every data point onto the right singular direction v_k (n-vector of dot products)', dimensions: 'n-vector' },
      { symbol: '\\|X v_k\\|', meaning: 'the norm of those projections = σ_k — the reason dividing by σ_k yields a UNIT vector', dimensions: 'feature units' },
    ],
    assumptions: ['σ_k > 0: for a rank-deficient direction (σ_k ≈ 0) the quotient is 0/0 — the module completes it deterministically instead (null-space completion, flagged rankDeficient)', 'v_k is a unit eigenvector of XᵀX (from svd-gram-eigen), so the construction is exact'],
    failureCases: ['Dividing by σ_k ≈ 0 on rank-deficient data: u_k = Xv_k/σ_k is undefined — a naive implementation produces NaN (the degenerate-rank failure); the module replaces it with a documented completion', 'Forgetting the normalization: Xv_k has norm σ_k, not 1 — using it raw breaks UᵀU = I and the reconstruction'],
    derivesFrom: ['svd-gram-eigen'],
    derivationIds: ['svd-u-construction'],
    connections: ['svd-factorization', 'svd-gram-eigen', 'PCA scores: Xv_k'],
    whyWorks: 'u_k = Xv_k/σ_k is the exact left singular vector: its norm is ‖Xv_k‖/σ_k = σ_k/σ_k = 1 (shown above), and orthogonality of the v vectors makes the u vectors orthogonal to machine precision (measured u₁·u₂ = 3.9e-18 on the default run). Every u_k is unit, UᵀU = I holds, and the factorization X = UΣVᵀ reconstructs the data exactly (4.4e-16).',
  },
  {
    id: 'svd-eckart-young',
    latex: '\\hat{X}_k = \\sum_{j \\le k} \\sigma_j u_j v_j^T = U_k \\Sigma_k V_k^T',
    symbols: [
      { symbol: '\\hat{X}_k', meaning: 'the rank-k truncated SVD — the best rank-k approximation of X', dimensions: 'n × d' },
      { symbol: 'U_k, \\Sigma_k, V_k^T', meaning: 'the ECONOMY factors: the first k columns of U, the k×k diagonal of the top-k singular values, and the first k rows of Vᵀ', dimensions: 'n×k, k×k, k×d' },
      { symbol: '\\sigma_j u_j v_j^T', meaning: 'the j-th rank-1 layer of the factorization — one direction, one energy, one set of per-point coefficients', dimensions: 'n × d' },
    ],
    assumptions: ['Singular values sorted σ₁ ≥ σ₂ ≥ 0 so truncation keeps the largest energies', 'The approximation is measured in the spectral or Frobenius norm (see svd-eckart-young-error)'],
    failureCases: ['Truncating WITHOUT re-sorting: if σ₂ > σ₁ were kept while dropping σ₁ the "rank-1" approximation would be the WORST rank-1 matrix (Eckart–Young applies to the sorted truncation only)', 'Using non-economy factors: U (n×n) times Σ (n×d) wastes the zero rows/columns — the economy form carries exactly the same product with smaller matrices'],
    derivesFrom: ['svd-factorization'],
    derivationIds: ['svd-eckart-young'],
    connections: ['svd-eckart-young-error', 'Image compression (a low-rank image IS X̂_k)', 'PCA reconstruction (project then invert)'],
    whyWorks: 'Eckart–Young–Mirsky: among ALL rank-k matrices, X̂_k = U_kΣ_kV_kᵀ minimizes both ‖X − X̂‖_F and ‖X − X̂‖₂. On the default run X̂₁ = σ₁u₁v₁ᵀ is the best single-direction picture of the cloud and X̂₂ = X exactly (measured residual 1.28e-15). The rank slider animates exactly this: k = 1 keeps the dominant direction, k = 2 reconstructs the whole plane.',
  },
  {
    id: 'svd-eckart-young-error',
    latex: '\\|X - \\hat{X}_k\\|_F = \\sqrt{\\sum_{j > k} \\sigma_j^2} = \\sigma_{k+1}\\ (\\text{2D}), \\quad \\|X - \\hat{X}_k\\|_2 = \\sigma_{k+1}',
    symbols: [
      { symbol: '\\|X - \\hat{X}_k\\|_F', meaning: 'Frobenius norm of the residual — the total dropped energy; in 2D (only σ_{k+1} dropped) it equals σ_{k+1} exactly', dimensions: 'feature units' },
      { symbol: '\\|X - \\hat{X}_k\\|_2', meaning: 'spectral norm (largest singular value) of the residual — also exactly σ_{k+1} (Eckart–Young)', dimensions: 'feature units' },
      { symbol: '\\sigma_{k+1}', meaning: 'the first DROPPED singular value — the single number that bounds the error of the best rank-k approximation', dimensions: 'feature units' },
    ],
    assumptions: ['2D data (d = 2) so the k = 1 residual has exactly one dropped direction σ₂ — the identity the tests assert; in d dimensions the Frobenius error sums σ_{k+1}²…σ_d²', 'The residual is measured from the ACTUAL reconstructed matrix X̂_k, not just the closed form — the module builds X̂_k and measures'],
    failureCases: ['Forgetting the square root in the Frobenius norm: the ERROR is σ₂, while the squared error is σ₂² — the loss metric uses the per-sample MSE (1/n)‖X−X̂_k‖²_F = σ₂²/n, a different number (measured: 0.2784 vs 3.3374)', 'Claiming rank-k error is σ_k (keeping instead of dropping): the error is governed by what you DROP, not what you keep'],
    derivesFrom: ['svd-eckart-young'],
    connections: ['svd-eckart-young', 'The loss-curve (reconstructionError) plots exactly σ_{k+1}²/n', 'Image compression error vs rank'],
    whyWorks: 'The residual X − X̂_k = Σ_{j>k} σ_j u_j v_jᵀ is itself an SVD with singular values σ_{k+1}, σ_{k+2}, …: its Frobenius norm is √(Σ_{j>k} σ_j²) and its spectral norm is σ_{k+1}. In 2D only σ₂ remains at k = 1, so BOTH norms equal σ₂ — measured 3.33735163206527 (Frobenius) and 3.337351632065269 (spectral) on the default run, and the per-sample MSE σ₂²/n = 0.278447897901218.',
  },
  {
    id: 'svd-economy',
    latex: 'X = U_k \\Sigma_k V_k^T, \\quad U_k \\in \\mathbb{R}^{n \\times k},\\; \\Sigma_k \\in \\mathbb{R}^{k \\times k},\\; V_k^T \\in \\mathbb{R}^{k \\times d}',
    symbols: [
      { symbol: 'U_k', meaning: 'economy left factor — first k columns of U; the per-point coefficients for the top-k directions', dimensions: 'n × k' },
      { symbol: '\\Sigma_k', meaning: 'economy diagonal — the top-k singular values; the zero rows/columns of the full Σ are dropped', dimensions: 'k × k' },
      { symbol: 'V_k^T', meaning: 'economy right factor — first k rows of Vᵀ; the top-k feature directions', dimensions: 'k × d' },
    ],
    assumptions: ['k ≤ min(n, d): the full SVD keeps at most min(n,d) nonzero singular values; the rank slider here is k ∈ {1, 2}', 'U_kᵀU_k = I_k and V_kᵀV_k = I_k — the economy factors stay orthonormal'],
    failureCases: ['Forgetting the dimensions: U_kΣ_kV_kᵀ is n×d ONLY because Σ_k is k×k — writing U (n×n)·Σ(n×d) wastes space and invites shape bugs', 'KEEPING zero singular values: a rank-deficient matrix has σ₂ = 0, so the economy k = 1 form is already exact — the economy factorization makes the "effective rank" visible'],
    derivesFrom: ['svd-factorization'],
    connections: ['svd-eckart-young', 'Rank-revealing form of the SVD', 'Dimension counts in the matrix-animator view'],
    whyWorks: 'All singular values beyond min(n,d) are zero and every zero singular value contributes nothing to X = UΣVᵀ, so the zero rows of Σ (and the matching U columns) can be dropped without changing the product. The module\'s matrix-animator shows the economy factors with their exact dims: U₁ (40×1), Σ₁ (1×1), V₁ᵀ (1×2) at rank 1 and U₂ (40×2), Σ₂ (2×2), V₂ᵀ (2×2) at rank 2.',
  },
  {
    id: 'svd-pseudoinverse',
    latex: 'X^+ = V \\Sigma^+ U^T, \\quad \\Sigma^+ = \\operatorname{diag}\\big(1/\\sigma_1, \\ldots, 1/\\sigma_k, 0, \\ldots\\big), \\quad \\hat{\\beta} = X^+ y',
    symbols: [
      { symbol: 'X^+', meaning: 'the Moore–Penrose pseudoinverse of X — the "least-squares inverse" that works even when X is rectangular or rank-deficient', dimensions: 'd × n' },
      { symbol: '\\Sigma^+', meaning: 'the inverted singular values — 1/σ_j for the nonzero σ_j, 0 for the zero ones (the honest handling of rank deficiency)', dimensions: 'd × n' },
      { symbol: '\\hat{\\beta} = X^+ y', meaning: 'the minimum-norm least-squares solution of Xβ = y — the SVD route to ordinary least squares', dimensions: 'd-vector' },
    ],
    assumptions: ['X has the SVD X = UΣVᵀ (it always does — the pseudoinverse is DEFINED via the SVD)', 'σ_j = 0 entries are inverted as 0, not 1/0 — the regularization of the pseudoinverse'],
    failureCases: ['Inverting the zero singular values: 1/0 explodes — the pseudoinverse convention (0) is what makes rank-deficient OLS solvable (the collinear failure data: σ₂ = 0 yet X⁺ is well-defined)', 'Using (XᵀX)⁻¹Xᵀ on a rank-deficient X: XᵀX is singular and the normal equation is NOT solvable — the pseudoinverse is the numerically stable alternative (the normal-equation comparison)', 'Wrong order: X⁺ = VΣ⁺Uᵀ, not UΣ⁺Vᵀ — the factors invert order when transposed'],
    derivesFrom: ['svd-factorization'],
    connections: ['svd-eckart-young', 'Linear regression / normal equation', 'Numerically stable least squares (the normal-equation comparison)'],
    whyWorks: 'The SVD diagonalizes the least-squares problem: with X = UΣVᵀ, the normal equation XᵀXβ = Xᵀy becomes VΣ²Vᵀβ = VΣUᵀy, so β = VΣ⁺Uᵀy. Inverting each singular value separately (1/σ_j) is far more stable than inverting the Gram matrix XᵀX, whose condition number is (σ₁/σ₂)² — squared relative to the SVD\'s own σ₁/σ₂. On the default run σ₁/σ₂ = 2.64, so the Gram matrix has condition ~6.96 while the SVD route works directly with 2.64.',
  },
];