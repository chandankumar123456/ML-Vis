// src/topics/pca-svd/mistakes.ts
// Measured anchors cited below (n 40, corr 0.7, rotDeg 30, noise 0.15, seed 42):
//   σ₁ = 8.804484511776364, σ₂ = 3.3373516320652685, λ(XᵀX) = 77.51894751810988,
//   λ(cov) = 1.9379736879527472, V₁ = (0.2350, 0.9720) at 76.41°, ratio = 2.6382,
//   errFroK1 = 3.3374 = σ₂, errMseK1 = 0.2784 = σ₂²/n.
//   Sign-flip case (rotDeg 140, seed 2): raw v₁ = (−0.9978, 0.0667) at 176.18°.
import type { Mistake } from '../../engine/types';

export const svdMistakes: Mistake[] = [
  {
    id: 'svd-left-right-confusion',
    pattern: 'Confusing the left and right singular vectors — treating U as the feature directions and V as the per-point coefficients',
    example: 'U \\;\\leftrightarrow\\; \\text{per-POINT coefficients (scores)}, \\quad V \\;\\leftrightarrow\\; \\text{per-FEATURE directions (PCs)}',
    whyWrong: 'The dimensions decide the meaning: U is n × n (one row per data point) and V is d × d (one column per feature). In X = UΣVᵀ, V acts FIRST on the feature frame (the right singular vectors ARE the PCA principal directions) and U holds the per-point coefficients. On the default run V₁ = (0.2350, 0.9720) is the direction at 76.41° — a 2-vector in FEATURE space — while u₁ is a 40-vector, one coefficient per point. Swapping them projects points along "directions" that live in the wrong space: every reconstruction, score, and PCA comparison silently breaks.',
    gateTrap: true,
    relatedConcept: 'svd-factorization',
  },
  {
    id: 'svd-sigma-not-diagonal-sorted',
    pattern: 'Forgetting that Σ is diagonal with SORTED entries — or placing the singular values off the diagonal',
    example: '\\Sigma = \\operatorname{diag}(\\sigma_1, \\sigma_2), \\; \\sigma_1 \\ge \\sigma_2 \\ge 0 \\;\\neq\\; \\text{any matrix with entries in arbitrary positions}',
    whyWrong: 'The SVD is a rotation–stretch–rotation: Σ is the STRETCH and it must be diagonal, with σ₁ ≥ σ₂ so that V₁ is the direction of largest energy. On the default run σ₁ = 8.8045, σ₂ = 3.3374, ratio 2.64. A non-diagonal "Σ" stops being an SVD entirely (it would fold a shear into the factorization), and swapping the order makes V₁ the MINIMUM-energy direction — the Eckart–Young guarantee ("the truncated SVD is optimal") fails at once.',
    gateTrap: true,
    relatedConcept: 'svd-eckart-young',
  },
  {
    id: 'svd-needs-square',
    pattern: 'Thinking the SVD requires a SQUARE matrix, or that it is just "eigendecomposition for non-square matrices"',
    example: '\\text{SVD: } X \\in \\mathbb{R}^{n \\times d} \\text{ ANY } n, d \\;\\neq\\; \\text{"only square matrices have singular values"}',
    whyWrong: 'The SVD exists for EVERY real matrix — rectangular or square, full-rank or not — because it does not diagonalize X itself but its symmetric Gram matrix XᵀX (d × d). The singular values are the square roots of the Gram eigenvalues, and the extra rows/columns of the full U, Σ just carry zeros. On this topic the data matrix is n × 2 with n = 40 — rectangular by design — and the module factorizes it exactly. The confusion usually comes from the fact that EIGEN-decomposition needs a square (and ideally symmetric) matrix, while the SVD is its generalization.',
    gateTrap: true,
    relatedConcept: 'svd-gram-eigen',
  },
  {
    id: 'svd-sign-ambiguity',
    pattern: 'Ignoring the sign ambiguity of singular vectors — assuming (v, u) is the only valid SVD when ±v, ±u both are',
    example: '\\pm u_k, \\pm v_k \\text{ are both valid — } \\sigma_k u_k v_k^T = \\sigma_k (-u_k)(-v_k)^T',
    whyWrong: 'Eigenvectors (and hence singular vectors) are defined up to a sign: flipping both u_k and v_k leaves the product σ_k u_k v_kᵀ unchanged, so X = UΣVᵀ still holds. But a raw solver returns an ARBITRARY orientation: measured on rotDeg 140 / seed 2, the raw v₁ = (−0.9978, 0.0667) at 176.18° — the module\'s deterministic convention (largest-|component| positive) flips it to (0.9978, −0.0667) so every run, every seed, and every topic comparison uses the SAME orientation. Without a convention, PCA loadings and reconstructions flip sign between runs and "v₁ points left" today, "right" tomorrow.',
    gateTrap: true,
    relatedConcept: 'svd-factorization',
  },
  {
    id: 'svd-sigma-vs-lambda',
    pattern: 'Confusing singular values σ with eigenvalues λ — using λ where σ belongs (or forgetting the √ / 1/n scales)',
    example: '\\sigma_k = \\sqrt{\\lambda_k(X^T X)} = \\sqrt{n \\cdot \\lambda_k(\\text{cov})}, \\quad \\sigma_k \\neq \\lambda_k \\text{ (and } \\lambda_k \\neq \\sigma_k^2/n \\text{ only for the covariance)}',
    whyWrong: 'The singular values are the square roots of the Gram eigenvalues: on the default run λ₁(XᵀX) = 77.52 but σ₁ = 8.80 (√77.52) — using λ directly overstates the stretch by a factor of √λ. Against the covariance the scaling is σ²/n = λ_cov: λ_cov₁ = 1.938, and √(40·1.938) = 8.80 = σ₁. The three quantities λ(XᵀX), λ(cov), and σ are related but NOT interchangeable — a GATE question that asks for "the second singular value" wants 3.34, not 11.14 (λ₂) and not 0.28 (λ_cov₂).',
    gateTrap: true,
    relatedConcept: 'svd-gram-eigen',
  },
  {
    id: 'svd-forgetting-centering',
    pattern: 'Running the SVD on the RAW (uncentered) data matrix and calling the result "the PCA directions"',
    example: '\\text{SVD}(X) \\;\\text{vs}\\; \\text{SVD}(X_c), \\quad X_c = X - \\mathbf{1}\\mu^T',
    whyWrong: 'PCA directions are eigenvectors of the COVARIANCE, which is built from the CENTERED matrix. An uncentered XᵀX mixes the mean offset μμᵀ into the Gram matrix — the dominant direction then points at the mean (‖μ‖² ≈ 8.93 on the default run, four times the total data variance 2.216) instead of at the variance structure. The module factorizes the CENTERED design matrix (mean μ = (2.5121, 1.6170) subtracted), exactly as the sibling pca topic does — same cloud, same PCs, apples-to-apples.',
    gateTrap: true,
    relatedConcept: 'svd-factorization',
  },
];