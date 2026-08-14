// src/topics/pca/formulas.ts
// Measured anchors (n 40, corr 0.7, rotDeg 30, noise 0.15, seed 42 — the default,
// ALL verified by running the module):
//   centered covariance Σ = [[0.3701, 0.3791],[0.3791, 1.8463]] (measured),
//   λ₁ = 1.9379736879527472, λ₂ = 0.2784478979012177, total = 2.216421585853965,
//   ratio₁ = 0.8744, ratio₂ = 0.1256,
//   PC1 = (0.2350, 0.9720) at θ₁ = 76.41°, PC2 ⊥ PC1 (exact by construction),
//   reconErrK1 = 0.2784 (= λ₂), reconErrK2 = 0.
//   Centering contrast (rotDeg 80): centered PC1 at 127.15°; RAW (uncentered)
//   covariance PC1 at 33.30° — the uncentered PCs point at the data mean μ
//   (mean direction ≈ 32.8°), and explain only 12.6% of the CENTERED variance.
import type { Formula } from '../../engine/types';

export const pcaFormulas: Formula[] = [
  {
    id: 'pca-covariance',
    latex: '\\Sigma = \\frac{1}{n} X_c^T X_c = \\frac{1}{n}\\sum_{i=1}^n (x_i - \\mu)(x_i - \\mu)^T',
    symbols: [
      { symbol: 'X_c', meaning: 'the CENTERED design matrix — every row is xᵢ − μ (mean-subtracted); X_c = X − 1·μᵀ', dimensions: 'n × 2' },
      { symbol: '\\mu', meaning: 'empirical mean (1/n)Σᵢ xᵢ — the data is centered on it before any covariance is formed', dimensions: 'feature units' },
      { symbol: '\\Sigma', meaning: 'the sample covariance matrix — encodes the spread AND the correlation of the cloud', dimensions: 'feature²' },
      { symbol: 'n', meaning: 'number of points (the division is by n — the sample, not n−1; the convention of this topic)', dimensions: 'count' },
    ],
    assumptions: ['The data is CENTERED first — without this, (1/n)XᵀX = Σ + μμᵀ and the eigenvectors point at the mean, not the variance (measured: the uncentered PC1 sits at 33.30° instead of 127.15° on the rotDeg-80 config, and its PC1 explains only 12.6% of the CENTERED variance)', 'Real-valued features; the covariance is symmetric (2×2 here) and positive semi-definite'],
    failureCases: ['Forgetting to center: the raw covariance mixes the mean offset μμᵀ into the matrix and rotates every PC (the plan\'s #1 PCA mistake — measured in test case 4)', 'Zero-variance data (all points identical): Σ is the zero matrix and the eigen-solve throws (honest telemetry failure)', 'Too few points (n < 3): the sample covariance is rank-deficient and PC2 is meaningless'],
    derivesFrom: [],
    derivationIds: ['pca-variance-along-w'],
    connections: ['pca-rayleigh', 'pca-eigenequation', 'SVD: XᵀX = VΣ²Vᵀ', 'Statistics: sample covariance'],
    whyWorks: 'Each entry of Σ is an inner product of centered coordinates: Σ₁₁ = (1/n)Σ(xᵢ−μx)² is the x-variance, Σ₂₂ = (1/n)Σ(yᵢ−μy)² the y-variance, Σ₁₂ = (1/n)Σ(xᵢ−μx)(yᵢ−μy) the x–y covariance. Along any unit direction w the quadratic form wᵀΣw is exactly the variance of the projections — the quantity PCA maximizes. On the default run Σ = [[0.3701, 0.3791],[0.3791, 1.8463]] shows the y-direction spreads more (1.8463 > 0.3701) and the two features co-move positively (0.3791) — so PC1 points up-right at 76.4°.',
  },
  {
    id: 'pca-rayleigh',
    latex: 'R(w) = \\frac{w^T \\Sigma w}{w^T w} = \\operatorname{Var}\\big(\\text{projections of the data on } w\\big)',
    symbols: [
      { symbol: 'R(w)', meaning: 'the Rayleigh quotient — the variance of the data projected onto the direction w, for a unit w', dimensions: 'feature²' },
      { symbol: 'w^T \\Sigma w', meaning: 'quadratic form of the covariance (for a UNIT w this IS the projected variance; the denominator wᵀw = 1 keeps it scale-free)', dimensions: 'feature²' },
      { symbol: '\\operatorname{Var}(\\ldots)', meaning: 'empirical variance of the scalar projections w·(xᵢ − μ) — the loss-curve quantity this topic plots', dimensions: 'feature²' },
    ],
    assumptions: ['w is a unit direction (‖w‖ = 1); the quotient is scale-invariant so the normalization is cosmetic', 'Σ is the centered covariance of pca-covariance'],
    failureCases: ['At w = 0 the quotient is undefined — but a zero direction is never a candidate in the sweep', 'Isotropic data (Σ ∝ I, e.g. the high-noise failure): R(w) is (nearly) the SAME for every w — the Rayleigh quotient has no meaningful peak and PC1 is unstable'],
    derivesFrom: ['pca-covariance'],
    derivationIds: ['pca-variance-along-w', 'pca-maximize-lagrange'],
    connections: ['pca-eigenequation', 'Eigenvalues as extrema', 'Loss curve (axisVariance)'],
    whyWorks: 'R(w) is the objective PCA maximizes: the direction of maximum projected variance. The sweep rotates w and plots R(w) — a smooth curve over [0°, 180°) whose single peak sits at the first eigenvector. On the default run the curve rises from R(0°) = 0.3701 (the x-variance) to its peak R(θ₁) = λ₁ = 1.9380 at θ₁ = 76.4°, then falls back — the peak IS PC1.',
  },
  {
    id: 'pca-eigenequation',
    latex: '\\Sigma v = \\lambda v, \\quad \\lambda_1 \\ge \\lambda_2 \\ge 0',
    symbols: [
      { symbol: 'v', meaning: 'an eigenvector of Σ — a principal component direction (unit vector)', dimensions: 'feature units' },
      { symbol: '\\lambda', meaning: 'an eigenvalue of Σ — the variance of the data along v (λ_k = Var of projections on v_k)', dimensions: 'feature²' },
      { symbol: '\\lambda_1 \\ge \\lambda_2', meaning: 'eigenvalues sorted descending — λ₁ belongs to PC1 (max-variance direction)', dimensions: 'feature²' },
    ],
    assumptions: ['Σ is symmetric (it is: the sample covariance) — so its eigenvectors are ORTHOGONAL and its eigenvalues real and non-negative', '2×2 closed form (this topic): Δ = (a−c)² + 4b², λ₁,₂ = (a+c ± √Δ)/2, θ = ½·atan2(2b, a−c) — exact, no iterative solver'],
    failureCases: ['Sorting mistakes: swapping λ₁ and λ₂ makes PC1 the MINIMUM-variance direction (the plan\'s eigenvalue-ordering mistake)', 'Degenerate covariance (λ₂ ≈ 0, the noise-0/|corr|→1 warning): the second eigenvalue is meaningless and PC2 is numerically unstable', 'Non-symmetric matrices need SVD-style treatment — the eigen-decomposition of XᵀX is the SVD of X (the pca-svd link)'],
    derivesFrom: ['pca-covariance'],
    derivationIds: ['pca-maximize-lagrange'],
    connections: ['pca-rayleigh', 'pca-explained', 'pca-svd-link', 'SVD: eigenvalues of XᵀX = squared singular values'],
    whyWorks: 'The stationary points of the Rayleigh quotient are exactly the eigenvectors (derivation 2), and its extrema are the eigenvalues. So solving Σv = λv IS finding the variance-maximizing directions. The 2×2 closed form is exact: the characteristic polynomial (a−λ)(c−λ) − b² = 0 has roots (a+c ± √((a−c)² + 4b²))/2, and the major-axis angle θ = ½·atan2(2b, a−c) gives v₁ = (cos θ, sin θ) — verified Σv = λv to 1e-9 on the default run (λ₁ = 1.9380, λ₂ = 0.2784).',
  },
  {
    id: 'pca-projection',
    latex: 'z_i = v_k \\cdot (x_i - \\mu), \\quad P = X_c v_k',
    symbols: [
      { symbol: 'z_i', meaning: 'the score of point i on PC k — its signed position along the principal axis', dimensions: 'feature units' },
      { symbol: 'P', meaning: 'the score/projection matrix — the data expressed in the principal-coordinate frame (n × 1 for k=1)', dimensions: 'n × 1' },
      { symbol: 'X_c v_k', meaning: 'matrix product of the centered data with the k-th principal direction — the projection of every row onto v_k', dimensions: 'n × 1' },
    ],
    assumptions: ['The data is centered before projecting (projection is around μ)', 'v_k is a unit eigenvector — the projection is an orthogonal projection, so the residual is perpendicular to the axis'],
    failureCases: ['Projecting UNcentered data: zᵢ = v·xᵢ instead of v·(xᵢ − μ) shifts every score by the constant v·μ — harmless for variance but wrong for reconstruction', 'Using the raw features as scores without the eigen-directions — that is feature SELECTION, not extraction (see the comparison)'],
    derivesFrom: ['pca-eigenequation'],
    connections: ['pca-reconstruction', 'Dimensionality reduction (n features → k scores)'],
    whyWorks: 'An orthogonal projection of xᵢ onto the line through μ along v_k is μ + (v_k·(xᵢ−μ))v_k; the scalar coefficient zᵢ is the point\'s 1-D coordinate in the PC frame. PCA\'s dimensionality reduction is this projection onto the top-k axes — on the default run projecting onto PC1 keeps 87.4% of the total variance in a single number per point.',
  },
  {
    id: 'pca-explained',
    latex: '\\text{explained ratio}_k = \\frac{\\lambda_k}{\\sum_j \\lambda_j}, \\quad \\text{cumulative}(k) = \\frac{\\sum_{j \\le k} \\lambda_j}{\\sum_j \\lambda_j}',
    symbols: [
      { symbol: '\\lambda_k', meaning: 'k-th eigenvalue (sorted) — the variance captured by PC k', dimensions: 'feature²' },
      { symbol: '\\sum_j \\lambda_j', meaning: 'total variance = trace(Σ) — the full spread of the data', dimensions: 'feature²' },
      { symbol: '\\text{cumulative}(k)', meaning: 'fraction of the total variance retained by the top-k PCs — the k-selection (elbow) criterion', dimensions: 'dimensionless' },
    ],
    assumptions: ['Eigenvalues sorted descending (λ₁ ≥ λ₂) so the ratios are monotone', 'The retained variance is measured on the CENTERED data (uncentered scores inflate the total by the mean-offset term)'],
    failureCases: ['Sorting eigenvalues wrong inverts the ratios and points k-selection at the wrong elbow', 'On near-degenerate data (λ₂ ≈ 0) the ratio₂ ≈ 0 looks like a perfect k=1 choice — but it only means the second dimension was nearly constant, not that PC1 is meaningful', 'In high dimensions the elbow is a heuristic, not a theorem — there is no objectively correct k'],
    derivesFrom: ['pca-eigenequation'],
    connections: ['pca-projection', 'pca-reconstruction', 'Scree / variance-explained bars (the eigenviewer renders these)'],
    whyWorks: 'Trace(Σ) is invariant under rotation, so the eigenvalues partition the total variance into orthogonal directions. The ratio λ_k/Σλ answers "how much of the spread does PC k alone explain?" On the default run PC1 explains 87.4% and PC2 the remaining 12.6% — the eigenviewer\'s two variance bars show exactly this split at the PC1 angle.',
  },
  {
    id: 'pca-reconstruction',
    latex: '\\hat{x}_i = \\mu + \\sum_{j \\le k} z_{ij} v_j, \\quad \\text{error}_k = \\frac{1}{n}\\sum_{i=1}^n \\|x_i - \\hat{x}_i\\|^2 = \\sum_{j > k} \\lambda_j',
    symbols: [
      { symbol: '\\hat{x}_i', meaning: 'the reconstruction of xᵢ from its top-k scores — the orthogonal projection of xᵢ onto the k-dimensional PC subspace', dimensions: 'feature units' },
      { symbol: 'z_{ij}', meaning: 'score of point i on PC j: v_j·(xᵢ − μ)', dimensions: 'feature units' },
      { symbol: '\\text{error}_k', meaning: 'mean squared reconstruction error — the variance lost by dropping PCs k+1…d', dimensions: 'feature²' },
      { symbol: '\\sum_{j > k} \\lambda_j', meaning: 'the SUM OF THE DROPPED EIGENVALUES — the closed form the derivation proves and the tests assert to 1e-12', dimensions: 'feature²' },
    ],
    assumptions: ['The full PCA frame is orthonormal (v₁ ⊥ v₂ in 2D), so reconstructing with all d = 2 PCs is lossless (error = 0)', 'Error is measured in Euclidean (squared) norm per point, averaged over the sample'],
    failureCases: ['Forgetting the mean: reconstructing as Σ zᵢⱼvⱼ without adding μ shifts the whole cloud by −μ and inflates the error by ‖μ‖²', 'Using fewer PCs than the data has structure (k too small at high noise) — the error is then the large dropped eigenvalues, not a small tail'],
    derivesFrom: ['pca-projection', 'pca-explained'],
    derivationIds: ['pca-reconstruction-error'],
    connections: ['pca-projection', 'Eckart–Young (low-rank approximation, pca-svd)', 'Data compression / denoising'],
    whyWorks: 'Because v₁ and v₂ span the plane, the decomposition xᵢ − μ = zᵢ₁v₁ + zᵢ₂v₂ is exact. Dropping PC2 leaves the residual zᵢ₂v₂ per point, whose average squared norm is exactly the variance along v₂ = λ₂. Measured on the default run: error₁ = 0.2784 = λ₂ exactly (1e-12), error₂ = 0. Keeping one PC therefore trades 12.6% of the variance for a 2× compression.',
  },
  {
    id: 'pca-svd-link',
    latex: 'X_c = U \\Sigma_{svd} V^T \\;\\Rightarrow\\; X_c^T X_c = V \\Sigma_{svd}^2 V^T \\;\\Rightarrow\\; \\text{PCs} = V \\;(\\text{right singular vectors}),\\; \\lambda_k = \\sigma_k^2 / n',
    symbols: [
      { symbol: 'U', meaning: 'left singular vectors (n × 2) — the data expressed in the principal frame', dimensions: 'n × 2' },
      { symbol: '\\Sigma_{svd}', meaning: 'diagonal matrix of singular values σ₁ ≥ σ₂ (NOT the covariance — the SVD\'s own diagonal)', dimensions: '2 × 2' },
      { symbol: 'V', meaning: 'right singular vectors — the PCA principal directions v₁, v₂', dimensions: '2 × 2' },
      { symbol: '\\sigma_k^2 / n', meaning: 'link between the SVD and PCA: the eigenvalue of the covariance is the squared singular value scaled by 1/n', dimensions: 'feature²' },
    ],
    assumptions: ['X_c is the centered design matrix (n × 2, rows = points, columns = features)', 'The SVD convention X = UΣVᵀ with orthonormal U and V (the full/compact forms differ only by zero rows/columns)'],
    failureCases: ['Confusing U and V: U holds the SCORES (per-point coefficients), V holds the DIRECTIONS — swapping them projects points onto the wrong axes', 'Sign ambiguity: ±v are both eigenvectors (and ±u both singular vectors) — PCA directions are defined up to a sign flip'],
    derivesFrom: ['pca-covariance'],
    connections: ['pca-eigenequation', 'pca-svd topic', 'Eckart–Young theorem', 'Numerically stable eigendecomposition'],
    whyWorks: 'X_cᵀX_c is a symmetric matrix whose eigen-decomposition the SVD of X_c reproduces: its eigenvectors ARE the right singular vectors V, and its eigenvalues are σ_k²/n. So PCA "is" the SVD of the centered data — the numerically preferred route in practice (the pca-svd topic builds on exactly this identity). The closed-form 2×2 solver in this topic is the exact equivalent for d = 2.',
  },
];