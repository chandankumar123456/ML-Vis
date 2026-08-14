// src/topics/pca/mistakes.ts
// Measured anchors cited below (n 40, corr 0.7, rotDeg 30, noise 0.15, seed 42):
//   default: λ₁ = 1.9379736879527472, λ₂ = 0.2784478979012177, PC1 at 76.41°;
//   rotDeg 80: centered PC1 at 127.15° vs RAW PC1 at 33.30° (raw points at the
//   mean of that draw, direction ≈ 34.06°; default-config raw PC1 at 37.98° vs
//   its mean direction 32.77°); raw PC1 explains only 12.6% of CENTERED variance.
import type { Mistake } from '../../engine/types';

export const pcaMistakes: Mistake[] = [
  {
    id: 'pca-forgetting-centering',
    pattern: 'Forgetting to center the data — using the raw second-moment matrix (1/n)XᵀX instead of the covariance (1/n)X_cᵀX_c',
    example: '\\frac{1}{n}X^T X = \\Sigma + \\mu\\mu^T \\;\\neq\\; \\Sigma = \\frac{1}{n}X_c^T X_c',
    whyWrong: 'The raw matrix is Σ + μμᵀ, and the rank-1 offset μμᵀ has eigenvalue ‖μ‖²: on the default config μ = (2.512, 1.617) so ‖μ‖² ≈ 8.93 — four times the total data variance (2.216). The dominant eigenvector of the raw matrix therefore points AT THE MEAN (default-config raw PC1 measured at 37.98°, mean direction 32.77°) instead of at the variance structure: on the rotDeg-80 config the uncentered PC1 sits at 33.30° while the correct centered PC1 is at 127.15°. The uncentered PC1 explains only 12.6% of the CENTERED variance — a rotationally catastrophic mistake that silently destroys the whole analysis.',
    gateTrap: true,
    relatedConcept: 'pca-covariance',
  },
  {
    id: 'pca-eigenvector-vs-pc',
    pattern: 'Confusing the eigenvectors (the directions) with the principal components (the projected scores)',
    example: 'v_k = \\text{direction (unit vector in feature space)} \\;\\neq\\; z_{ik} = v_k \\cdot (x_i - \\mu) = \\text{score of point } i',
    whyWrong: 'An eigenvector v_k is a UNIT DIRECTION in the original feature space — "the axis". The principal component of point i is the SCALAR score z_ik = v_k·(x_i − μ), its coordinate along that axis. On the default run v₁ = (0.235, 0.972) is the direction (angle 76.4°), while the scores are 40 numbers — one per point. Treating v₁ itself as "the projection of the data" gives a single vector instead of a whole projected dataset, and reconstructing from it is meaningless.',
    gateTrap: true,
    relatedConcept: 'pca-projection',
  },
  {
    id: 'pca-sorting-eigenvalues',
    pattern: 'Sorting the eigenvalues wrong — taking PC1 to be the eigenvector of the SMALLEST eigenvalue',
    example: '\\lambda_1 \\ge \\lambda_2 \\;\\Rightarrow\\; \\text{PC1} = v(\\lambda_1); \\quad \\text{swapping gives PC1} = v(\\lambda_2) = \\text{the MIN-variance direction}',
    whyWrong: 'PCA\'s entire premise is that PC1 carries the MOST variance. Sorting wrong inverts the explained-variance story: on the default run the true PC1 explains 87.4% of the variance, but the swapped choice (the λ₂ eigenvector, the thin direction) explains only 12.6% — the reconstruction error jumps from λ₂ = 0.2784 to λ₁ = 1.9380, a 7× loss. Because Σ is symmetric the frame is still orthogonal, so the error is silent: every diagram looks fine while the answer is inverted.',
    gateTrap: true,
    relatedConcept: 'pca-eigenequation',
  },
  {
    id: 'pca-is-supervised',
    pattern: 'Thinking PCA is supervised — that class labels guide the principal directions',
    example: '\\text{PCA: no } y_i \\text{ anywhere — directions maximize UNSUPERVISED variance} \\;\\neq\\; \\text{LDA: maximize BETWEEN-class separation}',
    whyWrong: 'PCA never looks at labels: the objective uᵀΣu is built from the centered data alone, and the default run\'s cloud is a single unlabeled Gaussian. The moment labels enter the covariance — e.g. by using LDA\'s between-class scatter — the directions change and the problem becomes supervised (LDA on the same data uses class means and shared covariance instead of total variance). On the pca topic the eigenviewer\'s single blue cloud (no class colors) is the visual tell.',
    gateTrap: true,
    relatedConcept: 'pca-covariance',
  },
];