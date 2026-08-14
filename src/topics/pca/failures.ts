// src/topics/pca/failures.ts
// Every numeric claim below is measured against the module:
//   - F1 high noise (noise 1.5, corr 0.2, rotDeg 30): λ₁ = 3.4636, λ₂ = 3.2206,
//     ratio₁ = 0.518 — the Rayleigh quotient is nearly flat (isotropic) and the
//     PC angle jumps 166.94° → 172.39° between adjacent seeds 42 and 43 (5.5°).
//   - F2 outliers (points override with a lone [9,9] point): λ₁ = 15.9557,
//     ratio₁ = 0.9955, PC1 = (0.707, 0.707) at 45.01° — the outlier alone sets
//     the direction (pointing at it); the 8 inlier points carry 0.4% of the weight.
//   - F3 unscaled features (points with y = 10·x): λ₁ = 31.3661, λ₂ = 0,
//     ratio₁ = 1, PC1 = (0.0995, 0.9950) at 84.29° — PC1 IS the y-axis; the
//     x feature\'s structure is invisible.
import type { FailureDemo } from '../../engine/types';

export const pcaFailureDemos: FailureDemo[] = [
  {
    id: 'pca-fail-high-noise',
    title: 'High noise: the PC direction becomes unstable',
    scenario: 'high-noise',
    params: { n: 40, corr: 0.2, rotDeg: 30, noise: 1.5, seed: 42 },
    narration: 'With the noise dial at σ = 1.5 (ten times the default 0.15) and correlation 0.2, the isotropic noise σ²I adds 2.25 to BOTH eigenvalues: λ₁ = 3.4636 and λ₂ = 3.2206. The explained-ratio curve collapses to near-flat — PC1 explains only 51.8% of the variance (vs 87.4% at the default) — and the "principal" direction is almost arbitrary: adjacent seeds 42 and 43 report PC angles 166.94° and 172.39°, a 5.5° swing, while the eigenviewer\'s variance bars are nearly equal heights.',
    whyItBreaks: 'PCA\'s objective uᵀΣu is dominated by the isotropic noise term: when σ²I dwarfs the correlation structure, every direction has nearly the same Rayleigh quotient and the argmax is decided by tiny residual differences — the eigen-decomposition is exact but the ANSWER is meaningless (the plan\'s "PCs unstable under high noise"). The fix: denoise first, or interpret the explained-variance curve — a flat curve is the honest signal that no dominant direction exists.',
  },
  {
    id: 'pca-fail-outliers',
    title: 'A single outlier dominates the variance',
    scenario: 'outliers',
    params: {
      n: 40, corr: 0.7, rotDeg: 30, noise: 0.15, seed: 42,
      points: '[[0,0],[0.5,0.2],[-0.3,0.1],[0.2,-0.4],[-0.1,0.3],[0,0.1],[0.4,-0.2],[-0.4,0],[9,9]]',
    },
    narration: 'Eight points hug the origin while ONE outlier sits at (9, 9). The measured covariance is hijacked: λ₁ = 15.9557 (vs 1.9380 for a clean cloud) and PC1 = (0.707, 0.707) at 45.01° — pointing EXACTLY at the outlier. The clean 8-point structure is invisible: the outlier accounts for 99.55% of the total variance and the other axis carries only λ₂ = 0.0720.',
    whyItBreaks: 'Variance is a quadratic quantity — squared distances weight far points disproportionately. A single point at distance r contributes r² to the covariance, so one outlier at (9,9) (distance ≈ 9) outvotes eight points at distance ≈ 0.4 (distance² ≈ 0.16 each). The eigenvector of the largest eigenvalue simply points at the outlier. Robust fixes: trim/winsorize, use median-based or robust covariance estimators, or inspect the scree plot for the single dominant eigenvalue this failure produces.',
  },
  {
    id: 'pca-fail-unscaled',
    title: 'Unscaled features: the big-unit feature dominates PC1',
    scenario: 'unscaled-features',
    params: {
      n: 40, corr: 0.7, rotDeg: 30, noise: 0.15, seed: 42,
      points: '[[0,0],[1,10],[1.5,15],[0.5,5],[2,20],[0.8,8],[1.2,12],[0.3,3],[1.7,17],[0.9,9],[1.1,11],[0.6,6]]',
    },
    narration: 'The dataset is perfectly linear with y = 10·x — but the features are on different scales. The y-feature carries 100× the variance of x, so PC1 = (0.0995, 0.9950) at 84.29°: essentially the y-AXIS. The measured eigenvalues are λ₁ = 31.3661 and λ₂ = 0 — PC1 "explains" 100% of the variance, and the x-direction contributes nothing because its spread is invisible next to y\'s.',
    whyItBreaks: 'PCA is scale-sensitive: it maximizes variance in the ORIGINAL units, so the feature with the largest numeric range dominates. Here y\'s variance is 100× x\'s, so the "optimal" direction collapses onto the y-axis and the correlation structure between the features is lost. The fix is standardization (z-scoring each feature to unit variance) BEFORE PCA — then the demo\'s slope-10 line would be seen for what it is: a single strong direction through the origin of the scaled space.',
  },
];