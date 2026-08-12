// src/topics/lda/failures.ts
import type { FailureDemo } from '../../engine/types';

// Failure demos mirror the cluster convention: each demo carries its OWN params
// (deterministic) + narration + whyItBreaks. The first is a HONEST FAILURE via
// telemetry: its `points` are collinear, so computeLdaStats throws
// "within-class scatter S_W is singular..." and computeRun records
// failedAtStep (the svm-hard-margin non-separable precedent). The others run
// fine (J and the threshold are still computed) but show suboptimal boundaries.
export const ldaFailureDemos: FailureDemo[] = [
  {
    id: 'lda-fail-singular-sw',
    title: 'Singular S_W: collinear classes make S_W⁻¹(μ₁−μ₂) undefined',
    scenario: 'singular-within-scatter',
    // All six points lie on the line y = x: each per-class covariance is
    // rank-1 on (1,1)/√2, so S_W = C₀ + C₁ is rank-1 (det = 0).
    params: { points: '[[0,0,0],[1,1,0],[2,2,0],[3,3,1],[4,4,1],[5,5,1]]', seed: 42 },
    narration:
      'Every point sits on the line y = x: class 0 occupies (0,0)–(2,2), class 1 occupies (3,3)–(5,5). ' +
      'The per-class covariance of a collinear set is rank-1, so S_W = C₀ + C₁ has determinant 0 and no inverse. ' +
      'The closed form w = S_W⁻¹(μ₁−μ₂) does not exist — LDA has no solution for this data. The simulator surfaces this as an honest run failure (telemetry.failedAtStep) rather than emitting NaN.',
    whyItBreaks:
      'LDA needs a full-rank within-class scatter to whiten the geometry. Few samples per class (n_c < 3) or perfectly collinear classes collapse C_c to rank ≤ 1, making S_W singular. ' +
      'The fix: more samples, decorrelated features, or regularized LDA (add εI to S_W — the pseudo-inverse / shrinkage route). This is the plan\'s "few samples per class" failure in its most extreme form.',
  },
  {
    id: 'lda-fail-multimodal',
    title: 'Non-Gaussian / multimodal classes: one mean cannot represent two modes',
    scenario: 'non-gaussian-multimodal',
    params: {
      // Class 1 is TWO widely separated modes (left blob near class 0, right blob
      // far away): LDA collapses it to a single mean sitting between the modes,
      // so the linear threshold misplaces one of them. Class 0 is a compact blob.
      points: '[[0,0,0],[1,0,0],[0,1,0],[1,1,0],[-4,0,1],[-3,0,1],[-4,1,1],[-3,1,1],[5,0,1],[6,0,1],[5,1,1],[6,1,1]]',
      seed: 42,
    },
    narration:
      'Class 1 genuinely clusters around TWO centers — a left blob overlapping class 0 and a right blob far away. ' +
      'LDA fits ONE Gaussian per class, so its μ₁ is the midpoint between the modes and its pooled S_W smears both modes\' spread. ' +
      'The resulting threshold cuts one of the two class-1 modes into the wrong region: a single linear boundary cannot track a bimodal density.',
    whyItBreaks:
      'LDA assumes each class is unimodal Gaussian (shared Σ). A bimodal class violates this: the mean is a poor summary and the covariance overstates the spread, so the Fisher direction and threshold are miscalibrated. ' +
      'The fix: a mixture/nearest-neighbour model, or split the class (a non-parametric boundary such as k-NN handles the two modes directly).',
  },
  {
    id: 'lda-fail-outlier',
    title: 'Outliers in covariance estimation: one far point rotates S_W⁻¹(μ₁−μ₂)',
    scenario: 'outliers',
    params: {
      // Class 0 is a tight blob PLUS one distant outlier at (0,−6): the outlier
      // adds a huge outer-product term to C₀, stretching the within-class
      // covariance along y and tilting the LDA axis away from the true x-gap.
      points: '[[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-6,0],[4,0,1],[5,0,1],[4,1,1],[3,0,1],[4,-1,1]]',
      seed: 42,
    },
    narration:
      'Class 0 is a tight blob except for ONE point dragging it to (0,−6). That single outlier adds a squared deviation of 36 to C₀\'s y-entry, ' +
      'so S_W is dominated by a direction (y) that has nothing to do with the class signal. The whitened direction S_W⁻¹(μ₁−μ₂) and the threshold follow the outlier\'s pull, ' +
      'materially worsening the boundary the 9 well-behaved points justify.',
    whyItBreaks:
      'LDA\'s statistics — means and (especially) covariances — are least-squares quantities: one extreme point contributes quadratically. ' +
      'Covariance estimation is outlier-driven, and the whole closed form runs through S_W⁻¹. ' +
      'The fix: robust covariance estimation (M-estimators, trimming), or detect and remove the outlier before fitting.',
  },
  {
    id: 'lda-fail-heteroscedastic',
    title: 'Shared-covariance violation: unequal class covariances (QDA territory)',
    scenario: 'heteroscedastic-classes',
    params: {
      // Class 0 is tight (C₀ ≈ 0.25·I), class 1 is wide (C₁ ≈ 2·I). One shared
      // Σ (here the pooled average) is wrong for both: the linear threshold
      // misplaces the boundary the quadratic Bayes rule would choose.
      points: '[[0,0,0],[0.5,0,0],[-0.5,0,0],[0,0.5,0],[0,-0.5,0],[4,-2,1],[6,-2,1],[4,2,1],[6,2,1],[5,-2,1],[5,2,1],[4,0,1],[6,0,1],[2,0,1]]',
      seed: 42,
    },
    narration:
      'Class 0 is compact (spread ≈ 0.25 per axis) while class 1 is wide (spread ≈ 2 per axis) — the shared-covariance assumption of LDA is violated by construction. ' +
      'With unequal Σ_C the exact Bayes boundary between two Gaussians is a QUADRATIC curve, not a line; a single pooled threshold approximates it crudely, ' +
      'leaning toward the wide class\'s density where class 0 should have won.',
    whyItBreaks:
      'LDA\'s linearity derives from the shared covariance: when Σ₀ ≠ Σ₁ the quadratic terms in the log-posterior do not cancel. ' +
      'The single S_W is a compromise covariance that is exact for neither class. ' +
      'The fix: Quadratic Discriminant Analysis (per-class Σ) — the natural next topic for this exact scenario.',
  },
];