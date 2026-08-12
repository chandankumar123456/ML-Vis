// src/topics/lda/formulas.ts
import type { Formula } from '../../engine/types';

// NOTE ON THE SCATTER CONVENTION (also documented in module.ts):
//   C_c = per-class sample covariance (normalized by n_c),
//   S_W := C_0 + C_1,  S_B := (μ₁−μ₂)(μ₁−μ₂)ᵀ  (Bishop's 2-class form).
// With this exact convention J(ŵ) = wᵀS_Bw / wᵀS_Ww IS the plan's test-2
// formula (μ̄₁−μ̄₂)²/(s₀²+s₁²) — one metric, no ambiguity. For balanced
// classes this equals the textbook raw-scatter J up to the constant n.

export const ldaFormulas: Formula[] = [
  {
    id: 'lda-between-scatter',
    latex: 'S_B = (\\mu_1 - \\mu_2)(\\mu_1 - \\mu_2)^T',
    symbols: [
      { symbol: '\\mu_c', meaning: 'class-c mean vector (the estimator used here: empirical class mean over the seeded points)', dimensions: 'feature units' },
      { symbol: 'S_B', meaning: 'between-class scatter matrix — the 2-class form: the outer product of the mean difference', dimensions: 'feature²' },
      { symbol: '(\\mu_1 - \\mu_2)(\\mu_1 - \\mu_2)^T', meaning: 'rank-1 outer product — ALL of S_B lives in ONE direction (the vector μ₁−μ₂)', dimensions: 'feature²' },
    ],
    assumptions: [
      'Two classes (the general Σ_c n_c(μ_c−μ)(μ_c−μ)ᵀ form reduces to a scaled version of this for exactly 2 classes — the scale does not change the argmax of J)',
      'Projecting onto w, the numerator of J is the SQUARED projected mean gap: wᵀS_Bw = (wᵀ(μ₁−μ₂))² ≥ 0',
    ],
    failureCases: [
      'Rank 1 by construction — with C classes, S_B has rank at most C−1, so LDA can find at most C−1 informative directions (a 2-class problem → exactly ONE)',
      'Overlapping means (μ₁ ≈ μ₂) make the between scatter tiny — the direction is then driven entirely by within-class structure',
    ],
    derivesFrom: [],
    connections: ['lda-within-scatter', 'lda-fisher-criterion', 'PCA covariance matrix'],
    whyWorks: 'S_B measures how far the class centers are from each other in a matrix sense. As an outer product of the mean difference it is rank one, which is exactly why Fisher LDA collapses a 2-class problem to a single projection direction — there is only one axis of "betweenness".',
  },
  {
    id: 'lda-within-scatter',
    latex: 'S_W = C_0 + C_1 \\\\ \\text{where } C_c = \\frac{1}{n_c}\\sum_{i\\in c} (x_i - \\mu_c)(x_i - \\mu_c)^T',
    symbols: [
      { symbol: 'C_c', meaning: 'per-class sample covariance (normalized by n_c — the convention of this topic)', dimensions: 'feature²' },
      { symbol: 'S_W', meaning: 'within-class scatter — the pooled per-class covariance sum', dimensions: 'feature²' },
      { symbol: 'n_c', meaning: 'number of samples in class c', dimensions: 'count' },
    ],
    assumptions: [
      'The classes share ONE covariance in the generative model (the LDA Gaussian assumption) — S_W pools them; QDA would keep them separate',
      'S_W must be INVERTIBLE: it needs enough full-rank spread per class (≥ 3 non-collinear samples per class in 2D)',
    ],
    failureCases: [
      'Singular S_W: few samples per class (2 points per class give a rank-1 covariance) or collinear classes — then S_W⁻¹ does not exist and LDA has no closed form (the failure demo)',
      'Outliers inflate C_c: one far point adds a large outer-product term and tilts S_W⁻¹(μ₁−μ₂) (the outliers failure demo)',
    ],
    derivesFrom: [],
    connections: ['lda-between-scatter', 'lda-fisher-criterion', 'lda-solution'],
    whyWorks: 'S_W aggregates how spread the points are around their own means. Along a direction w, wᵀS_Ww = s₀² + s₁² is exactly the sum of the projected within-class variances — the "compactness" half of the Fisher ratio. Inverting it whitens the within-class geometry before measuring the mean gap.',
  },
  {
    id: 'lda-fisher-criterion',
    latex: 'J(w) = \\frac{w^T S_B w}{w^T S_W w} = \\frac{(\\bar\\mu_1 - \\bar\\mu_0)^2}{s_0^2 + s_1^2}',
    symbols: [
      { symbol: '\\bar\\mu_c = w^T \\mu_c', meaning: 'projected class-c mean onto the candidate direction w', dimensions: 'feature units' },
      { symbol: 's_c^2 = w^T C_c w', meaning: 'projected within-class variance of class c along w', dimensions: 'feature²' },
      { symbol: 'J(w)', meaning: 'Fisher criterion — squared between-class gap over total within-class variance; scale-invariant in w', dimensions: 'dimensionless' },
    ],
    assumptions: [
      'Any nonzero scale of w leaves J unchanged (J(αw) = J(w)) — only the DIRECTION matters, which is why the axis sweep over [0, π) covers all distinct candidates',
      'The equality of the two forms is EXACT under this topic\'s convention (C_c normalized) — asserted numerically in the tests',
    ],
    failureCases: [
      'J is unbounded as s₀²+s₁² → 0 (fully collapsed classes): the ratio is numerically unstable on degenerate draws',
      'J measures separation ONLY along w — two directions with the same J are indistinguishable to the criterion though their 2-D boundaries differ',
    ],
    derivesFrom: ['lda-between-scatter', 'lda-within-scatter'],
    connections: ['lda-solution', 'lda-eigenproblem', 'lda-threshold', 'PCA variance ratio'],
    whyWorks: 'A good projection should push the projected class means far apart (numerator) while keeping each class tight (denominator). The Rayleigh-quotient form makes this a matrix problem: maximizing J over directions is the generalized eigenproblem S_Bw = λS_Ww — the same mathematics as PCA\'s variance ratio, but supervised (means, not pooled variance, define the goal).',
  },
  {
    id: 'lda-solution',
    latex: 'w = S_W^{-1}(\\mu_1 - \\mu_2)',
    symbols: [
      { symbol: 'S_W^{-1}', meaning: 'inverse within-class scatter (whitening the within-class geometry)', dimensions: '1/feature²' },
      { symbol: '\\mu_1 - \\mu_2', meaning: 'class-mean difference (the betweenness vector)', dimensions: 'feature units' },
      { symbol: 'w', meaning: 'the optimal (un-normalized) projection direction for 2 classes', dimensions: '1/feature' },
    ],
    assumptions: [
      'S_W invertible (full-rank within-class scatter)',
      'Exactly 2 classes — for C > 2 the analogue is the top eigenvectors of S_W⁻¹S_B',
    ],
    failureCases: [
      'Singular S_W → no solution (the failure demo; the topic fails honestly instead of emitting NaN)',
      'The direction depends on the EMPIRICAL covariance: a noisy estimate tilts w away from the true optimum (small-n brittleness)',
    ],
    derivesFrom: ['lda-fisher-criterion', 'lda-eigenproblem'],
    connections: ['lda-threshold', 'lda-within-scatter', 'linear discriminant function'],
    whyWorks: 'For rank-1 S_B, maximizing the Rayleigh quotient J(w) = wᵀS_Bw/wᵀS_Ww yields w ∝ S_W⁻¹(μ₁−μ₂): you first whiten the data with S_W (making the within-class spread isotropic), then read off the mean difference. The scale of w is irrelevant (J is scale-invariant), so the unit direction ŵ = w/‖w‖ is what the simulations draw and the classifier thresholds.',
  },
  {
    id: 'lda-threshold',
    latex: 'z = \\hat{w}^T x, \\qquad \\hat{y} = \\begin{cases} 1 & z > \\tau \\\\ 0 & z \\le \\tau \\end{cases} \\qquad \\tau = \\hat{w}^T \\frac{\\mu_0 + \\mu_1}{2}',
    symbols: [
      { symbol: 'z', meaning: 'the 1-D projected coordinate (the dimensionality reduction)', dimensions: 'feature units' },
      { symbol: '\\tau', meaning: 'the decision threshold on the projection axis — midpoint of the projected class means (equal priors, shared covariance)', dimensions: 'feature units' },
      { symbol: '\\hat{y}', meaning: 'the predicted class index (0/1) under the threshold rule', dimensions: 'class label' },
      { symbol: '\\hat{w}', meaning: 'unit LDA direction, oriented so ŵᵀ(μ₁−μ₂) ≥ 0 (class 1 projects high)', dimensions: 'dimensionless' },
    ],
    assumptions: [
      'Equal class priors AND shared covariance: with those, the Bayes-optimal boundary sits at the midpoint of the projected means; unequal priors shift τ by ln(P₁/P₀) in the log-odds',
      'The affine classifier used by the decision-boundary view is exactly w·x + b > 0 with b = −τ',
    ],
    failureCases: [
      'Non-Gaussian / multimodal classes: the threshold is still a single cut, but the Bayes error can be far larger than a density-aware rule (the multimodal failure demo)',
      'Overlapping classes have irreducible error at ANY threshold — LDA cannot perfectly separate them (a threshold, not a magic line)',
    ],
    derivesFrom: ['lda-solution', 'lda-fisher-criterion'],
    connections: ['logistic-regression decision rule', 'lda-projection', 'Bayes decision rule'],
    whyWorks: 'With one shared Gaussian per class, the log-posterior-ratio log P(C₁|x)/P(C₀|x) is a LINEAR function of x (the covariance terms cancel), so the decision boundary is a line perpendicular to ŵ. Projecting to the scalar z = ŵᵀx loses nothing for classification — 2-D data becomes a 1-D threshold problem, which is exactly "LDA as dimensionality reduction + classifier in one step".',
  },
  {
    id: 'lda-eigenproblem',
    latex: 'S_B w = \\lambda S_W w \\;\\Longleftrightarrow\\; S_W^{-1} S_B w = \\lambda w',
    symbols: [
      { symbol: '\\lambda', meaning: 'generalized eigenvalue — how much discrimination the eigen-direction carries', dimensions: '1/feature² times feature² = dimensionless' },
      { symbol: 'S_W^{-1} S_B', meaning: 'the discrimination matrix; its eigenvectors are the LDA directions', dimensions: 'dimensionless' },
      { symbol: 'w', meaning: 'generalized eigenvector (LDA direction)', dimensions: '1/feature' },
    ],
    assumptions: [
      'S_W invertible (otherwise one uses a pseudo-inverse or regularization)',
      'The top-k eigenvectors of S_W⁻¹S_B give the k-dimensional supervised subspace (LDA as dimensionality reduction)',
    ],
    failureCases: [
      'Rank of S_W⁻¹S_B ≤ C−1: with 2 classes only ONE nonzero eigenvalue exists — there is no second informative axis, no matter how many features',
      'When S_W is near-singular the eigenproblem is numerically violent: tiny covariance changes flip the directions',
    ],
    derivesFrom: ['lda-fisher-criterion'],
    connections: ['lda-solution', 'PCA eigen-decomposition', 'lda-between-scatter'],
    whyWorks: 'Maximizing the Rayleigh quotient wᵀS_Bw/wᵀS_Ww is the generalized eigenproblem S_Bw = λS_Ww (the same Lagrange-multiplier path as PCA, but with S_B instead of the pooled covariance Σ). For 2 classes the rank-1 S_B leaves exactly one nonzero eigenvalue λ = dᵀS_W⁻¹d = J(w*) — the tests verify this identity to machine precision, which is why the plots can report "λ = J(θ*)" without extra machinery.',
  },
  {
    id: 'lda-gaussian-assumption',
    latex: 'P(x \\mid C) = \\frac{1}{2\\pi \\sqrt{|\\Sigma|}} \\exp\\!\\left(-\\tfrac12 (x-\\mu_c)^T \\Sigma^{-1} (x-\\mu_c)\\right), \\quad \\Sigma \\text{ SHARED}',
    symbols: [
      { symbol: '\\mu_c', meaning: 'class-c Gaussian mean', dimensions: 'feature units' },
      { symbol: '\\Sigma', meaning: 'the class-SHARED covariance matrix (the defining LDA assumption; QDA relaxes it)', dimensions: 'feature²' },
      { symbol: 'P(x \\mid C)', meaning: 'class-conditional density — the generative model behind the discriminant', dimensions: '1/feature²' },
    ],
    assumptions: [
      'Both classes are Gaussian AND share one covariance — this is what makes the decision boundary LINEAR (the quadratic terms cancel)',
      'The topic\'s data generator actually draws from this model (rotated, elongated shared ellipse), so the assumption holds in the simulator',
    ],
    failureCases: [
      'Heteroscedastic classes (different Σ_C): LDA\'s single shared Σ is wrong — the boundary should be quadratic (QDA) (the shared-covariance failure demo)',
      'Multimodal or heavy-tailed classes: a single mean + variance cannot represent them (the multimodal failure demo)',
    ],
    derivesFrom: [],
    connections: ['lda-threshold', 'lda-within-scatter', 'gaussian-distribution', 'naive-bayes'],
    whyWorks: 'LDA is a generative classifier: it assumes the data came from Gaussian blobs with a common shape. Under that assumption the log-posterior is linear in x, giving the threshold rule — but the assumption is exactly why LDA fails on non-Gaussian or heteroscedastic data. The simulation\'s shared-covariance sliders let you watch the boundary tilt as Σ rotates.',
  },
  {
    id: 'lda-projection',
    latex: 'z_i = \\hat{w}^T x_i \\in \\mathbb{R} \\qquad \\text{(2-D data } \\to \\text{ 1-D coordinates)}',
    symbols: [
      { symbol: 'z_i', meaning: 'the 1-D coordinate of point i after projection onto the LDA axis', dimensions: 'feature units' },
      { symbol: '\\hat{w}', meaning: 'unit projection direction', dimensions: 'dimensionless' },
      { symbol: 'x_i', meaning: 'the 2-D feature vector of point i', dimensions: 'feature units' },
    ],
    assumptions: [
      'Orthogonal (perpendicular) projection onto the axis through the data center — the projection is idempotent: projecting twice gives the same point (verified: residual-to-axis = 0 in the tests)',
      'The projected within-class variance wᵀC_cw equals the sample variance of the projected class points — the variance-of-projection = projection-of-variance identity (asserted on the toy set)',
    ],
    failureCases: [
      'In high dimension the projection loses whatever the single direction cannot capture (LDA keeps at most C−1 axes)',
      'If the classes are not linearly separable in the original space, NO 1-D projection fully separates them',
    ],
    derivesFrom: ['lda-fisher-criterion'],
    connections: ['PCA projection', 'lda-threshold', 'dimensionality reduction'],
    whyWorks: 'The whole Fisher machinery reduces each point to one number — its position on the optimal axis. The simulation draws this literally: guide lines drop every point onto the axis, the class means and the threshold τ live on the same line, and classification is a single comparison z > τ. This is the "LDA as dimensionality reduction" face of the same model.',
  },
];