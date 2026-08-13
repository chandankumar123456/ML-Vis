// src/topics/lda/questions.ts
import type { Question } from '../../engine/types';

// The NAT questions use the module's TOY_POINTS dataset, whose scatter-matrix
// math is fully hand-derivable (printed in module.ts / testCases.ts):
//   μ₀ = (1,1), μ₁ = (5,2), d = (4,1),
//   C₀ = I, C₁ = [[1,1],[1,2]], S_W = [[2,1],[1,3]], det = 5,
//   w = S_W⁻¹d = (11/5, −2/5), J(w*) = dᵀS_W⁻¹d = 42/5 = 8.4,
//   ŵᵀμ₀ = 9/(5√5), ŵᵀμ₁ = 51/(5√5), τ = 6/√5.
// The testCases.test.ts file asserts these anchors against the module's own
// computeLdaStats(TOY_POINTS), so the questions' numbers are machine-verified.

export const ldaQuestions: Question[] = [
  {
    id: 'lda-001',
    mode: 'nat',
    prompt:
      'Two classes with means μ₀ = (1, 1) and μ₁ = (5, 2), within-class covariance sum S_W = [[2, 1],[1, 3]]. ' +
      'Enter the FIRST component w₁ of the LDA direction w = S_W⁻¹(μ₁ − μ₂). (Hint: det S_W = 5.)',
    answer: 22 / 10, // 11/5 = 2.2
    tolerance: 1e-9,
    explanation:
      'd = μ₁ − μ₂ = (4, 1). S_W⁻¹ = (1/5)·[[3, −1],[−1, 2]], so w = S_W⁻¹d = (1/5)(3·4 − 1·1, −1·4 + 2·1) = (11/5, −2/5). ' +
      'w₁ = 11/5 = 2.2, w₂ = −2/5. This is the exact direction the toy-set simulation computes (verified to 1e-9 in the tests).',
    concepts: ['lda', 'within-class scatter', 'matrix inverse'],
    difficulty: 3,
    tags: ['numerical', 'formula', 'matrix'],
  },
  {
    id: 'lda-002',
    mode: 'nat',
    prompt:
      'Continuing lda-001 (d = (4,1), S_W = [[2,1],[1,3]]): enter the Fisher criterion value J(w*) = dᵀS_W⁻¹d ' +
      'at the LDA optimum — the single nonzero eigenvalue of S_W⁻¹S_B.',
    answer: 42 / 5, // 8.4
    tolerance: 1e-9,
    explanation:
      'J(w*) = dᵀS_W⁻¹d = (4,1)·(11/5, −2/5) = 44/5 − 2/5 = 42/5 = 8.4. Because S_B is rank 1, this scalar IS the ' +
      'generalized eigenvalue λ — the "eigen-Link" identity the tests assert to machine precision.',
    concepts: ['lda', 'fisher criterion', 'generalized eigenvalue'],
    difficulty: 4,
    tags: ['numerical', 'formula', 'matrix'],
  },
  {
    id: 'lda-003',
    mode: 'gate-mcq',
    prompt: 'GATE-style: you have labeled data (two classes) and want to reduce 2-D features to 1-D for classification downstream. Which is true?',
    options: [
      'LDA uses the labels (maximizes between-class / within-class separation) and is the supervised choice; PCA ignores labels and is the unsupervised choice',
      'PCA is always better than LDA because it maximizes total variance',
      'LDA and PCA always give the same projection direction',
      'PCA uses class means while LDA uses the overall data covariance',
    ],
    answer: 'A',
    explanation:
      'LDA is SUPERVISED: its objective wᵀS_Bw/wᵀS_Ww is built from class means and per-class covariances — it needs the labels. ' +
      'PCA is UNSUPERVISED: it maximizes wᵀΣw (total variance) using only the pooled covariance, ignoring which class each point belongs to. ' +
      'In the simulation, the J(θ) sweep curve peaks at a different angle than a variance-maximizing (PCA-style) direction would: LDA\'s optimum is where the between/within RATIO is highest, not where the variance is — that is the story of the sweep.',
    trapExplanations: {
      B: 'LDA maximizes separation, not variance — with overlapping but separable classes the variance-maximizing axis can destroy the class signal.',
      C: 'They coincide only in degenerate cases (e.g. class means aligned with the first eigenvector); in general the objectives differ (seen in the sweep).',
      D: 'It is the reverse: PCA uses the pooled covariance, LDA uses class means and per-class scatter.',
    },
    concepts: ['lda', 'pca', 'supervised learning', 'unsupervised learning'],
    difficulty: 2,
    tags: ['conceptual', 'indirect'],
  },
  {
    id: 'lda-004',
    mode: 'matrix',
    prompt: 'For the two-class LDA generalized eigenproblem S_B w = λ S_W w, which statement is correct?',
    options: [
      'S_B has rank 1, so there is exactly one nonzero eigenvalue — a 2-class problem yields ONE LDA projection direction, no matter how many features',
      'S_B has rank n−1 and there are n nonzero eigenvalues',
      'S_W⁻¹S_B is symmetric, so every feature contributes its own independent direction',
      'λ is a vector, one component per class mean',
    ],
    answer: 'A',
    explanation:
      'S_B = (μ₁−μ₂)(μ₁−μ₂)ᵀ is an outer product — rank exactly 1. ' +
      'Therefore S_W⁻¹S_B is also rank 1 and has a single nonzero eigenvalue λ = dᵀS_W⁻¹d = J(w*); all other eigenvalues are 0. ' +
      'This is why 2-class LDA collapses to one axis regardless of the feature dimension — the matrix question students get wrong by expecting one direction per feature.',
    trapExplanations: {
      B: 'Rank of S_B relates to C−1 (classes), not n−1 (samples).',
      C: 'S_W⁻¹S_B is generally NOT symmetric — and its rank is still 1 regardless.',
      D: 'Eigenvalues are scalars; a rank-1 matrix has one nonzero scalar eigenvalue, not a vector.',
    },
    concepts: ['lda', 'matrix algebra', 'eigenproblem'],
    difficulty: 3,
    tags: ['matrix'],
  },
  {
    id: 'lda-005',
    mode: 'visual',
    prompt: 'Scrub the LDA simulation through the 36-step angle sweep, then to the final closed-form step. What does the Fisher criterion J(θ) curve do?',
    options: [
      'It rises and falls over the sweep and the FINAL closed-form step lands at (or above) the grid maximum — the LDA direction maximizes J exactly',
      'It decreases monotonically, so the last grid angle is always best',
      'It is flat because J is a constant for all directions',
      'The final step is just the first grid angle repeated (θ = 0°)',
    ],
    answer: 'A',
    explanation:
      'J(θ) = (μ̄₁−μ̄₀)²/(s₀²+s₁²) peaks at the direction that best separates the projected means relative to within-class spread — ' +
      'the 37th (final) step evaluates the closed form w = S_W⁻¹(μ₁−μ₂) exactly, which provably maximizes J, so it sits at or above every grid point. ' +
      'The projected class-mean markers and the within-class variance values s₀², s₁² (reported as metrics on each sweep step) let you SEE why: bad axes smear the two classes together along the projection — s₀²+s₁², the denominator of J, blows up.',
    trapExplanations: {
      B: 'J is not monotone over the rotation — the sweep crosses a maximum and falls again.',
      C: 'J varies strongly with direction; that variation IS the point of the sweep.',
      D: 'The final step is the analytic optimum from S_W⁻¹(μ₁−μ₂), not a grid angle.',
    },
    concepts: ['lda', 'projection', 'fisher criterion'],
    difficulty: 2,
    tags: ['visual'],
  },
  {
    id: 'lda-006',
    mode: 'gate-mcq',
    prompt: 'GATE-style trap: a dataset has two well-separated Gaussian classes WITH DIFFERENT covariance matrices (one tight, one wide). A student applies Fisher LDA anyway. What breaks?',
    options: [
      'LDA assumes ONE shared covariance; the pooled S_W is wrong, the boundary it draws is suboptimal — QDA (per-class Σ) is the correct generative choice',
      'Nothing breaks — LDA is assumption-free',
      'The classes must be linearly separable first, or LDA fails catastrophically',
      'LDA requires more than 2 features',
    ],
    answer: 'A',
    explanation:
      'LDA\'s linearity comes from the shared-covariance assumption: with different Σ_C, the quadratic terms do NOT cancel and the true Bayes boundary is a conic section (QDA territory). ' +
      'With unequal spreads, LDA\'s single threshold misplaces the boundary between the tight and wide classes. ' +
      'The failure demos include this "heteroscedastic classes" scenario with hand-crafted covariances.',
    trapExplanations: {
      B: 'LDA is generative — it assumes Gaussians with shared covariance; violating that changes the optimal boundary.',
      C: 'Separability is not required for LDA to run (it still gives a threshold rule, just with Bayes error); the shared-Σ assumption is the real issue.',
      D: '2-class LDA works in any feature dimension ≥ 1.',
    },
    concepts: ['lda', 'qda', 'covariance assumption', 'gaussian'],
    difficulty: 3,
    tags: ['trap', 'conceptual'],
  },
  {
    id: 'lda-007',
    mode: 'conceptual-mcq',
    prompt: 'Why is the optimal LDA direction not simply the vector μ₁ − μ₂ (the class-mean difference)?',
    options: [
      'Within-class spread matters: directions with high variance inflate the denominator s₀²+s₁², so LDA whitens with S_W⁻¹ before measuring the mean gap',
      'μ₁ − μ₂ is always correct; S_W⁻¹ is a red herring',
      'Because the data must be centered at the origin first',
      'Because LDA maximizes variance, not separation',
    ],
    answer: 'A',
    explanation:
      'w = S_W⁻¹(μ₁−μ₂) first whitens the within-class geometry. In the simulation with an elongated rotated covariance, the mean-difference axis (pointing straight at the other class) cuts diagonally through the wide spread — ' +
      'projecting along it smears each class. The S_W⁻¹ term re-aims the direction to run along the classes\' tight axes. ' +
      'This is precisely the "forgetting S_W⁻¹" mistake.',
    trapExplanations: {
      B: 'S_W⁻¹ is exactly what makes LDA different from a naive mean-gap direction — it is the core of the method.',
      C: 'Centering only shifts the threshold; it does not rotate the optimal direction.',
      D: 'LDA maximizes the between/within RATIO, not raw variance (that is PCA).',
    },
    concepts: ['lda', 'whitening', 'within-class scatter'],
    difficulty: 3,
    tags: ['conceptual', 'formula'],
  },
  {
    id: 'lda-008',
    mode: 'conceptual-mcq',
    prompt: 'Two classes project onto the LDA axis with means ŵᵀμ₀ = 9/(5√5) and ŵᵀμ₁ = 51/(5√5) (equal priors, shared covariance). A test point projects to z = 20/(5√5). Which class does LDA predict?',
    options: [
      'Class 0 — its projection 20/(5√5) is below the threshold τ = 6/√5 = 30/(5√5)',
      'Class 1 — LDA always predicts the larger mean side',
      'Neither — 20/(5√5) is exactly on the boundary',
      'Class 1 for equal priors, class 0 for unequal priors',
    ],
    answer: 'A',
    explanation:
      'τ = (ŵᵀμ₀ + ŵᵀμ₁)/2 = (9 + 51)/(2·5√5) = 6/√5 = 30/(5√5). The point projects to 20/(5√5) < 30/(5√5), so ' +
      'ŵᵀx < τ → class 0. Note 20 ≠ 30: it is NOT on the boundary (trap option C). The threshold is a midpoint in projected coordinate, not an average of class counts.',
    trapExplanations: {
      B: 'LDA predicts by the threshold, not by "whichever mean is bigger in 2-D" — the 1-D comparison is decisive.',
      C: '20/(5√5) < 30/(5√5) — strictly below the boundary, so class 0.',
      D: 'The prior appears only in τ\'s derivation; the prediction is made by comparing z to τ regardless.',
    },
    concepts: ['lda', 'decision rule', 'threshold'],
    difficulty: 2,
    tags: ['numerical', 'indirect'],
  },
];