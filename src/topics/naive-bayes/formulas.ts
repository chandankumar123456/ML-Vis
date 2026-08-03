// src/topics/naive-bayes/formulas.ts
import type { Formula } from '../../engine/types';

export const nbFormulas: Formula[] = [
  {
    id: 'bayes',
    latex: 'P(C \\mid x) = \\frac{P(x \\mid C)\\, P(C)}{P(x)}',
    symbols: [
      { symbol: 'P(C|x)', meaning: 'posterior — probability of class C given the features x', units: 'probability' },
      { symbol: 'P(x|C)', meaning: 'class-conditional likelihood of the feature vector', units: 'probability' },
      { symbol: 'P(C)', meaning: 'prior probability of class C (empirically n_C/N)', units: 'probability' },
      { symbol: 'P(x)', meaning: 'evidence / marginal likelihood (normalization constant)', units: 'probability' },
    ],
    assumptions: ['There is a finite set of mutually exclusive classes'],
    derivationIds: ['nb-posterior', 'nb-logspace'],
    failureCases: [
      'P(x) = 0 when the feature vector has zero probability under every class — posterior undefined (the zero-probability story)',
      'Underflow: the numerator P(x|C)P(C) may underflow to 0 for many features',
    ],
    connections: ['prior', 'naive-likelihood', 'posterior'],
    whyWorks: 'Bayes theorem rewrites the target P(C|x) — what we want to predict — in terms of P(x|C), P(C) and P(x), all of which can be estimated from training data.',
  },
  {
    id: 'prior',
    latex: 'P(C) = \\frac{n_C}{N}',
    symbols: [
      { symbol: 'n_C', meaning: 'number of training samples with label C', units: 'count' },
      { symbol: 'N', meaning: 'total number of training samples', units: 'count' },
    ],
    assumptions: ['The training set is representative of the class distribution at prediction time'],
    failureCases: ['Class imbalance: ignoring the prior over-predicts the frequent class, so the Bayes-optimal decision is lost'],
    connections: ['bayes', 'log-posterior'],
    whyWorks: 'The empirical frequency n_C/N is the maximum-likelihood estimate of the categorical prior.',
  },
  {
    id: 'naive-likelihood',
    latex: 'P(x \\mid C) = \\prod_{j=1}^{d} P(x_j \\mid C)',
    symbols: [
      { symbol: 'x_j', meaning: 'the j-th feature of the sample', units: 'feature value' },
      { symbol: 'd', meaning: 'number of features (2 in this simulator)', units: 'count' },
    ],
    assumptions: [
      'Conditional independence: given the class, every pair of features is independent (the naive assumption)',
    ],
    derivationIds: ['nb-posterior'],
    failureCases: [
      'Correlated features: the product double-counts shared evidence, inflating the posterior of the wrong class',
      'A single zero factor (an unseen value without smoothing) annihilates the whole product',
    ],
    connections: ['gaussian-likelihood', 'laplace-smoothing', 'posterior'],
    whyWorks: 'Conditional independence lets a d-dimensional likelihood factor into d univariate ones, each cheap to estimate from per-feature statistics.',
  },
  {
    id: 'posterior',
    latex: 'P(C \\mid x) \\propto P(C)\\, \\prod_{j=1}^{d} P(x_j \\mid C)',
    symbols: [
      { symbol: 'P(C|x)', meaning: 'posterior (unnormalized here)', units: 'probability' },
      { symbol: 'P(C)', meaning: 'class prior', units: 'probability' },
    ],
    assumptions: ['Same as naive-likelihood (independence); the evidence P(x) is dropped because it does not depend on C'],
    failureCases: [
      'Forgetting to normalize if actual probabilities (not just the argmax) are needed',
      'A zero likelihood factor removes the class entirely (no smoothing)',
    ],
    connections: ['bayes', 'naive-likelihood', 'log-posterior'],
    whyWorks: 'For choosing the best class only relative posterior values matter, and P(x) is class-independent — so it cancels.',
  },
  {
    id: 'gaussian-likelihood',
    latex: 'P(x_j \\mid C) = \\frac{1}{\\sqrt{2\\pi\\sigma_{jC}^2}}\\;\\exp\\!\\left(-\\frac{(x_j-\\mu_{jC})^2}{2\\sigma_{jC}^2}\\right)',
    symbols: [
      { symbol: 'μ_{jC}', meaning: 'mean of feature j in class C (sample mean)', units: 'feature units' },
      { symbol: 'σ²_{jC}', meaning: 'variance of feature j in class C (sample variance + smoothing floor)', units: 'feature units squared' },
    ],
    assumptions: ['Within each class each feature is (approximately) normally distributed'],
    failureCases: [
      'Zero variance (all identical values) → the density is degenerate; the simulator floors σ² at α = smoothing',
      'Heavy-tailed or bimodal features fit poorly under a Gaussian assumption',
    ],
    connections: ['naive-likelihood', 'variance-floor'],
    whyWorks: 'The Gaussian density gives a smooth, everywhere-positive likelihood that is cheap to fit (two statistics per class-feature) and compute in log space.',
  },
  {
    id: 'log-posterior',
    latex: '\\log P(C \\mid x) = \\underbrace{\\log P(C) + \\sum_{j}\\Big[\\!-\\tfrac{1}{2}\\log(2\\pi\\sigma_{jC}^2) - \\frac{(x_j-\\mu_{jC})^2}{2\\sigma_{jC}^2}\\Big]}_{\\ell_C} - \\log P(x)',
    symbols: [
      { symbol: 'ℓ_C', meaning: 'log-likelihood + log-prior for class C (no normalization)', units: 'nats' },
      { symbol: 'P(x)', meaning: 'evidence; recovered via logsumexp for normalization', units: 'probability' },
    ],
    assumptions: ['Same as gaussian-likelihood + independence'],
    failureCases: ['Computing exp(ℓ_C) naively underflows for far-away points; the raw product P(x|C) can be 0 while ℓ_C is a perfectly usable finite number'],
    connections: ['bayes', 'posterior', 'gaussian-likelihood'],
    whyWorks: 'A product of many small probabilities becomes a sum of moderate negatives in log space — no underflow, and the argmax is unchanged because log is monotone.',
  },
  {
    id: 'laplace-smoothing',
    latex: 'P(x_j = v \\mid C) = \\frac{n_{C,v} + \\alpha}{n_C + \\alpha\\, V}',
    symbols: [
      { symbol: 'n_{C,v}', meaning: 'training count of value v for feature j in class C', units: 'count' },
      { symbol: 'n_C', meaning: 'training count of class C', units: 'count' },
      { symbol: 'α', meaning: 'Laplace smoothing parameter (0 = none)', units: 'pseudo-count' },
      { symbol: 'V', meaning: 'number of distinct values of the feature (vocabulary size)', units: 'count' },
    ],
    assumptions: ['Features are categorical (this formula is the discrete-mode model)'],
    failureCases: [
      'α = 0 with an unseen value: numerator 0 → the likelihood is exactly 0 and the class is dropped',
      'α too large: the data is washed out and the likelihoods flatten toward the uniform',
    ],
    connections: ['naive-likelihood', 'variance-floor'],
    whyWorks: 'Adding α pseudo-counts to every value keeps every likelihood strictly positive, so an unseen value costs the class a factor ε = α/(n_C + αV) instead of killing it.',
  },
  {
    id: 'variance-floor',
    latex: '\\tilde{\\sigma}_{jC}^2 = \\hat{\\sigma}_{jC}^2 + \\alpha',
    symbols: [
      { symbol: 'σ̂²_{jC}', meaning: 'raw sample variance of feature j in class C', units: 'feature units squared' },
      { symbol: 'α', meaning: 'smoothing / variance floor (the same slider as Laplace α)', units: 'feature units squared' },
    ],
    assumptions: ['Continuous Gaussian NB is the active model (smoothing = floor, not additive count)'],
    failureCases: ['α = 0 with a degenerate class (zero spread) → division by zero in the density'],
    connections: ['gaussian-likelihood', 'laplace-smoothing'],
    whyWorks: 'Adding α to every variance guarantees strictly positive variances, keeping the Gaussian density and log-posterior finite everywhere.',
  },
  {
    id: 'joint-gaussian',
    latex: 'P(x \\mid C) = \\frac{1}{2\\pi\\sqrt{|\\Sigma_C|}}\\;\\exp\\!\\left(-\\tfrac{1}{2}(x-\\mu_C)^{\\mathsf{T}}\\Sigma_C^{-1}(x-\\mu_C)\\right)',
    symbols: [
      { symbol: 'Σ_C', meaning: 'full within-class covariance matrix (includes the correlation the naive model ignores)', units: 'matrix' },
      { symbol: '|Σ_C|', meaning: 'determinant of the covariance', units: 'positive scalar' },
    ],
    assumptions: ['Features are jointly Gaussian within each class (the generative truth the simulator compares against)'],
    failureCases: ['A singular Σ_C (perfect collinearity) — the simulator floors the determinant at 1e-12 as a numerical guard'],
    connections: ['gaussian-likelihood', 'naive-likelihood'],
    whyWorks: 'The full covariance encodes how x₂ varies with x₁ within a class; using it is exactly "not assuming independence", so comparing this posterior with the naive one measures the cost of the naive assumption.',
  },
];
