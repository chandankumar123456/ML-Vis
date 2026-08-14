// src/topics/mle/comparisons.ts
// Comparisons per the plan: MLE vs MAP (prior term), MLE vs CE loss (same math,
// different lens), MLE vs OLS (MLE is the OLS instance under Gaussian noise).
// NOTE: no dedicated 'map' topic exists in this build yet — the Bayesian
// comparison anchors on 'naive-bayes' (the posterior-maximizing topic) and
// spells out the prior term in the axis values.
import type { Comparison } from '../../engine/types';

export const mleComparisons: Comparison[] = [
  {
    id: 'mle-vs-ols',
    title: 'MLE vs OLS (the same estimator under Gaussian noise)',
    topics: ['mle', 'simple-linear-regression'],
    axes: [
      {
        axis: 'Objective',
        entries: [
          { topic: 'mle', value: 'Maximize the Gaussian log-likelihood ℓ(θ) — maximizing −‖y−Xθ‖²/(2σ²) is minimizing the squared error' },
          { topic: 'simple-linear-regression', value: 'Minimize the residual sum of squares RSS = ‖y − Xθ‖² — no distributional story' },
        ],
      },
      {
        axis: 'Solution',
        entries: [
          { topic: 'mle', value: 'Score Xᵀ(y−Xθ) = 0 → θ̂ = (XᵀX)⁻¹Xᵀy, the normal equation (measured: θ̂ = (−0.467, 1.591) at n=100)' },
          { topic: 'simple-linear-regression', value: 'The identical normal equation — OLS and MLE agree to machine precision (score measured 8.4e−15)' },
        ],
      },
      {
        axis: 'Variance estimate',
        entries: [
          { topic: 'mle', value: 'σ̂² = RSS/n — the ÷n MLE, biased (measured 0.710527 vs unbiased 0.725028 at n=100)' },
          { topic: 'simple-linear-regression', value: 'Often the unbiased RSS/(n−2) — same data, different convention; the ÷n-vs-÷(n−1) gap is 1/(n−2) of the MLE' },
        ],
      },
      {
        axis: 'Generality',
        entries: [
          { topic: 'mle', value: 'Any parametric family (Bernoulli, Gaussian, exponential, Poisson…) — OLS is only its Gaussian-noise instance' },
          { topic: 'simple-linear-regression', value: 'Least squares works for ANY error distribution, but is only the MLE when errors are Gaussian' },
        ],
      },
    ],
    notes: [
      'The plan\'s "MLE = OLS for Gaussian noise" is the theorem; the testCases verify it numerically on the normal equation.',
      'The difference is interpretive, not computational: MLE carries a probabilistic model (and a variance estimator), OLS is pure geometry.',
    ],
  },
  {
    id: 'mle-vs-ce',
    title: 'MLE vs Cross-Entropy Loss (same math, different lens)',
    topics: ['mle', 'cross-entropy-loss'],
    axes: [
      {
        axis: 'Definition',
        entries: [
          { topic: 'mle', value: 'ℓ(θ) = Σ ln p(xᵢ|θ) — maximize the joint log-probability of the data under the model' },
          { topic: 'cross-entropy-loss', value: 'CE = −(1/n)Σ ln p̂ᵢ — average negative log-likelihood of the fitted predictive distribution (a LOSS to minimize)' },
        ],
      },
      {
        axis: 'Direction',
        entries: [
          { topic: 'mle', value: 'Maximize ℓ; the score S(θ) = ∂ℓ/∂θ is set to zero' },
          { topic: 'cross-entropy-loss', value: 'Minimize CE; gradient descent steps against ∂CE/∂θ = −(1/n)·S(θ)' },
        ],
      },
      {
        axis: 'Measured link',
        entries: [
          { topic: 'mle', value: 'The module plots nllPerSample = −ℓ(θ̂)/n = 0.6048 at n=1000, descending toward H(0.7) = 0.610864' },
          { topic: 'cross-entropy-loss', value: 'That same nllPerSample IS the empirical cross-entropy — the loss-curve metric both topics share' },
        ],
      },
      {
        axis: 'Floor',
        entries: [
          { topic: 'mle', value: 'As n grows the MLE nll approaches the true entropy of the data-generating distribution (0.610864 for the coin)' },
          { topic: 'cross-entropy-loss', value: 'The irreducible loss is the true entropy; CE cannot go below it, which is the deep link to KL divergence' },
        ],
      },
    ],
    notes: [
      'Cross-entropy is negative log-likelihood averaged per sample — the same objective with opposite sign and a per-sample normalization.',
      'The MLE topic\'s loss-curve view (nllPerSample) and cross-entropy-loss topic plot the same quantity from the two directions.',
    ],
  },
  {
    id: 'mle-vs-map',
    title: 'MLE vs MAP (the prior term — Bayesian estimation)',
    topics: ['mle', 'naive-bayes'],
    axes: [
      {
        axis: 'Objective',
        entries: [
          { topic: 'mle', value: 'θ̂ = argmax L(θ) — the data alone decides the parameter (no prior)' },
          { topic: 'naive-bayes', value: 'MAP maximizes the POSTERIOR: p(θ|x) ∝ L(θ)·p(θ) — a prior term is multiplied in (naive-bayes uses Bayes\' rule with class priors)' },
        ],
      },
      {
        axis: 'Bias trade',
        entries: [
          { topic: 'mle', value: 'Unbiased in the limit (consistent) but high variance at small n — p̂ = 0.8 from n=10 flips' },
          { topic: 'naive-bayes', value: 'The prior shrinks estimates toward it (regularization): less variance at small n, at the cost of bias when the prior is wrong' },
        ],
      },
      {
        axis: 'Small-n behaviour',
        entries: [
          { topic: 'mle', value: 'k = 0 or k = n gives p̂ = 0 or 1 — degenerate; the module clamps the log domain (P_LOG_FLOOR = 1e−12)' },
          { topic: 'naive-bayes', value: 'A Beta/Laplace prior keeps the estimate away from 0/1 (add-one smoothing) — the classic fix the MLE lacks' },
        ],
      },
      {
        axis: 'Asymptotics',
        entries: [
          { topic: 'mle', value: 'The prior term vanishes as n grows: MLE and MAP coincide in the limit (both → the truth)' },
          { topic: 'naive-bayes', value: 'The posterior concentrates on the MLE as n → ∞ — the prior becomes negligible' },
        ],
      },
    ],
    notes: [
      'MAP = MLE + a prior (documented here because a standalone MAP topic is not in this build — naive-bayes is the closest Bayesian anchor).',
      'The plan\'s phrasing "MLE vs MAP (prior term)" is exactly the L(θ) vs L(θ)·p(θ) difference this comparison tables.',
    ],
  },
];