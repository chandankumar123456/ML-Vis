// src/topics/lasso-regression/formulas.ts
import type { Formula } from '../../engine/types';

export const lassoFormulas: Formula[] = [
  {
    id: 'lasso-objective',
    latex: 'J(\\tilde{\\theta}) = \\frac{1}{2n} \\sum_{i=1}^{n} \\left( y_i - \\tilde{b} - \\sum_{j=1}^{d} \\tilde{\\theta}_j z_{ij} \\right)^2 + \\lambda \\sum_{j=1}^{d} |\\tilde{\\theta}_j|',
    symbols: [
      { symbol: 'z_{ij}', meaning: 'standardized feature j of sample i (z-scored before fitting)', dimensions: 'unitless' },
      { symbol: '\\tilde{\\theta}_j', meaning: 'standardized weight for feature j', dimensions: 'y units per z-unit' },
      { symbol: '\\tilde{b}', meaning: 'intercept — NOT penalized', dimensions: 'output units' },
      { symbol: '\\lambda', meaning: 'regularization strength — larger λ ⇒ more zeros', dimensions: 'non-negative scalar' },
    ],
    assumptions: ['Features are z-scored (else the penalty weights each feature unequally)', 'n > d (well-conditioned regime)', 'λ ≥ 0'],
    failureCases: ['λ too large → every weight becomes 0 (underfit, ŷ = ȳ)', 'Nearly collinear features → arbitrary selection'],
    connections: ['MSE', 'Ridge objective'],
    whyWorks: 'MSE is convex and the L1 penalty is convex, so J is convex with a unique global minimum; the kink of |θ| at 0 is what produces exact zeros.',
  },
  {
    id: 'soft-threshold',
    latex: 'S(z, \\lambda) = \\operatorname{sign}(z) \\cdot \\max(|z| - \\lambda, \\, 0)',
    symbols: [
      { symbol: 'z', meaning: 'normalized correlation ρⱼ/n for coordinate j (the OLS coordinate solution)', dimensions: 'y per z-unit' },
      { symbol: '\\lambda', meaning: 'penalty strength — the amount subtracted from |z|', dimensions: 'same as z' },
      { symbol: 'S(z, \\lambda)', meaning: 'new standardized coefficient — EXACTLY 0 when |z| ≤ λ', dimensions: 'y per z-unit' },
    ],
    assumptions: ['z and λ are on the same scale (features standardized)'],
    failureCases: ['Thresholding the raw correlation ρ instead of ρ/n → wrong scale, wrong zeros'],
    derivesFrom: ['subgradient'],
    connections: ['Coordinate descent update'],
    whyWorks: 'S(z,λ) is the exact minimizer of the 1-D problem min_w ½(w − z)² + λ|w|: shift by λ, clamp at 0.',
  },
  {
    id: 'cd-update',
    latex: '\\tilde{\\theta}_j \\leftarrow S\\!\\left(\\frac{\\rho_j}{n},\\ \\lambda\\right), \\quad \\rho_j = \\sum_{i=1}^{n} z_{ij} \\left( y_i - \\tilde{b} - \\sum_{k \\neq j} \\tilde{\\theta}_k z_{ik} \\right)',
    symbols: [
      { symbol: '\\rho_j', meaning: 'correlation of feature j with the partial residual (all other coordinates fixed)', dimensions: 'n × y per z-unit' },
      { symbol: '\\rho_j / n', meaning: 'OLS coordinate solution z — the target before thresholding', dimensions: 'y per z-unit' },
      { symbol: 'S', meaning: 'soft-threshold operator — shrinks then possibly zeroes', dimensions: '—' },
    ],
    assumptions: ['Σᵢ zᵢⱼ² = n (z-scored features, population variance)'],
    failureCases: ['Updating the bias with the penalty — the intercept is unpenalized'],
    derivesFrom: ['soft-threshold'],
    connections: ['Lasso objective'],
    whyWorks: 'Coordinate descent minimizes J along one coordinate at a time; the exact coordinate minimizer of a quadratic plus λ|·| is the soft-thresholded OLS solution.',
  },
  {
    id: 'subgradient',
    latex: '\\partial |\\theta| = \\begin{cases} \\operatorname{sign}(\\theta) & \\theta \\neq 0 \\\\ [-1,\\ 1] & \\theta = 0 \\end{cases}',
    symbols: [
      { symbol: '\\partial |\\theta|', meaning: 'subgradient of the L1 penalty — a SET at the kink', dimensions: '—' },
      { symbol: '[-1,\\ 1]', meaning: 'at θ = 0 the subgradient is the whole interval between the left and right derivatives', dimensions: '—' },
    ],
    assumptions: ['None — subgradients generalize derivatives to non-differentiable points'],
    failureCases: ['Treating the L1 penalty as differentiable everywhere and setting |θ|′ = ±1 at θ = 0'],
    derivesFrom: [],
    connections: ['Soft-threshold', 'L1 vs L2 geometry'],
    whyWorks: 'A stationary point needs 0 ∈ ∂J. At θ = 0 the interval [−1,1] can contain −z + λ·s for a range of z — this is exactly the |z| ≤ λ region that produces exact zeros.',
  },
  {
    id: 'ridge-objective',
    latex: 'J_{\\text{ridge}}(\\theta) = \\frac{1}{2n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 + \\lambda \\sum_{j=1}^{d} \\theta_j^2',
    symbols: [
      { symbol: '\\theta_j^2', meaning: 'L2 penalty — smooth, differentiable everywhere', dimensions: 'squared' },
      { symbol: '\\lambda', meaning: 'ridge strength — shrinks proportionally, never to exact 0', dimensions: 'non-negative scalar' },
    ],
    assumptions: ['Same data and scaling conventions as lasso for a fair comparison'],
    failureCases: ['Claiming ridge produces exact zeros — its penalty is smooth, so θ = 0 is a measure-zero event'],
    derivesFrom: ['lasso-objective'],
    connections: ['L1 vs L2 geometry'],
    whyWorks: 'The L2 ball is a smooth circle: the constrained optimum touches it generically away from the axes, so every coordinate stays nonzero.',
  },
  {
    id: 'l1-geometry',
    latex: '\\min_\\theta \\|y - X\\theta\\|_2^2 \\quad \\text{s.t.} \\quad \\|\\theta\\|_1 \\le t \\qquad \\text{(vs. ridge: } \\|\\theta\\|_2 \\le t\\text{)}',
    symbols: [
      { symbol: '\\|\\theta\\|_1 \\le t', meaning: 'L1 constraint — a DIAMOND with vertices on the coordinate axes', dimensions: '—' },
      { symbol: '\\|\\theta\\|_2 \\le t', meaning: 'L2 constraint — a CIRCLE (smooth, no vertices)', dimensions: '—' },
      { symbol: 't', meaning: 'constraint radius — smaller t ⇒ stronger shrinkage', dimensions: 'non-negative scalar' },
    ],
    assumptions: ['The unconstrained optimum lies outside the feasible region (else the constraint is inactive)'],
    failureCases: ['Reading the classic figure backwards — the corner zeros belong to L1, not L2'],
    derivesFrom: ['ridge-objective'],
    connections: ['Coordinate descent update', 'Soft-threshold'],
    whyWorks: 'Level sets of the quadratic objective are ellipses. Touching a diamond at a VERTEX sets one coordinate to exactly 0; touching a circle on the smooth arc does not.',
  },
];
