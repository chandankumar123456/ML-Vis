// src/topics/ridge-regression/derivations.ts
import type { Derivation } from '../../engine/types';

export const ridgeDerivations: Derivation[] = [
  {
    id: 'ridge-closed-form',
    title: 'Deriving the Ridge Closed Form from the Penalized Objective',
    steps: [
      {
        latex: 'J(\\theta) = \\frac{1}{n}\\|y - X\\theta\\|_2^2 + \\lambda\\|\\theta\\|_2^2',
        justification: 'The ridge objective: MSE plus the squared-L2 penalty on all of θ (bias included — the standard GATE closed-form treatment).',
      },
      {
        latex: 'J(\\theta) = \\frac{1}{n}\\left( y^T y - 2\\theta^T X^T y + \\theta^T X^T X \\theta \\right) + \\lambda \\theta^T \\theta',
        justification: 'Expand the two quadratic terms: (y−Xθ)ᵀ(y−Xθ) and ‖θ‖² = θᵀθ.',
      },
      {
        latex: '\\nabla_\\theta J = \\frac{1}{n}\\left( -2X^T y + 2X^T X \\theta \\right) + 2\\lambda \\theta',
        justification: 'Matrix calculus: ∂(θᵀXᵀy)/∂θ = Xᵀy and ∂(θᵀXᵀXθ)/∂θ = 2XᵀXθ (XᵀX symmetric); the penalty contributes ∂(λθᵀθ)/∂θ = 2λθ — the extra term that OLS lacks.',
      },
      {
        latex: '0 = -\\frac{2}{n} X^T y + \\frac{2}{n} X^T X \\theta + 2\\lambda \\theta',
        justification: 'Set the gradient to zero. The objective is strictly convex (Hessian 2(XᵀX + λI)/n ≻ 0 for λ > 0), so this stationary point is the unique global minimum.',
      },
      {
        latex: '(X^T X + \\lambda I)\\,\\theta = X^T y',
        justification: 'Multiply through by n/2 and factor θ — the penalty adds λI to the Gram matrix.',
      },
      {
        latex: '\\theta = (X^T X + \\lambda I)^{-1} X^T y',
        justification: 'Left-multiply by (XᵀX + λI)⁻¹. This inverse ALWAYS exists for λ > 0, because every eigenvalue of XᵀX (≥ 0) is shifted to ≥ λ > 0.',
      },
    ],
    derivedFrom: ['ridge-closed-form'],
  },
  {
    id: 'ridge-constrained-equivalence',
    title: 'Ridge as Constrained Optimization (the L2 Circle)',
    steps: [
      {
        latex: '\\min_\\theta \\|y - X\\theta\\|_2^2 \\quad \\text{s.t.} \\quad \\|\\theta\\|_2 \\le t',
        justification: 'The constrained form: keep the fit inside an L2 ball of radius t. In 2D this is a circle centred at the origin.',
      },
      {
        latex: 'L(\\theta, \\lambda) = \\|y - X\\theta\\|_2^2 + \\lambda\\left(\\|\\theta\\|_2^2 - t\\right)',
        justification: 'Lagrangian with multiplier λ ≥ 0 (the constraint is an inequality, active at the solution for small t).',
      },
      {
        latex: '\\nabla_\\theta L = -2X^T y + 2X^T X \\theta + 2\\lambda \\theta = 0 \\;\\Rightarrow\\; \\theta = (X^T X + \\lambda I)^{-1} X^T y',
        justification: 'Stationarity of the Lagrangian reproduces EXACTLY the ridge closed form — λ is the Lagrange multiplier. Equivalent: the unconstrained ellipse optimum is pulled back until it just touches the L2 circle.',
      },
      {
        latex: '\\text{Ridge: } \\|\\theta\\|_2 \\le t \\;\\text{(circle)} \\qquad \\text{Lasso: } \\|\\theta\\|_1 \\le t \\;\\text{(diamond)}',
        justification: 'The geometric contrast that GATE loves: a circle has no corners, so the touch point lies on the smooth arc — no coefficient hits exactly 0. A diamond has corners at the axes, which is where lasso’s solutions land → exact zeros.',
      },
    ],
    derivedFrom: ['ridge-constrained'],
  },
  {
    id: 'ridge-bias-variance',
    title: 'Bias and Variance of the Ridge Estimator',
    steps: [
      {
        latex: '\\theta_{\\text{OLS}} = (X^T X)^{-1} X^T y',
        justification: 'Start from the unbiased OLS solution (assume XᵀX invertible).',
      },
      {
        latex: '\\theta_{\\lambda} = (X^T X + \\lambda I)^{-1} X^T y = R_\\lambda \\, \\theta_{\\text{OLS}}, \\qquad R_\\lambda = (X^T X + \\lambda I)^{-1} X^T X',
        justification: 'Rewrite the ridge solution as a linear map of the OLS solution. The map R_λ is symmetric with eigenvalues μₖ/(μₖ + λ) < 1.',
      },
      {
        latex: '\\text{Bias}(\\theta_\\lambda) = (R_\\lambda - I)\\,\\theta_{\\text{true}} = -\\lambda (X^T X + \\lambda I)^{-1} \\theta_{\\text{true}}',
        justification: 'E[θ_λ] = R_λ·θ_true (unbiased Xᵀy → mean XᵀXθ_true). Since R_λ ≠ I for λ > 0, ridge is biased. Verify: (XᵀX + λI)⁻¹XᵀX − I = (XᵀX + λI)⁻¹(XᵀX − (XᵀX + λI)) = −λ(XᵀX + λI)⁻¹.',
      },
      {
        latex: '\\text{Var}(\\theta_\\lambda) = \\sigma^2 (X^T X + \\lambda I)^{-1} X^T X (X^T X + \\lambda I)^{-1}',
        justification: 'Var(θ_λ) = R_λ·Var(θ_OLS)·R_λᵀ with Var(θ_OLS) = σ²(XᵀX)⁻¹. Each eigen-direction is scaled by (μₖ/(μₖ + λ))² < 1, so variance strictly shrinks as λ grows.',
      },
      {
        latex: '\\text{test error} \\approx \\underbrace{\\|(R_\\lambda - I)\\theta_{\\text{true}}\\|^2}_{\\text{bias}^2 \\uparrow} + \\underbrace{\\sigma^2 \\sum_k \\frac{\\mu_k}{(\\mu_k + \\lambda)^2}}_{\\text{variance} \\downarrow} + \\sigma^2',
        justification: 'Combining both: the tradeoff dial λ moves one term up and the other down. The optimal λ (test-error minimum) sits where the marginal variance cut equals the marginal bias cost — the sweet spot.',
      },
    ],
    derivedFrom: ['ridge-bias-variance'],
  },
];
