// src/topics/lasso-regression/derivations.ts
import type { Derivation } from '../../engine/types';

export const lassoDerivations: Derivation[] = [
  {
    id: 'lasso-soft-threshold',
    title: 'Deriving the Soft-Threshold Update from the Subgradient',
    steps: [
      {
        latex: 'J_j(w) = \\frac{1}{2} (w - z)^2 + \\lambda |w|',
        justification: 'Freeze every coordinate except j. The MSE part becomes a single quadratic in w with z = ρⱼ/n the OLS coordinate solution; the penalty contributes λ|w|.',
      },
      {
        latex: 'w > 0:\\quad \\frac{dJ_j}{dw} = (w - z) + \\lambda = 0 \\quad\\Rightarrow\\quad w = z - \\lambda \\;\\text{ valid iff } z > \\lambda',
        justification: 'For positive w the penalty is differentiable with derivative +λ. The solution z − λ is only consistent when z > λ.',
      },
      {
        latex: 'w < 0:\\quad \\frac{dJ_j}{dw} = (w - z) - \\lambda = 0 \\quad\\Rightarrow\\quad w = z + \\lambda \\;\\text{ valid iff } z < -\\lambda',
        justification: 'For negative w the derivative of |w| is −λ. The solution z + λ is consistent when z < −λ.',
      },
      {
        latex: 'w = 0:\\quad \\partial J_j(0) = -z + \\lambda \\cdot [-1, 1] \\ni 0 \\;\\iff\\; |z| \\le \\lambda',
        justification: 'At the kink the subgradient is the interval [−1,1] scaled by λ. Zero is in the subgradient exactly when |z| ≤ λ — the coefficient is pulled to exactly 0.',
      },
      {
        latex: 'w^\\star = \\operatorname{sign}(z) \\cdot \\max(|z| - \\lambda, \\, 0) = S(z, \\lambda)',
        justification: 'Combine the three cases: shift by λ in the direction of z when |z| > λ, otherwise clamp to 0. This is the soft-threshold operator.',
      },
      {
        latex: '\\tilde{\\theta}_j \\leftarrow S\\!\\left(\\frac{1}{n} \\sum_{i=1}^{n} z_{ij}\\, r_i^{(-j)}, \\; \\lambda \\right)',
        justification: 'Substitute z = ρⱼ/n with the partial residual rᵢ^(−j) = yᵢ − b̃ − Σ_{k≠j} θ̃ₖ zᵢₖ — the full coordinate descent sweep.',
      },
    ],
    derivedFrom: ['cd-update', 'subgradient'],
  },
  {
    id: 'lasso-vs-ridge-geometry',
    title: 'Why L1 Touches a Corner (Sparsity) and L2 Does Not',
    steps: [
      {
        latex: '\\min_\\theta \\|y - X\\theta\\|_2^2 \\quad \\text{s.t.} \\quad \\|\\theta\\|_1 \\le t',
        justification: 'The lasso is the L2-loss problem with an L1 constraint — an equivalent (KKT) form of adding λ‖θ‖₁ to the objective.',
      },
      {
        latex: '\\|\\theta\\|_1 = |\\theta_1| + |\\theta_2| \\le t \\quad\\text{is a DIAMOND with vertices on the axes}',
        justification: 'In 2-D the L1 ball is the rotated square |θ₁| + |θ₂| ≤ t — its corners lie exactly on the θ₁ and θ₂ axes.',
      },
      {
        latex: '\\|\\theta\\|_2 = \\sqrt{\\theta_1^2 + \\theta_2^2} \\le t \\quad\\text{is a CIRCLE with no corners}',
        justification: 'The L2 ball is smooth — the boundary never passes through an axis-aligned vertex.',
      },
      {
        latex: '\\text{Level sets of } \\|y - X\\theta\\|^2 \\text{ are ellipses centered at } \\hat{\\theta}_{OLS}',
        justification: 'Each contour is an ellipse; as the constraint shrinks, the optimum slides from the OLS center toward the feasible region boundary.',
      },
      {
        latex: '\\text{L1: optimum lands ON a vertex} \\Rightarrow \\theta_j = 0 \\text{ exactly}',
        justification: 'For most positions of the unconstrained optimum, the first level set that touches the diamond does so at a corner — one coordinate is exactly 0.',
      },
      {
        latex: '\\text{L2: optimum lands on the smooth arc} \\Rightarrow \\theta_j \\neq 0 \\text{ for all } j',
        justification: 'A circle has no vertices, so the tangency point generically has every coordinate nonzero — ridge shrinks but never selects.',
      },
    ],
    derivedFrom: ['l1-geometry', 'ridge-objective'],
  },
];
