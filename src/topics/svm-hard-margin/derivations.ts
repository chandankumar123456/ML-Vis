// src/topics/svm-hard-margin/derivations.ts
import type { Derivation } from '../../engine/types';

export const svmDerivations: Derivation[] = [
  {
    id: 'svm-margin-formula',
    title: 'The Margin Formula: margin = 2/‖w‖',
    steps: [
      {
        latex: '\\hat{\\gamma}_i = y_i (w \\cdot x_i + b) \\quad\\Rightarrow\\quad \\gamma_i = \\frac{\\hat{\\gamma}_i}{\\|w\\|}',
        justification: 'The functional margin is the signed score yᵢ(w·xᵢ+b). Because the score is linear in w, dividing by ‖w‖ converts it into the perpendicular distance from xᵢ to the hyperplane — the geometric margin γᵢ.',
      },
      {
        latex: 'y_i (w \\cdot x_i + b) \\ge 1 \\;\\text{(canonical form)} \\;\\Rightarrow\\; \\gamma_i \\ge \\frac{1}{\\|w\\|}',
        justification: 'Fixing the functional margin of the closest point to 1 pins down the scale of w (otherwise w could be rescaled freely and the margin would be arbitrary). Under canonical scaling every point is at least 1/‖w‖ away from the boundary.',
      },
      {
        latex: '\\text{band} = \\gamma_{\\max} - \\gamma_{\\min} = \\frac{1}{\\|w\\|} - \\left(-\\frac{1}{\\|w\\|}\\right) = \\frac{2}{\\|w\\|}',
        justification: 'The two margin lines w·x + b = +1 and w·x + b = −1 lie at distances +1/‖w‖ and −1/‖w‖ from the boundary, so the total width of the margin band is 2/‖w‖. Measured on the default seed: ‖w‖ = 1.567 gives margin = 1.276, and the support-vector segment d9–d21 measures exactly 1.276.',
      },
      {
        latex: '\\max_{w,b} \\frac{2}{\\|w\\|} \\;\\equiv\\; \\min_{w,b} \\tfrac{1}{2}\\|w\\|^2',
        justification: 'Maximizing the band width is equivalent to minimizing ‖w‖. The ½·square is a cosmetic but crucial restatement: it makes the objective strictly convex (Hessian = identity), so the problem has a unique global optimum — the property the geometric solver and every QP solver lean on.',
      },
    ],
    derivedFrom: ['svm-functional-margin', 'svm-geometric-margin'],
  },
  {
    id: 'svm-primal-dual',
    title: 'Primal → Dual: The Lagrangian and the α Problem',
    steps: [
      {
        latex: 'L(w, b, \\alpha) = \\tfrac{1}{2}\\|w\\|^2 - \\sum_{i=1}^{n} \\alpha_i \\big[y_i (w \\cdot x_i + b) - 1\\big], \\quad \\alpha_i \\ge 0',
        justification: 'Attach a Lagrange multiplier αᵢ ≥ 0 to each constraint yᵢ(w·xᵢ+b) ≥ 1. At the optimum the constraints are active exactly for the support vectors, so their αᵢ are the nonzero ones.',
      },
      {
        latex: '\\nabla_w L = 0 \\;\\Rightarrow\\; w = \\sum_i \\alpha_i y_i x_i',
        justification: 'Stationarity in w: ½‖w‖² differentiates to w, and the constraint term to −Σαᵢyᵢxᵢ. Setting the gradient to zero gives the celebrated result: the optimal weight vector is a weighted sum of the DATA POINTS.',
      },
      {
        latex: '\\frac{\\partial L}{\\partial b} = 0 \\;\\Rightarrow\\; \\sum_i \\alpha_i y_i = 0',
        justification: 'Stationarity in the bias b. The constraint term contributes −Σαᵢyᵢ (the xᵢ term vanishes because w·xᵢ does not depend on b), giving the dual feasibility condition.',
      },
      {
        latex: '\\frac{1}{2}\\|w\\|^2 \\to \\frac{1}{2}\\Big(\\sum_i \\alpha_i y_i x_i\\Big) \\cdot \\Big(\\sum_j \\alpha_j y_j x_j\\Big) = \\frac{1}{2}\\sum_{i,j} \\alpha_i \\alpha_j y_i y_j\\, (x_i \\cdot x_j)',
        justification: 'Substitute w = Σαᵢyᵢxᵢ into the primal objective. The pairwise inner products xᵢ·xⱼ appear — the seed of the kernel trick: the dual depends on the data ONLY through inner products.',
      },
      {
        latex: '\\max_{\\alpha} \\sum_i \\alpha_i - \\frac{1}{2}\\sum_{i,j} \\alpha_i \\alpha_j y_i y_j (x_i \\cdot x_j) \\;\\;\\text{s.t.}\\; \\alpha_i \\ge 0,\\; \\sum_i \\alpha_i y_i = 0',
        justification: 'Combining the objective with the constraint term Σαᵢ (which survives because the primal had Σαᵢ(yᵢ(w·xᵢ+b) − 1) and the b-dependent part integrates to zero by Σαᵢyᵢ = 0) yields the dual QP. By strong duality (Slater holds on separable data) its optimum equals the primal optimum.',
      },
    ],
    derivedFrom: ['svm-primal'],
  },
  {
    id: 'svm-kkt-sv',
    title: 'KKT Complementary Slackness → Only Support Vectors Matter',
    steps: [
      {
        latex: '\\alpha_i \\big[y_i (w \\cdot x_i + b) - 1\\big] = 0',
        justification: 'Complementary slackness: for every point, at least one of the two factors is zero. Either the constraint is slack (the point is strictly inside its class region) or the multiplier is zero (the point lies strictly inside the band).',
      },
      {
        latex: 'y_i (w \\cdot x_i + b) > 1 \\;\\Rightarrow\\; \\alpha_i = 0',
        justification: 'A point NOT on the margin has zero multiplier — it does not "pull" on the boundary at all. In the default run 22 of 24 points fall into this bucket: their αᵢ are exactly 0.',
      },
      {
        latex: '\\alpha_i > 0 \\;\\Rightarrow\\; y_i (w \\cdot x_i + b) = 1 \\;\\Rightarrow\\; x_i \\;\\text{is a support vector}',
        justification: 'A nonzero multiplier forces the point onto its margin line (distance exactly 1/‖w‖ from the boundary). The default seed has exactly two such points: d9 (class 0) at (−0.406, −0.400) and d21 (class 1) at (0.832, −0.089), each 0.638 from the boundary.',
      },
      {
        latex: 'w = \\sum_{i \\in SV} \\alpha_i y_i x_i',
        justification: 'Combined with w = Σαᵢyᵢxᵢ, only the support vectors survive the sum. The boundary depends on a handful of points — which is why moving a NON-support point (d20, say) leaves the max-margin separator unchanged, while moving a support vector re-solves the whole problem.',
      },
    ],
    derivedFrom: ['svm-dual', 'svm-kkt'],
  },
];
