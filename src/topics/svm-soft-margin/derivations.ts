// src/topics/svm-soft-margin/derivations.ts
import type { Derivation } from '../../engine/types';

export const svmSoftDerivations: Derivation[] = [
  {
    id: 'hinge-convex-surrogate',
    title: 'Hinge Loss as a Convex Surrogate for the 0-1 Loss',
    steps: [
      {
        latex: '\\ell_{0-1}(y, f(x)) = [y\\,f(x) < 0] \\;=\\; \\begin{cases} 1 & y\\,f(x) < 0 \\\\ 0 & y\\,f(x) \\ge 0 \\end{cases}',
        justification: 'The classification error we actually care about: 1 if the sign of the score is wrong, 0 otherwise. It is a step function — neither convex nor differentiable, and minimizing it directly is NP-hard.',
      },
      {
        latex: '\\ell_{0-1}(y, f(x)) \\;\\le\\; \\max(0,\\; 1 - y\\,f(x)) = \\ell_{\\text{hinge}}(y, f(x))',
        justification: 'Check the three regimes: y·f ≥ 1 → hinge 0 = 0-1 0. 0 ≤ y·f < 1 → hinge = 1 − y·f ∈ (0, 1] but 0-1 = 0 (the point is correct — the hinge still pays!). y·f < 0 → hinge = 1 − y·f > 1 ≥ 1 = 0-1. So hinge ≥ 0-1 pointwise: it is an UPPER BOUND on the classification error.',
      },
      {
        latex: '\\max(0,\\; 1 - y\\,f) = \\max\\{ (0, 0), (1 - y\\,f, y\\,f) \\}',
        justification: 'A hinge is the maximum of two affine functions of (w, b) — a convex function. The 0-1 step is not convex; the hinge is the closest convex approximation that stays above it.',
      },
      {
        latex: '\\text{convex } \\Rightarrow \\text{ no local minima} \\Rightarrow \\text{ gradient/subgradient methods converge globally}',
        justification: 'This is why every practical SVM minimizes the hinge (or a smooth relative) instead of the 0-1 loss: convexity buys a unique optimum reachable by descent, at the cost of also penalizing correct-but-inside-the-margin points (the soft-margin tradeoff).',
      },
    ],
    derivedFrom: ['svm-hinge-loss'],
  },
  {
    id: 'primal-dual-box',
    title: 'Primal → Dual with the Box Constraint 0 ≤ αᵢ ≤ C',
    steps: [
      {
        latex: '\\min_{w,b,\\xi}\\; \\frac{1}{2}\\|w\\|^2 + C\\sum_i \\xi_i \\quad\\text{s.t.}\\quad y_i(w\\cdot x_i + b) \\ge 1 - \\xi_i,\\; \\xi_i \\ge 0',
        justification: 'The soft-margin primal: hinge slack Σξᵢ weighted by C, constraints as written. Convex (quadratic + linear) — strong duality holds.',
      },
      {
        latex: 'L = \\frac{1}{2}\\|w\\|^2 + C\\sum_i \\xi_i - \\sum_i \\alpha_i \\big[y_i(w\\cdot x_i + b) - 1 + \\xi_i\\big] - \\sum_i \\mu_i \\xi_i',
        justification: 'Lagrangian with multipliers αᵢ ≥ 0 (margin constraints) and μᵢ ≥ 0 (slack non-negativity). The two multiplier families are the reason the dual gets a box instead of a half-line.',
      },
      {
        latex: '\\frac{\\partial L}{\\partial w} = w - \\sum_i \\alpha_i y_i x_i = 0 \\;\\Rightarrow\\; w = \\sum_i \\alpha_i y_i x_i',
        justification: 'Stationarity in w: the optimal weight is a SPARSE weighted sum of the data — only points with αᵢ > 0 (support vectors) matter.',
      },
      {
        latex: '\\frac{\\partial L}{\\partial \\xi_i} = C - \\alpha_i - \\mu_i = 0 \\;\\Rightarrow\\; \\alpha_i + \\mu_i = C,\\; \\alpha_i \\ge 0,\\; \\mu_i \\ge 0 \\;\\Rightarrow\\; 0 \\le \\alpha_i \\le C',
        justification: 'THE key step: because μᵢ ≥ 0 must hold, stationarity forces αᵢ = C − μᵢ ≤ C. The upper BOX bound 0 ≤ αᵢ ≤ C is born — it does not exist in the hard-margin dual (there C = ∞).',
      },
      {
        latex: '\\max_{\\alpha}\\; \\sum_i \\alpha_i - \\frac{1}{2}\\sum_{i,j} \\alpha_i\\alpha_j\\, y_i y_j\\, (x_i\\cdot x_j) \\quad\\text{s.t.}\\quad \\sum_i \\alpha_i y_i = 0,\\; 0 \\le \\alpha_i \\le C',
        justification: 'Substituting w = Σαᵢyᵢxᵢ back and simplifying the Lagrangian gives the dual. It is a QP in n variables with one equality and n box constraints — every data point contributes (xᵢ·xⱼ), which is the slot the kernel trick later replaces with φ(xᵢ)·φ(xⱼ).',
      },
    ],
    derivedFrom: ['svm-dual-box'],
  },
  {
    id: 'slack-kkt',
    title: 'What the KKT Conditions Say About Slack and Support Vectors',
    steps: [
      {
        latex: 'w = \\sum_i \\alpha_i y_i x_i, \\qquad \\sum_i \\alpha_i y_i = 0, \\qquad 0 \\le \\alpha_i \\le C',
        justification: 'The stationarity and feasibility conditions restated — the box bound αᵢ ≤ C is the only change from the hard-margin dual.',
      },
      {
        latex: '\\alpha_i \\big[y_i(w\\cdot x_i + b) - 1 + \\xi_i\\big] = 0 \\;\\;\\text{(complementary slackness 1)}',
        justification: 'If the margin constraint is INACTIVE (point outside the band, y·f > 1), then αᵢ = 0 — the point exerts no force on w. Only points ON the band (y·f = 1, ξ = 0) or VIOLATING it (ξ > 0) can have αᵢ > 0: these are the support vectors.',
      },
      {
        latex: '\\mu_i\\, \\xi_i = 0 \\;\\Rightarrow\\; \\xi_i > 0 \\Rightarrow \\alpha_i = C \\;\\;\\text{(complementary slackness 2)}',
        justification: 'A point with positive slack (inside the band or misclassified) has μᵢ = 0, hence αᵢ = C — its multiplier is CLAMPED at the box. That is the precise sense in which C caps a single outlier\'s influence: it can contribute at most C·(its yᵢxᵢ) to w, however badly it is violated.',
      },
      {
        latex: '\\xi_i = 0,\\; \\alpha_i \\in (0, C): \\text{ free support vector (on the band)} \\qquad \\xi_i > 0,\\; \\alpha_i = C: \\text{ bounded support vector (inside/wrong side)}',
        justification: 'The three KKT regimes map exactly onto the geometry: free SVs on the band edges define the margin, bounded SVs with ξ > 0 are the points the soft margin tolerates. Both kinds are highlighted in the simulation.',
      },
    ],
    derivedFrom: ['svm-dual-box', 'svm-slack-constraints'],
  },
];
