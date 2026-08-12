// src/topics/svm-hard-margin/formulas.ts
// Measured anchors (nPerClass 12, margin 1.5, noise 0.45, seed 42 — the default):
//   ‖w‖ = 1.567, margin = 2/‖w‖ = 1.276, γ = 1/‖w‖ = 0.638, ½‖w‖² = 1.228
//   w = (1.520, 0.382), b = −0.230 (boundary tilted 14.1° off vertical)
//   2 support vectors: d9 (−0.406, −0.400) class 0 and d21 (0.832, −0.089) class 1,
//   both at distance 0.638 = 1/‖w‖ from the boundary; closest NON-support point
//   (d20) sits at 0.757 — the SV segment d9–d21 has length 1.276 = the margin.
//   Scaling the data by 2: ‖w‖ = 0.783 (= 1.567/2), margin = 2.553 (= 2×1.276),
//   boundary angle unchanged (14.1°), margin·‖w‖ = 2.000 — exact invariance.
import type { Formula } from '../../engine/types';

export const svmFormulas: Formula[] = [
  {
    id: 'svm-hyperplane',
    latex: 'w \\cdot x + b = 0',
    symbols: [
      { symbol: 'w', meaning: 'weight vector — the NORMAL to the separating hyperplane (in 2D: (w₁, w₂))', dimensions: 'per feature unit' },
      { symbol: 'b', meaning: 'bias / offset — the hyperplane\'s shift from the origin', dimensions: 'feature units' },
      { symbol: 'x', meaning: 'feature vector (2D point in the simulation)', dimensions: 'feature units' },
      { symbol: 'w \\cdot x + b', meaning: 'signed score: positive ⇒ class +1 side, negative ⇒ class −1 side', dimensions: 'score' },
    ],
    assumptions: ['Linearly separable data (hard margin)', 'Affine (linear) boundary — no kernel, no feature map in this topic'],
    failureCases: ['Non-separable data: no (w, b) satisfies the constraints — the hard-margin problem becomes infeasible', 'Zero weights (w = 0): the "hyperplane" degenerates to b = 0 and classifies nothing'],
    derivesFrom: [],
    derivationIds: ['svm-margin-formula'],
    connections: ['Linear algebra: the normal w is perpendicular to the boundary', 'Decision boundary', 'Logistic regression (z = w·x + b, same score)'],
    whyWorks: 'A hyperplane is the set of points where the score is exactly 0. Its normal w determines the ORIENTATION of the boundary; b shifts it. In the default simulation run w = (1.520, 0.382) tilts the boundary 14.1° off the vertical so the margin is widest — the orientation is NOT arbitrary: it is chosen to maximize the gap.',
  },
  {
    id: 'svm-functional-margin',
    latex: '\\hat{\\gamma}_i = y_i (w \\cdot x_i + b)',
    symbols: [
      { symbol: 'y_i', meaning: 'label in ±1 form (class 0 → −1, class 1 → +1)', dimensions: '±1' },
      { symbol: 'w \\cdot x_i + b', meaning: 'raw score of point i', dimensions: 'score' },
      { symbol: '\\hat{\\gamma}_i', meaning: 'FUNCTIONAL margin of point i — the unnormalized signed clearance', dimensions: 'score' },
    ],
    assumptions: ['Labels are ±1 (not 0/1): multiplying the score by yᵢ makes the margin positive exactly when the point is classified correctly'],
    failureCases: ['A point classified correctly can have ARBITRARY functional margin — scaling w by 2 doubles every functional margin (the metric is not scale-invariant)', 'Zero score ⇒ γ̂ᵢ = 0 — the point lies exactly on the boundary'],
    derivesFrom: ['svm-hyperplane'],
    derivationIds: ['svm-margin-formula'],
    connections: ['Geometric margin', 'Margin = 2/‖w‖'],
    whyWorks: 'The functional margin is the signed distance IN SCORE UNITS: yᵢ times the score is positive for a correct prediction and its magnitude measures confidence in score units. Its weakness is that it depends on the arbitrary scale of w — which is exactly why the canonical constraint fixes the scale (see svm-primal).',
  },
  {
    id: 'svm-geometric-margin',
    latex: '\\gamma_i = \\frac{y_i (w \\cdot x_i + b)}{\\|w\\|}',
    symbols: [
      { symbol: '\\|w\\|', meaning: 'Euclidean norm of the weight vector', dimensions: 'per feature unit' },
      { symbol: '\\gamma_i', meaning: 'GEOMETRIC margin of point i — the actual distance from xᵢ to the hyperplane', dimensions: 'feature units' },
    ],
    assumptions: ['Euclidean geometry (L2 norm)', 'Distance is measured perpendicular to the hyperplane'],
    failureCases: ['The geometric margin is the honest distance but it still varies per point — the SVM cares about the MINIMUM over all points', 'At w = 0 the quotient is undefined (the hyperplane does not exist)'],
    derivesFrom: ['svm-functional-margin'],
    derivationIds: ['svm-margin-formula'],
    connections: ['Margin = 2/‖w‖', 'Support vectors', 'Distance from a point to a plane'],
    whyWorks: 'Dividing the score by ‖w‖ converts score units into feature units: γᵢ is literally the perpendicular distance from xᵢ to the boundary. In the default run the two support vectors sit at exactly γ = 0.638 = 1/‖w‖, while the nearest ordinary point is 0.757 away — the geometric margin is what the eye sees as the width of the gap.',
  },
  {
    id: 'svm-margin',
    latex: '\\text{margin} = \\frac{2}{\\|w\\|}',
    symbols: [
      { symbol: '\\text{margin}', meaning: 'width of the margin band — the distance between the two parallel margin lines (w·x + b = ±1)', dimensions: 'feature units' },
      { symbol: '\\|w\\|', meaning: 'norm of the canonical weight vector', dimensions: 'per feature unit' },
      { symbol: '2', meaning: 'the canonical functional-margin width (yᵢ(w·xᵢ+b) = 1 at the margin lines)', dimensions: 'score' },
    ],
    assumptions: ['Canonical scaling: the closest points (support vectors) satisfy yᵢ(w·xᵢ+b) = 1', 'The margin is measured in geometric (feature) units — the formula is what remains after dividing the functional margin band 2 by ‖w‖'],
    failureCases: ['Without the canonical constraint the formula is meaningless: you could inflate the margin to any value by dividing w by 2', 'Non-separable data: the constraint yᵢ(w·xᵢ+b) ≥ 1 cannot be met at all — there is no margin to compute'],
    derivesFrom: ['svm-functional-margin', 'svm-geometric-margin'],
    derivationIds: ['svm-margin-formula'],
    connections: ['SVM objective ½‖w‖²', 'Support vectors'],
    whyWorks: 'Derived in derivations.ts: under canonical scaling the two margin lines sit at functional margin ±1, so the band has geometric width 2/‖w‖. Measured on the default seed: ‖w‖ = 1.567 ⇒ margin = 1.276 — and the two support vectors d9–d21 are exactly 1.276 apart, each 0.638 from the boundary. Maximizing this width is the entire SVM objective, restated in svm-primal.',
  },
  {
    id: 'svm-primal',
    latex: '\\min_{w,b} \\; \\tfrac{1}{2}\\|w\\|^2 \\quad \\text{s.t.} \\quad y_i (w \\cdot x_i + b) \\ge 1 \\;\\; \\forall i',
    symbols: [
      { symbol: '\\tfrac{1}{2}\\|w\\|^2', meaning: 'the SVM objective — half the squared norm of the weights (equivalent to maximizing the margin)', dimensions: 'per feature unit²' },
      { symbol: 'y_i (w \\cdot x_i + b) \\ge 1', meaning: 'canonical constraints: every point must be at least one functional-margin unit from the boundary', dimensions: 'score' },
      { symbol: '\\forall i', meaning: 'one constraint per training point', dimensions: 'count' },
    ],
    assumptions: ['Feasibility: a (w, b) satisfying all constraints exists — i.e. the data is linearly separable (hard margin)', 'Labels ±1'],
    failureCases: ['Infeasible when the data is NOT separable (see failures.ts: the non-separable demo fails cleanly via telemetry)', 'With outliers close to the other class, the feasible margin shrinks to a razor-thin band (½‖w‖² balloons: at margin 0.316 it is ≈ 19.97 on the failure demo)'],
    derivesFrom: ['svm-margin'],
    derivationIds: ['svm-primal-dual'],
    connections: ['Convex optimization', 'Dual problem', 'KKT conditions', 'Lagrange multipliers'],
    whyWorks: 'Maximizing the margin 2/‖w‖ is the same as minimizing ‖w‖; the ½ and the square make the objective convex and smooth. The ½‖w‖² is the loss curve this topic plots (lossMetricKey): on the default seed it descends 1.2409 → 1.2276 across the 40-step candidate sweep, bottoming out at the max-margin solution. The constraints pick the scale of w (the canonical one) so that ‖w‖ is not arbitrary.',
  },
  {
    id: 'svm-dual',
    latex: '\\max_{\\alpha} \\; \\sum_i \\alpha_i - \\frac{1}{2}\\sum_{i,j} \\alpha_i \\alpha_j y_i y_j (x_i \\cdot x_j) \\quad \\text{s.t.} \\; \\alpha_i \\ge 0,\\; \\sum_i \\alpha_i y_i = 0',
    symbols: [
      { symbol: '\\alpha_i', meaning: 'dual variable (Lagrange multiplier) for the i-th constraint', dimensions: '≥ 0' },
      { symbol: 'x_i \\cdot x_j', meaning: 'inner product of two training points — the kernel function K(xᵢ, xⱼ) in the linear case', dimensions: 'feature units²' },
      { symbol: '\\sum_i \\alpha_i y_i = 0', meaning: 'stationarity of the bias — the dual feasibility condition', dimensions: 'constraint' },
    ],
    assumptions: ['Strong duality holds (Slater: the primal is feasible — separable data)', 'Linear kernel K(x, x′) = x·x′'],
    failureCases: ['If the primal is infeasible (non-separable data) the dual is unbounded above — another face of the same failure', 'The dual only depends on inner products: without a kernel this fixes the boundary to be linear'],
    derivesFrom: ['svm-primal'],
    derivationIds: ['svm-primal-dual'],
    connections: ['Lagrange duality', 'KKT conditions', 'Kernel trick'],
    whyWorks: 'Derived from the primal via the Lagrangian in derivations.ts. The dual is a quadratic program over α with an O(n²) pairwise term; its optimum value equals the primal optimum (strong duality). Its real gift is structural: w = Σαᵢyᵢxᵢ, so the model is a WEIGHTED SUM OF DATA POINTS — and by KKT only the support vectors carry nonzero α.',
  },
  {
    id: 'svm-kkt',
    latex: '\\alpha_i \\ge 0, \\quad y_i (w \\cdot x_i + b) - 1 \\ge 0, \\quad \\alpha_i\\, [y_i (w \\cdot x_i + b) - 1] = 0',
    symbols: [
      { symbol: '\\alpha_i', meaning: 'Lagrange multiplier (dual variable) of constraint i', dimensions: '≥ 0' },
      { symbol: 'y_i (w \\cdot x_i + b) - 1', meaning: 'constraint slack in canonical form (0 exactly at the margin)', dimensions: 'score' },
      { symbol: '\\alpha_i \\cdot [\\ldots] = 0', meaning: 'complementary slackness: at least one factor is zero per point', dimensions: 'product' },
    ],
    assumptions: ['Convex primal + feasible (separable) ⇒ KKT conditions are necessary AND sufficient', 'Differentiable objective and constraints (they are: the margin band has no corners in the dual sense)'],
    failureCases: ['For non-separable data the KKT system has no solution — the soft-margin extension introduces slack ξᵢ and changes the complementary-slackness form', 'Degenerate cases with collinear support vectors can give more than 3 active constraints (a rectangle of points, for example, makes all 4 corners support vectors)'],
    derivesFrom: ['svm-primal', 'svm-dual'],
    connections: ['Support vectors', 'Dual problem', 'Complementary slackness'],
    whyWorks: 'Complementary slackness is the SVM\'s pruning theorem: for every point, EITHER its constraint is inactive (yᵢ(w·xᵢ+b) > 1 ⇒ αᵢ = 0) OR it lies exactly on the margin (αᵢ > 0 ⇒ yᵢ(w·xᵢ+b) = 1). Since w = Σαᵢyᵢxᵢ, all points with αᵢ = 0 contribute NOTHING to the model. The default seed keeps exactly 2 points (d9, d21) as support vectors out of 24 — the 22 others are irrelevant to the boundary.',
  },
  {
    id: 'svm-convexity',
    latex: '\\tfrac{1}{2}\\|w\\|^2 \\;\\text{is strictly convex in } w \\quad (\\nabla^2 = I \\succ 0)',
    symbols: [
      { symbol: '\\nabla^2 (\\tfrac{1}{2}\\|w\\|^2)', meaning: 'Hessian of the objective — the identity matrix, positive definite', dimensions: 'matrix' },
      { symbol: '\\tfrac{1}{2}\\|w\\|^2', meaning: 'the primal objective', dimensions: 'per feature unit²' },
    ],
    assumptions: ['The feasible set (intersection of half-spaces yᵢ(w·xᵢ+b) ≥ 1) is a convex polytope', 'Linearly separable data keeps the feasible set non-empty'],
    failureCases: ['The constraint set becomes EMPTY for non-separable data — convex but infeasible (the failure is feasibility, not non-convexity)', 'The margin lines are non-differentiable in the b-direction at the optimum — but the optimum is still unique in w'],
    derivesFrom: ['svm-primal'],
    connections: ['Convex optimization', 'Global optimum', 'KKT conditions'],
    whyWorks: 'A strictly convex objective over a convex feasible set has a UNIQUE global minimum — no local minima, no saddle points. That is why the geometric solver (this topic) and every QP solver agree on the same answer, and why the sweep\'s final snapshot is provably the max-margin separator: the problem has exactly one optimum. The 2D geometric solver finds it exactly by enumerating the finite set of candidate support directions rather than iterating.',
  },
];
