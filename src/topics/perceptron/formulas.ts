// src/topics/perceptron/formulas.ts
// Measured anchors (nPerClass 20, margin 1.2, noise 0.5, η 1, zero init, seed 42 —
// the default):
//   R = max‖x‖ relative to the centroid = 2.2446
//   converges in 4 updates; 6 snapshots (init + 4 updates + converged re-emission)
//   final w = (2.7135, 0.3850), b = 0, ‖w‖ = 2.7407 — well under 2R ≈ 4.489
//   geometric margin of the found separator γ = min y·(w·x+b)/‖w‖ = 0.0472
//   convergence bound (R·‖w*‖/γ)² ≈ 16982 vs 4 actual updates (the bound is loose)
//   per-update ‖Δw‖ ≤ η·√(R²+1) = 2.457; measured max ‖Δw‖ = 1.958 (incl. bias)
//   seed 7: 23 updates. Non-separable (separable: false): cap at 180 updates.
//   η-invariance: η = 1 and η = 0.5 both take exactly 4 updates; weights scale ×2.
import type { Formula } from '../../engine/types';

export const perceptronFormulas: Formula[] = [
  {
    id: 'perceptron-score',
    latex: 's(x) = w \\cdot x + b, \\qquad \\hat{y} = \\begin{cases} \\text{class 1} & s > 0 \\\\ \\text{class 0} & s \\le 0 \\end{cases}',
    symbols: [
      { symbol: 'w', meaning: 'weight vector (in 2D: (w₁, w₂)) — the normal to the decision boundary; accumulated from mistake updates', dimensions: 'per feature unit' },
      { symbol: 'x', meaning: 'feature vector (2D point in the simulation)', dimensions: 'feature units' },
      { symbol: 'b', meaning: 'bias — shifts the boundary; also accumulated from mistake updates (b ← b + η·y)', dimensions: 'feature units' },
      { symbol: 's(x)', meaning: 'linear score; its SIGN (not magnitude) decides the class, exactly as in SVM and logistic regression', dimensions: 'score' },
    ],
    assumptions: ['Linearly separable target concept (for convergence — the rule itself runs on any data)', 'Labels are ±1 internally (class 0 → −1, class 1 → +1) so the update formula is symmetric'],
    failureCases: ['With s exactly 0 (e.g. zero weights at the start) every point counts as a mistake — the scan picks the first one deterministically', 'On non-separable data the score is correct most of the time but a few points keep flipping — no (w, b) exists with y·s > 0 everywhere'],
    derivesFrom: [],
    derivationIds: ['perceptron-update-rule'],
    connections: ['SVM hyperplane (same score w·x + b)', 'Logistic regression (z = w·x + b)', 'Linear threshold unit (McCulloch–Pitts → perceptron)'],
    whyWorks: 'The score is the signed distance in score units: its sign is the prediction, its absolute value is the confidence. The perceptron only READS this score when deciding whether to update — unlike SVM it never optimizes it. In the default run the final score pattern on the training set is correct for all 40 points (accuracy 1) while the boundary itself is arbitrary among many separable lines.',
  },
  {
    id: 'perceptron-update',
    latex: '\\text{on a mistake } (y_i \\cdot s(x_i) \\le 0): \\quad w \\leftarrow w + \\eta\\, y_i\\, x_i, \\quad b \\leftarrow b + \\eta\\, y_i',
    symbols: [
      { symbol: 'y_i', meaning: 'true label of the triggering point in ±1 form (class 0 → −1, class 1 → +1)', dimensions: '±1' },
      { symbol: '\\eta', meaning: 'learning rate (fixed increment). With fixed η the rule is scale-invariant: the UPDATE SEQUENCE does not depend on η at all', dimensions: 'unitless' },
      { symbol: 'y_i\\, x_i', meaning: 'the correction vector — proportional to the feature vector of the mistaken point, aimed at the correct class side', dimensions: 'feature units' },
      { symbol: 'w \\leftarrow w + \\eta y_i x_i', meaning: 'one ONLINE update per mistake: the weight vector moves to pull the boundary toward correctness on this point', dimensions: 'per feature unit' },
    ],
    assumptions: ['Mistake-only update: correct points are examined and skipped, they never change w', 'Fixed learning rate η > 0 (not decaying — nothing in the classic rule decays)'],
    failureCases: ['An exact float weight-state repeat on non-separable data would loop forever — the simulator detects cycles and caps runs honestly (measured: no exact cycle within 5000 updates on seed 42; the 180-update cap fires first)', 'Huge η (≥ 1000) balloons the weights (final ‖w‖ ×1000) without changing the boundary — a numerical-scale failure, blocked by the parameter validation'],
    derivesFrom: ['perceptron-score'],
    derivationIds: ['perceptron-update-rule', 'perceptron-eta-invariance'],
    connections: ['Perceptron convergence theorem', 'Stochastic gradient descent (SGD is a gradient step; the perceptron update is a correction step — related, not identical)', 'Online learning'],
    whyWorks: 'Step semantics in the simulator: ONE update per step. Each step scans the points in a fixed deterministic order from the current position for the FIRST point with y·s ≤ 0, applies the update above, and moves past it. Measured on the default seed: the first update fires on point 0 (every score is 0 with zero weights), and each step adds at most η·√(R²+1) to the (w₁, w₂, b) vector — measured max ‖Δw‖ = 1.958 ≤ 2.457.',
  },
  {
    id: 'perceptron-geometric-margin',
    latex: '\\gamma = \\min_i \\frac{y_i\\,(w \\cdot x_i + b)}{\\|w\\|}',
    symbols: [
      { symbol: '\\gamma', meaning: 'geometric margin of a separator (w, b) — the smallest perpendicular distance from a training point to the boundary', dimensions: 'feature units' },
      { symbol: '\\|w\\|', meaning: 'Euclidean norm of the weight vector — divides the score into feature units', dimensions: 'per feature unit' },
      { symbol: '\\min_i', meaning: 'taken over the training points the separator classifies correctly', dimensions: 'index' },
    ],
    assumptions: ['w ≠ 0 (the boundary exists)', 'The margin is measured in GEOMETRIC units — unlike SVM the perceptron never optimizes it; it is a diagnostic quantity used by the convergence bound'],
    failureCases: ['The perceptron converges to SOME separator with γ > 0 on separable data — but γ is not maximized; a different update order yields a different (w, b) with a different γ', 'On non-separable data no separator has γ > 0 for all points — γ of any candidate is negative for the misclassified ones'],
    derivesFrom: ['perceptron-score'],
    derivationIds: ['perceptron-novikoff-bound'],
    connections: ['SVM geometric margin (identical definition, opposite role: SVM maximizes it, perceptron ignores it)', 'Distance from a point to a line'],
    whyWorks: 'This is the same geometric margin the SVM maximizes — but for the perceptron it is a LURKING quantity. The separability of the data is what makes γ > 0 achievable; the magnitude of γ relative to the data radius R is exactly what the convergence bound below converts into an update count. Measured on the default separator: γ = 0.0472 — the closest training point sits 0.047 feature units from the boundary, a tiny clearance compared to the SVM\'s max-margin solution.',
  },
  {
    id: 'perceptron-convergence-bound',
    latex: '\\text{updates} \\;\\le\\; \\left( \\frac{R\\,\\|w^*\\|}{\\gamma} \\right)^2',
    symbols: [
      { symbol: 'R', meaning: 'data radius — max‖x‖ over the training set (measured 2.2446 on the default draw)', dimensions: 'feature units' },
      { symbol: 'w^*', meaning: 'the separator the rule actually converges to (or any separating solution) — its norm does the bookkeeping', dimensions: 'per feature unit' },
      { symbol: '\\gamma', meaning: 'geometric margin of w* — the clearance of the closest point (measured 0.0472 on the default separator)', dimensions: 'feature units' },
      { symbol: '\\left(\\frac{R\\,\\|w^*\\|}{\\gamma}\\right)^2', meaning: 'Novikoff bound: the number of mistake updates is at most this — a FINITE bound whenever separable data makes γ > 0 feasible', dimensions: 'count' },
    ],
    assumptions: ['Linearly separable data: some (w, b) classifies every point correctly ⇒ the classic fixed-increment rule CONVERGES', 'Fixed learning rate η (the bound is η-independent, matching the measured η-invariance)'],
    failureCases: ['For NON-separable data γ does not exist for any separator and the bound is meaningless — the rule enters a cycle instead (the classical cycling theorem; the simulator caps this at 180 updates with honest telemetry)', 'The bound is typically enormous: on the default run it evaluates to ≈ 16982 updates while the rule takes 4 — reading the bound as a performance PREDICTION is a common mistake; it is a worst-case guarantee'],
    derivesFrom: ['perceptron-geometric-margin', 'perceptron-update'],
    derivationIds: ['perceptron-novikoff-bound'],
    connections: ['Perceptron convergence theorem (Novikoff 1962)', 'Margin-based generalization bounds (same R/γ structure)', 'SVM margin'],
    whyWorks: 'The bound is the perceptron\'s core guarantee: IF the data is linearly separable (γ > 0 achievable, R finite) THEN the rule makes at most (R‖w*‖/γ)² updates and then classifies perfectly — no learning-rate tuning needed, no local minima. Measured on the default seed: 4 updates ≈ 0.02% of the bound; on seed 7: 23. On non-separable data the theorem does NOT apply and the rule provably cycles — the simulator\'s honest cap demonstrates that exact failure mode.',
  },
  {
    id: 'perceptron-eta-invariance',
    latex: 'y\\,(w \\cdot x + b) \\le 0 \\;\\Longleftrightarrow\\; y\\,(\\eta w \\cdot x + \\eta b) \\le 0 \\quad (\\eta > 0)',
    symbols: [
      { symbol: '\\eta', meaning: 'learning rate — multiplies every weight; the comparison above uses y·(ηw·x + ηb) = η·y·(w·x + b)', dimensions: 'unitless' },
      { symbol: '\\Longleftrightarrow', meaning: 'the mistake condition is preserved under uniform rescaling of (w, b) by any η > 0', dimensions: 'logic' },
    ],
    assumptions: ['Fixed η from the start (the classic rule has a single constant learning rate)', 'Zero initialization (the standard textbook setting) — the initial state is η-agnostic'],
    failureCases: ['The SCALE of w changes by the same factor: the final weights are η·(the η = 1 weights) — measured ‖w‖ = 2.7407 at η = 1 and 1.3704 at η = 0.5', 'Huge η still blows up numerically (‖w‖ ×1000 at η = 1000) even though the boundary is identical — the invariance is algebraic, not digital'],
    derivesFrom: ['perceptron-update'],
    derivationIds: ['perceptron-eta-invariance'],
    connections: ['Learning-rate tuning (the classic rule needs NONE — a fact that surprises since every SGD-family rule needs one)', 'SGD vs perceptron'],
    whyWorks: 'Each update replaces w by w + η·y·x. Writing w = η·v, every update becomes v ← v + y·x with identical v-weights regardless of η — so the whole MISTAKE SEQUENCE is scale-invariant. The simulator measures exactly this: η = 1 and η = 0.5 both converge in 4 updates on the default seed, with final weights differing by exactly the η ratio. The learning rate is a red herring in the classic perceptron.',
  },
];