// src/topics/perceptron/derivations.ts
// All numeric claims measured against the simulator (default seed 42):
//   4 updates on the default draw, final ‖w‖ = 2.7407, γ = 0.0472, R = 2.2446,
//   (R/γ)² ≈ 2261; seed 7 → 23 updates; non-separable → cap at 180.
import type { Derivation } from '../../engine/types';

export const perceptronDerivations: Derivation[] = [
  {
    id: 'perceptron-update-rule',
    title: 'Why the Update is w ← w + η·y·x (the mistake-correction geometry)',
    steps: [
      {
        latex: 'y_i\\, s(x_i) \\le 0 \\;\\Rightarrow\\; \\text{point } i \\text{ is on the wrong side of } w \\cdot x + b = 0',
        justification: 'With ±1 labels, yᵢ·s(xᵢ) is positive exactly when the prediction agrees with the label. The scan (deterministic order, first mistake wins) fires an update only for such points — correct points never touch w.',
      },
      {
        latex: 's(x_i + \\eta y_i x_i) = w \\cdot x_i + b + \\eta y_i \\|x_i\\|^2',
        justification: 'After the update w ← w + η·yᵢ·xᵢ, the score of the triggering point rises by η·yᵢ·‖xᵢ‖² — the point moves TOWARD the correct side by an amount proportional to its norm. The geometric meaning of the update: pull the boundary toward this specific point so its score flips sign.',
      },
      {
        latex: '\\|\\Delta w\\| = \\eta\\,\\|y_i x_i\\| = \\eta\\,\\|x_i\\| \\le \\eta R, \\qquad \\|\\Delta (w, b)\\| \\le \\eta\\sqrt{R^2 + 1}',
        justification: 'Each update changes the weight vector by a vector of length η·‖xᵢ‖ ≤ η·R (triangle inequality bounds the accumulated ‖w‖ by the number of updates times ηR). Including the bias b ← b + η·yᵢ in the 3D state gives the measured snapshot bound: every ‖Δw‖ ≤ η·√(R²+1) = 2.457, with max observed 1.958.',
      },
      {
        latex: '\\text{perceptron: } w \\leftarrow w + \\eta y_i x_i \\qquad \\text{SGD: } w \\leftarrow w - \\eta \\nabla_w \\ell_i(w)',
        justification: 'The perceptron update resembles an SGD step but is NOT a gradient step: there is no loss function being minimized. SGD on a loss would shrink the weights (regularization) and update on EVERY point; the perceptron updates ONLY on mistakes and never regularizes. The classic rule is a pure correction rule, not an optimizer.',
      },
    ],
    derivedFrom: ['perceptron-update'],
  },
  {
    id: 'perceptron-novikoff-bound',
    title: 'The Convergence Theorem: updates ≤ (R/γ)²',
    steps: [
      {
        latex: '\\|w_{k+1}\\|^2 = \\|w_k + \\eta\\, y_i x_i\\|^2 = \\|w_k\\|^2 + 2\\eta\\, y_i (w_k \\cdot x_i) + \\eta^2 \\|x_i\\|^2 \\le \\|w_k\\|^2 + \\eta^2 R^2',
        justification: 'Expand the square of the update w_{k+1} = w_k + η·yᵢ·xᵢ. The update fires ONLY on a MISTAKE, where yᵢ·(w_k·xᵢ) ≤ 0 — so the cross term 2η·yᵢ(w_k·xᵢ) is non-positive and can be dropped. The remaining term is bounded by η²·‖xᵢ‖² ≤ η²R² because every point lies inside the data radius R = max‖x‖. Correct points never touch w, so this single-step bound holds for every update.',
      },
      {
        latex: '\\|w_k\\|^2 \\le k\\, \\eta^2 R^2 \\qquad (\\|w_0\\| = 0; \\; \\text{sum the recursion over } k \\text{ updates})',
        justification: 'Telescope the step-1 recursion from the zero start w₀ = 0: each of the first k updates adds at most η²R² to the squared norm, so ‖w_k‖² ≤ k·η²R² — the norm grows like O(√k), NOT like the loose k·ηR triangle bound. This tighter bound is exactly what makes the sandwich in step 4 cancel cleanly.',
      },
      {
        latex: 'w^* \\cdot w_k \\ge \\eta\\, \\gamma \\, \\|w^*\\| \\cdot k',
        justification: 'Let w* be the separator the rule converges to and γ its geometric margin, so yᵢ(w*·xᵢ+b*) ≥ γ‖w*‖ for every point. Each update adds η·yᵢ·xᵢ to w_k, contributing w*·(η·yᵢ·xᵢ) = η·yᵢ·(w*·xᵢ) ≥ η·γ‖w*‖ — so after k updates the dot product w*·w_k grew by at least η·γ‖w*‖ per update.',
      },
      {
        latex: '(\\eta\\gamma\\|w^*\\|\\, k)^2 \\le (w^* \\cdot w_k)^2 \\le \\|w^*\\|^2\\, \\|w_k\\|^2 \\le \\|w^*\\|^2 \\, k\\, \\eta^2 R^2 \\;\\Rightarrow\\; k \\le \\left(\\frac{R}{\\gamma}\\right)^2',
        justification: 'The LEFT inequality squares the step-3 lower bound (both sides are non-negative, so squaring preserves the order). The RIGHT side chains Cauchy–Schwarz, (w*·w_k)² ≤ ‖w*‖²‖w_k‖², with the step-2 norm bound ‖w_k‖² ≤ k·η²R². The chain forces (ηγ‖w*‖k)² ≤ ‖w*‖²kη²R²; cancelling the common factor η²‖w*‖²k leaves γ²k ≤ R², i.e. k ≤ (R/γ)² — the Novikoff bound, with γ the GEOMETRIC margin of the separator. Measured on the default seed: (2.2446/0.0472)² ≈ 2261 while the run took 4 updates — the bound is a worst-case guarantee, not a prediction.',
      },
      {
        latex: '\\text{separable } \\Rightarrow\\; \\gamma > 0 \\;\\Rightarrow\\; k \\le \\left(\\frac{R}{\\gamma}\\right)^2 < \\infty \\;\\Rightarrow\\; \\text{the rule terminates}',
        justification: 'The key inference: separability makes γ > 0 feasible, which makes the bound FINITE — the perceptron cannot cycle forever on separable data. Contrapositive: a run that provably repeats a weight state is CERTIFICATE that the data is not linearly separable. The simulator reports non-convergence honestly — an exact cycle is the definitive certificate; the oscillation cap alone only says the run did not settle within the budget (a separable draw could in principle need more updates than the cap).',
      },
    ],
    derivedFrom: ['perceptron-convergence-bound', 'perceptron-geometric-margin'],
  },
  {
    id: 'perceptron-eta-invariance',
    title: 'Why the Learning Rate η is Irrelevant to the Classic Rule',
    steps: [
      {
        latex: 'w_k = \\eta \\, v_k \\qquad \\text{(factor } \\eta \\text{ out of the initial zero state and every update)}',
        justification: 'With zero initialization, w₀ = η·v₀ = 0 for any η. Every update w ← w + η·yᵢ·xᵢ becomes η·v ← η·(v + yᵢ·xᵢ): the η factor survives each step, so the sequence of v-weights is INDEPENDENT of η.',
      },
      {
        latex: 'y_i\\,(w \\cdot x_i + b) = \\eta \\cdot y_i\\,(v \\cdot x_i + c) \\;\\Rightarrow\\; \\text{sign } y_i\\, s_i \\text{ is η-independent}',
        justification: 'Scores scale by η, so the MISTAKE CONDITION yᵢ·s(xᵢ) ≤ 0 is evaluated identically for every η > 0: the exact same points trigger the exact same v-updates in the exact same order. Measured: η = 1 and η = 0.5 both make exactly 4 updates on the default seed.',
      },
      {
        latex: 'w^{\\text{final}}(\\eta) = \\eta \\, w^{\\text{final}}(1)',
        justification: 'The final weight vector is η times the η = 1 solution — measured ‖w‖ = 2.7407 at η = 1 vs 1.3704 at η = 0.5 (exact ×½). The DECISION boundary (the zero set of the score) is unchanged by the scaling. Tuning η is therefore pointless in the classic perceptron — a sharp contrast with SGD-family algorithms.',
      },
      {
        latex: '\\eta \\to 1000: \\quad w^{\\text{final}} \\text{ scales ×1000, boundary identical — but numerically} \\; w, b \\text{ lose relative precision}',
        justification: 'The invariance is algebraic; in floating point a huge η inflates ‖w‖ (measured ×1000 at η = 1000) and the boundary still separates, but the weights dwarf the feature scale — the parameter validation blocks η ≥ 1000 as a numerical risk for exactly this reason.',
      },
    ],
    derivedFrom: ['perceptron-eta-invariance'],
  },
];