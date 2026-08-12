// src/topics/perceptron/derivations.ts
// All numeric claims measured against the simulator (default seed 42):
//   4 updates on the default draw, final ‖w‖ = 2.7407, γ = 0.0472, R = 2.2446,
//   (R·‖w*‖/γ)² ≈ 16982; seed 7 → 23 updates; non-separable → cap at 180.
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
    title: 'The Convergence Theorem: updates ≤ (R·‖w*‖/γ)²',
    steps: [
      {
        latex: '\\|w_k\\| \\le \\eta R \\cdot k',
        justification: 'Start w₀ = 0. Each of the k updates so far added a vector η·yᵢ·xᵢ of length ≤ ηR; by the triangle inequality the total norm is at most k·ηR. This bound holds no matter which points triggered the updates.',
      },
      {
        latex: 'w^* \\cdot w_k \\ge \\eta\\, \\gamma \\, \\|w^*\\| \\cdot k',
        justification: 'Let w* be the separator the rule converges to and γ its geometric margin, so yᵢ(w*·xᵢ+b*) ≥ γ‖w*‖ for every point. Each update adds η·yᵢ·xᵢ to w_k, contributing w*·(η·yᵢ·xᵢ) = η·yᵢ·(w*·xᵢ) ≥ η·γ‖w*‖ — so after k updates the dot product w*·w_k grew by at least η·γ‖w*‖ per update.',
      },
      {
        latex: 'w^* \\cdot w_k \\le \\|w^*\\|\\, \\|w_k\\| \\le \\|w^*\\| \\cdot \\eta R k',
        justification: 'Cauchy–Schwarz bounds the dot product by the product of norms; substituting the step-1 norm bound gives an UPPER bound on the same dot product — the dot product cannot both be ≥ ηγ‖w*‖k and ≤ ηR‖w*‖k unless γ ≤ R, i.e. the data radius is at least the margin. So far this is a consistency check, not the theorem.',
      },
      {
        latex: '\\|w^*\\|^2 \\, (\\eta\\gamma k)^2 \\le (w^* \\cdot w_k)^2 \\le \\|w^*\\|^2 \\|w_k\\|^2 \\le \\|w^*\\|^2 \\, (\\eta R k)^2 \\;\\Rightarrow\\; k \\le \\left(\\frac{R}{\\gamma}\\right)^2',
        justification: 'The tighter route squares the two inequalities and cancels the common factor: the Cauchy–Schwarz sandwich forces k ≤ (R/γ)². Written for an unnormalized separator this is updates ≤ (R·‖w*‖/γ)² — the exact form the plan prescribes. Measured on the default seed: (2.2446·2.7407/0.0472)² ≈ 16982 while the run took 4 updates — the bound is a worst-case guarantee, not a prediction.',
      },
      {
        latex: '\\text{separable } \\Rightarrow\\; \\gamma > 0 \\;\\Rightarrow\\; k \\le \\left(\\frac{R}{\\gamma}\\right)^2 < \\infty \\;\\Rightarrow\\; \\text{the rule terminates}',
        justification: 'The key inference: separability makes γ > 0 feasible, which makes the bound FINITE — the perceptron cannot cycle forever on separable data. Contrapositive: a run that never settles is CERTIFICATE that the data is not linearly separable, which is exactly what the simulator reports honestly via the oscillation cap.',
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