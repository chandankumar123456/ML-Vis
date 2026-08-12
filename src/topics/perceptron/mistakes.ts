// src/topics/perceptron/mistakes.ts
import type { Mistake } from '../../engine/types';

export const perceptronMistakes: Mistake[] = [
  {
    id: 'perceptron-max-margin',
    pattern: 'Believing the perceptron finds the "best" or max-margin separating line',
    example: '\\text{perceptron stops at ANY separator; } \\gamma = 0.047 \\;\\text{(measured)} \\;\\neq\\; \\gamma_{\\max}',
    whyWrong: 'The perceptron stops at the FIRST separator its update order produces. On the default seed it converges after 4 updates with a tiny clearance γ = 0.0472 — the SVM (same data) would maximize that clearance. The perceptron has no notion of margin at all: it only knows "mistake" vs "no mistake". Training error 0 is its only criterion, and that criterion admits infinitely many lines.',
    gateTrap: true,
    relatedConcept: 'perceptron-convergence-bound',
  },
  {
    id: 'perceptron-learning-rate',
    pattern: 'Tuning the learning rate η to make the perceptron converge faster or at all',
    example: '\\eta = 0.01 \\;\\Rightarrow\\; \\text{same 4 updates; } \\; \\eta = 0.5 \\;\\Rightarrow\\; \\text{same 4 updates}',
    whyWrong: 'The classic fixed-increment rule is η-INVARIANT: the mistake condition y·(w·x + b) ≤ 0 is unchanged when all weights scale by η (and zero init starts from the η-independent zero state). Measured: η = 1 and η = 0.5 both take exactly 4 updates on the default seed, with final weights scaled by exactly η. The learning rate changes the SCALE of the final weights, nothing else — spending effort tuning it is wasted time (a sharp contrast with SGD-family algorithms).',
    gateTrap: true,
    relatedConcept: 'perceptron-eta-invariance',
  },
  {
    id: 'perceptron-no-loss',
    pattern: 'Treating mistakesPerEpoch (or training error) as a loss the perceptron minimizes',
    example: '\\text{no } L(w) \\text{ exists: updates fire on mistakes, not on gradients}',
    whyWrong: 'The perceptron is a mistake-correction rule, not an optimizer. There is no differentiable loss whose gradient drives the updates (the 0-1 loss is piecewise constant — gradient 0 almost everywhere), and the rule never shrinks the weights, so no regularization term hides inside either. mistakesPerEpoch is a DIAGNOSTIC metric the simulator plots honestly — the loss-curve layer is titled "perceptron has NO loss function" because plotting it as a loss would be a lie.',
    gateTrap: true,
    relatedConcept: 'perceptron-update',
  },
  {
    id: 'perceptron-nonseparable-convergence',
    pattern: 'Expecting the perceptron to converge on overlapping/non-separable data',
    example: '\\text{separable: false, seed 42 } \\Rightarrow\\; \\text{180 updates, never settles (measured)}',
    whyWrong: 'Novikoff\'s theorem REQUIRES linear separability (γ > 0). On non-separable data the rule provably cycles (classical cycling theorem) — the weight state keeps changing forever. The simulator demonstrates this honestly: on the overlapping-cloud demo no exact float cycle fires within 5000 updates for any seed 0–60, so the deterministic OSCILLATION_CAP (180 updates → 181 snapshots, failedAtStep 181) reports the non-convergence with an honest oscillation message instead of a fake "converged" state.',
    gateTrap: true,
    relatedConcept: 'perceptron-convergence-bound',
  },
  {
    id: 'perceptron-dropping-bias',
    pattern: 'Dropping the bias b (writing w·x = 0) or forgetting the ±1 label form',
    example: 'w \\cdot x = 0 \\;\\text{(wrong: forces the boundary through the origin)} \\qquad y \\in \\{0,1\\} \\;\\Rightarrow\\; \\text{updates skew one class}',
    whyWrong: 'Without b every boundary passes through the origin, which is almost never the separating line for real data — the bias update b ← b + η·yᵢ (measured first update: b = −1 from the class-0 point d0) is what lets the boundary sit anywhere. And the symmetric update w ← w + η·yᵢ·xᵢ only makes sense with ±1 labels: with y ∈ {0, 1} the class-0 points would never move the weights. Both omissions silently break the geometry.',
    gateTrap: true,
    relatedConcept: 'perceptron-score',
  },
  {
    id: 'perceptron-bound-as-prediction',
    pattern: 'Reading (R·‖w*‖/γ)² as a performance prediction or tight runtime estimate',
    example: '\\left(\\frac{R\\,\\|w^*\\|}{\\gamma}\\right)^2 \\approx 16982 \\;\\text{vs measured 4 updates}',
    whyWrong: 'Novikoff\'s bound is a worst-case guarantee, and for practical draws it is extremely loose: on the default seed it evaluates to ≈ 16982 updates while the run takes 4 (seed 7: 23). The bound\'s real content is QUALITATIVE — separability ⇒ finitely many updates — plus the honest warning that tiny margins γ make the bound huge. Using it to predict runtime or to compare algorithms misreads the mathematics.',
    gateTrap: false,
    relatedConcept: 'perceptron-convergence-bound',
  },
];