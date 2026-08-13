// src/topics/perceptron/failures.ts
// Every numeric claim measured against the simulator:
//   - F1 (separable: false, nPerClass 20, margin 1.2, noise 0.5, seed 42): the
//     classic cycling failure — 180 updates, 181 snapshots, failedAtStep 181,
//     failureReason reports oscillation; NO exact float cycle fires within 5000
//     updates on any seed 0–60, so the OSCILLATION_CAP does the honest work.
//   - F2 (separable: true, margin 0.9, noise 0.6): the toggle lies — the RAW
//     draw is genuinely non-separable. Measured: seeds 42/7/123 never converge
//     (cap at 180); seeds 1/99 converge at 51/60 updates. validateParams warns.
//   - F3 (η = 1000): measured — 4 updates, boundary identical, but final ‖w‖
//     scales ×1000 (≈ 2741 vs 2.741): numerical-scale failure; validateParams
//     flags η ≥ 1000 as a risk.
//   - F4 (imbalanced classes: nPerClass 36, nClass1 6, margin 0.8, noise 0.6,
//     seed 42): the cap fires at 180 — final state 36/36 majority correct but
//     only 1/6 minority correct (37/42 = 0.881 overall); the boundary sits at
//     x ≈ +1.78, past the minority cluster; 89 of the 180 updates fire on just
//     3 of the 6 minority points.
import type { FailureDemo } from '../../engine/types';

export const perceptronFailureDemos: FailureDemo[] = [
  {
    id: 'perceptron-fail-nonseparable',
    title: 'Overlapping clusters: the perceptron cycles forever',
    scenario: 'non-separable',
    params: { nPerClass: 20, margin: 1.2, noise: 0.5, eta: 1, init: 'zero', seed: 42, separable: false },
    narration: 'With separable = false the class-0 cluster stays centered at −margin and only the class-1 cluster moves to the origin — heavily overlapping clouds, not linearly separable. Mistakes keep firing forever: the weight arrow keeps jumping as each new mistake drags the boundary back. The simulator demonstrates the honest end state: 180 updates, 181 snapshots, and run telemetry reports failedAtStep 181 with an oscillation reason — final accuracy settles at 0.725 with mistakesPerEpoch 4, never reaching a clean sweep.',
    whyItBreaks: 'The cycling theorem: on non-separable data no (w, b) separates the classes, so every state must eventually trigger another mistake. Notably the exact float weight-state cycle — the classical proof\'s mechanism — is almost never the observed one: the (w₁, w₂, b, scan-pos) state drifts like a bounded random walk (measured: no exact repeat within 5000 updates on any seed 0–60). The simulator therefore terminates non-separable runs with the deterministic OSCILLATION_CAP and an honest failure message — the perceptron does not converge, and pretending it does would be a lie.',
  },
  {
    id: 'perceptron-fail-harsh-separable',
    title: 'The "separable data" toggle does not guarantee separability',
    scenario: 'false-positive-separable',
    params: { nPerClass: 20, margin: 0.9, noise: 0.6, eta: 1, init: 'zero', seed: 42, separable: true },
    narration: 'With separable = true the clusters are drawn around shifted centers — but when separation (0.9) is small relative to spread (0.6), individual draws still overlap. Measured: seed 42 (the default) never converges (180-update cap), as do seeds 7 and 123; only seeds 1 and 99 among the test set converge, taking 51–60 updates. The toggle controls the GENERATOR, not the outcome: the learner must still face the honest truth that this particular draw is not linearly separable.',
    whyItBreaks: 'Separability is a property of the realized SAMPLE, not of the sampling recipe. Novikoff\'s theorem needs γ > 0 for the actual 40 points; two Gaussian clouds with σ ≈ 2/3 of the center separation produce overlapping draws with positive probability on every seed. The perceptron reliably exposes the lie: it should converge on separable data and cycle on non-separable data, so a cap-terminated run is a precise, honest witness that the draw does not separate — the module even warns via validateParams when margin/noise < 2.4.',
  },
  {
    id: 'perceptron-fail-eta-scale',
    title: 'A huge learning rate destroys the numbers while leaving the boundary intact',
    scenario: 'huge-learning-rate',
    params: { nPerClass: 20, margin: 1.2, noise: 0.5, eta: 1000, init: 'zero', seed: 42, separable: true },
    narration: 'Set η = 1000 and rerun the default draw. Because the rule is η-invariant the TRAJECTORY is untouched: the same 4 updates fire, the same points trigger them, and the boundary is pixel-identical to the η = 1 run. But the weights themselves are scaled by 1000: the final weight vector reads ‖w‖ ≈ 2741 instead of 2.741 — the numbers dwarf the feature scale and lose relative precision. The parameter validation flags η ≥ 1000 as a numerical risk.',
    whyItBreaks: 'η-invariance is algebraic (every weight is η·v after factoring η out of the zero init), but floating point is not: w₂ ≈ 385 on a b = 0 state and features of order 1 is harmless, while w₂ ≈ 385000 at η = 1000 puts the fine structure of the decision boundary into the tail bits of the representation. The boundary survives — until it stops being representable. The classic rule therefore needs no η for CONVERGENCE; the only reason η matters at all is numerical hygiene, which is why the validator blocks the extreme range.',
  },
  {
    id: 'perceptron-fail-imbalance',
    title: 'Imbalanced classes: the boundary is dragged into the minority cluster',
    scenario: 'imbalanced-classes',
    params: { nPerClass: 36, nClass1: 6, margin: 0.8, noise: 0.6, eta: 1, init: 'zero', seed: 42, separable: true },
    narration: 'A 6:1 class imbalance — 36 class-0 points at −0.8 against only 6 class-1 points at +0.8 (margin/noise = 1.3, a draw the validator warns may not separate). Measured: the run does NOT settle — the OSCILLATION_CAP fires at 180 updates (181 snapshots, failedAtStep 181) — and at the cap the boundary has been dragged to x ≈ +1.78, past the whole minority cluster. The majority is spotless (36/36 correct) while the minority is nearly wiped out (1 of 6 correct): overall accuracy 37/42 = 0.881 hides a minority accuracy of 0.167. Telling detail: 89 of the 180 updates fire on just 3 of the 6 minority points, yet the boundary still lands on the majority\'s side.',
    whyItBreaks: 'The perceptron carries no class prior, but its update stream is slanted by the data\'s composition: the fixed scan order always lists the 36 majority points before the 6 minority ones, so a majority mistake drags the boundary back the moment the minority pulls it across. Even with a generous share of updates spent on the minority (measured: 89 of 180, vs ~26 for a proportional 6/42 split), the cap state is a MAJORITY BOUNDARY — 5 of the 6 minority points sit misclassified while the majority is perfect. It is the textbook accuracy paradox of skewed classes: 0.881 looks healthy until you read the minority rows. Contrast the balanced non-separable demo (20:20), where errors fall on BOTH classes (0.725): imbalance concentrates every error on the minority.',
  },
];