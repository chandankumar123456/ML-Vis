// src/topics/perceptron/questions.ts
// Simulation anchors referenced below (default: nPerClass 20, margin 1.2, noise
// 0.5, η 1, zero init, seed 42):
//   4 updates, 6 snapshots, final w = (2.7135, 0.3850), b = 0, ‖w‖ = 2.7407,
//   γ = 0.0472; the first update (zero init) fires on point d0 = (−2.8825,
//   −0.1092) — a class-0 point (y = −1), giving w = (2.8825, 0.1092), b = −1.
//   Seed 7: 23 updates. Non-separable (separable: false, seed 42): cap at 180
//   updates → 181 snapshots, final accuracy 0.725, mistakesPerEpoch 4.
import type { Question } from '../../engine/types';

export const perceptronQuestions: Question[] = [
  {
    id: 'perceptron-001',
    mode: 'nat',
    prompt: 'The perceptron starts at w = (0, 0), b = 0 (zero init). The first scan finds point d0 = (−2.88, −0.11) — every score is 0 so it counts as a mistake. One update with η = 1 fires and the simulator reads w = (2.88, 0.11), b = −1. What is the label y of d0 in ±1 form? (Enter −1 or +1.)',
    answer: -1,
    tolerance: 0.01,
    explanation: 'The update is w ← w + η·y·x and b ← b + η·y. From b: −1 = 0 + 1·y ⇒ y = −1. Check w: (−2.88)(−1) = 2.88 and (−0.11)(−1) = 0.11 — matching the measured first update. So d0 is a class-0 point (the simulator maps class 0 → −1).',
    concepts: ['perceptron', 'update rule', 'labels'],
    difficulty: 2,
    tags: ['numerical', 'formula'],
  },
  {
    id: 'perceptron-002',
    mode: 'nat',
    prompt: 'On the default separable draw the perceptron with η = 1 converges in exactly 4 updates. You rerun the identical data with η = 0.5. How many updates does it take now? (Enter the number.)',
    answer: 4,
    tolerance: 0,
    explanation: 'The classic fixed-increment rule is η-INVARIANT: the mistake condition y·(w·x + b) ≤ 0 is preserved when every weight is scaled by η (the scores just scale by η), and zero init makes the whole update sequence scale-invariant. The simulator measures exactly 4 updates for both η = 1 and η = 0.5 — the learning rate does not affect the mistake count at all.',
    concepts: ['perceptron', 'learning rate', 'eta-invariance'],
    difficulty: 3,
    tags: ['numerical', 'trap'],
  },
  {
    id: 'perceptron-003',
    mode: 'gate-mcq',
    prompt: 'GATE-style: which statement about the perceptron learning algorithm is correct?',
    options: [
      'For linearly separable data with a fixed learning rate, the algorithm is guaranteed to find a separating hyperplane in finitely many updates (bounded by (R/γ)²)',
      'For linearly separable data it always finds the max-margin separating hyperplane',
      'It converges only if the learning rate is decayed over time',
      'It is guaranteed to converge for any training data, separable or not',
    ],
    answer: 'A',
    explanation: 'Novikoff\'s theorem: if some (w*, b*) separates the data (so the geometric margin γ > 0 exists) then the fixed-increment rule converges after at most (R·‖w*‖/γ)² updates. Measured on the default seed: 4 updates (seed 7: 23), against a bound of ≈ 16982. The result is the perceptron\'s one great guarantee — but it is only as strong as the separability assumption.',
    trapExplanations: {
      B: 'The perceptron converges to SOME separator — whichever the update order produces. Maximizing the margin is the SVM\'s job, and the perceptron never does it (its final γ = 0.047 is far from the max-margin value).',
      C: 'A decaying rate changes the final weights\' scale but not the mistake sequence; the classic rule has a single fixed η and still converges on separable data.',
      D: 'On non-separable data the rule cycles (the classical cycling theorem) — the simulator demonstrates this honestly with its oscillation cap.',
    },
    concepts: ['perceptron', 'convergence', 'novikoff'],
    difficulty: 2,
    tags: ['conceptual'],
  },
  {
    id: 'perceptron-004',
    mode: 'conceptual-mcq',
    prompt: 'You run the perceptron on two overlapping Gaussian clusters (separable: false, seed 42). What does the algorithm actually do, and what does the simulator report?',
    options: [
      'The weights cycle forever without settling — no hyperplane can fit the data, so mistakes keep firing; the simulator caps the run at 180 updates and reports an honest oscillation failure',
      'It converges to the best possible hyperplane, then stays there with low, constant error',
      'It converges quickly and misclassifies the exact same few points forever',
      'It stops after the first full epoch and reports a non-separable warning',
    ],
    answer: 'A',
    explanation: 'The classical cycling theorem: on non-separable data the perceptron never settles — there is no (w, b) with y·s > 0 everywhere, so mistakes fire indefinitely. The exact float state drifts (measured: NO exact cycle within 5000 updates on any seed 0–60), so the simulator terminates with its deterministic OSCILLATION_CAP: 180 updates → 181 snapshots, failedAtStep 181, failureReason mentioning oscillation, final accuracy 0.725 and mistakesPerEpoch 4.',
    trapExplanations: {
      B: 'There is no "best possible" for a cycling rule: the boundary keeps being dragged by the latest mistake and never reaches a fixed point.',
      C: 'The mistaken points are not fixed — as w drifts, different points take turns triggering updates; the mistake count never reaches 0.',
      D: 'The perceptron has no "stop after one epoch" behavior; runs are capped by the simulator, not ended by the algorithm.',
    },
    concepts: ['perceptron', 'non-separable', 'oscillation'],
    difficulty: 3,
    tags: ['trap', 'visual'],
  },
  {
    id: 'perceptron-005',
    mode: 'matrix',
    prompt: 'Match each perceptron quantity to its role in the default run (class 0 → y = −1, class 1 → y = +1).',
    options: [
      's(x) = w·x + b',
      'w ← w + η·y·x',
      'mistakesPerEpoch',
      'yᵢ·s(xᵢ) ≤ 0',
    ],
    answer: ['the linear score — its SIGN decides the class (s > 0 → class 1)', 'one online correction per mistake; pulls the boundary toward the mistaken point', 'diagnostic metric — mistakes per completed epoch; perceptron has NO loss to plot', 'the trigger condition — the point that fires the next update'],
    explanation: 'The score is read-only for the bare perceptron: it only determines whether an update fires. Correct points are skipped, mistaken ones trigger the single update per step. Since no loss is minimized, mistakesPerEpoch (measured 0 at convergence, 4 on the capped non-separable run) is an honest diagnostic rather than a loss value.',
    concepts: ['perceptron', 'online learning', 'metric'],
    difficulty: 2,
    tags: ['matrix'],
  },
  {
    id: 'perceptron-006',
    mode: 'visual',
    prompt: 'Run the default simulation (nPerClass 20, margin 1.2, noise 0.5, η 1, zero init, seed 42) and scrub snapshot by snapshot. Which statement matches the 6-snapshot trajectory?',
    options: [
      'Snapshots 1–4 each add exactly one update (a mistake fires and the weight arrow jumps); snapshot 5 re-emits the same converged state with a clean sweep — the boundary no longer moves',
      'All 6 snapshots move the boundary; the perceptron keeps refining until it reaches the max-margin line',
      'The first 4 snapshots move the boundary, then updates continue past snapshot 5 forever',
      'Only snapshot 0 moves the boundary; everything after it is a duplicate',
    ],
    answer: 'A',
    explanation: 'Step semantics: one mistake-driven update per step, so snapshots 1–4 move the weight arrow once each (measured updates 1,2,3,4; epoch marks {3: 2 mistakes, 4: 0}). When an epoch completes with zero updates the run converges: the simulator emits a converged snapshot with the SAME weights (re-emission) and timeline stage "Converge" — snapshot 5 duplicates the final state, and the next step returns null.',
    trapExplanations: {
      B: 'The perceptron never maximizes the margin — it stops at the FIRST separator it finds (here after 4 updates); the max-margin line would take many more careful steps that the rule does not take.',
      C: 'Convergence means a full clean epoch: snapshot 5 is the last one; a step after it returns null (two-phase termination).',
      D: 'Steps 1–4 genuinely move w — the wrong claim would only be true of the converged duplication.',
    },
    concepts: ['perceptron', 'visualization', 'convergence'],
    difficulty: 2,
    tags: ['visual'],
  },
  {
    id: 'perceptron-007',
    mode: 'gate-mcq',
    prompt: 'GATE-style: a student claims "the perceptron training loop is gradient descent on the training error". Which response is correct?',
    options: [
      'The update w ← w + η·yᵢ·xᵢ is a MISTAKE-CORRECTION rule, not a gradient step — the classic perceptron minimizes no loss function at all',
      'It is exactly SGD with the 0-1 loss, whose gradient is the update vector',
      'It is gradient descent on the squared error (w·xᵢ − yᵢ)²',
      'It is coordinate descent on the per-point hinge loss max(0, 1 − yᵢ·sᵢ)',
    ],
    answer: 'A',
    explanation: 'The update is applied only when a mistake fires (yᵢ·sᵢ ≤ 0) and only for that point — unlike any gradient method it is triggered by a discrete condition, not by a gradient magnitude, and it never shrinks the weights (no regularization). The 0-1 loss has zero gradient almost everywhere, so it cannot drive GD. Measured side effect of the true rule: η-invariance — real SGD depends on η; the perceptron does not. (Rosenblatt\'s original rule is a correction rule; later "perceptron as SGD" readings add structure the classic rule does not have.)',
    trapExplanations: {
      B: 'The 0-1 loss is piecewise constant — its gradient is 0 almost everywhere, so SGD on it could never move the weights at all.',
      C: 'Squared error would update on EVERY point proportionally to the error (and is smooth), whereas the perceptron updates only on mistakes by a fixed increment.',
      D: 'The hinge loss has a gradient even for correctly classified points inside the margin — the perceptron is exactly the η-scaling of the subgradient ONLY at the mistake boundary, not a descent on hinge loss; and it has no max-margin bias.',
    },
    concepts: ['perceptron', 'loss function', 'gradient descent', 'trap'],
    difficulty: 4,
    tags: ['trap', 'conceptual'],
  },
];