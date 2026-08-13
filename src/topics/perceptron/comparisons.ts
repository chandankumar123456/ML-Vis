// src/topics/perceptron/comparisons.ts
import type { Comparison } from '../../engine/types';

export const perceptronComparisons: Comparison[] = [
  {
    id: 'perceptron-vs-svm',
    title: 'Perceptron vs SVM (any separator vs the max-margin separator)',
    topics: ['perceptron', 'svm-hard-margin'],
    axes: [
      {
        axis: 'Objective',
        entries: [
          { topic: 'perceptron', value: 'None — a mistake-correction rule; stops at the first separator its update order finds' },
          { topic: 'svm-hard-margin', value: 'Maximize the margin (minimize ½‖w‖²) — a unique optimum' },
        ],
      },
      {
        axis: 'Convergence / solution',
        entries: [
          { topic: 'perceptron', value: 'Converges in ≤ (R/γ)² updates IF separable (Novikoff) — measured 4 updates on the default seed, γ = 0.047 (an arbitrary separator)' },
          { topic: 'svm-hard-margin', value: 'The unique max-margin line — measured margin 1.276 on the SVM default, the same data the perceptron leaves with γ = 0.047' },
        ],
      },
      {
        axis: 'Data usage',
        entries: [
          { topic: 'perceptron', value: 'Every mistake fires an update; the final w is the sum of η·yᵢ·xᵢ over all mistakes ever made' },
          { topic: 'svm-hard-margin', value: 'Only support vectors matter (αᵢ > 0); interior points contribute nothing' },
        ],
      },
      {
        axis: 'Non-separable data',
        entries: [
          { topic: 'perceptron', value: 'Cycles forever (theorem); the simulator caps it at 180 updates with honest oscillation telemetry' },
          { topic: 'svm-hard-margin', value: 'Infeasible — the hard-margin constraints cannot all hold; soft-margin is the relaxation' },
        ],
      },
    ],
    notes: [
      'The perceptron asks "does any line separate?", the SVM asks "which line separates BEST?". Same linear family, opposite selection principle.',
      'Margin-robustness is the historical motivation for SVM: the perceptron is content with a line that hugs a support vector; the SVM pulls it as far from the data as possible.',
    ],
  },
  {
    id: 'perceptron-vs-logistic',
    title: 'Perceptron vs Logistic Regression (correction rule vs probability model)',
    topics: ['perceptron', 'logistic-regression'],
    axes: [
      {
        axis: 'Update mechanism',
        entries: [
          { topic: 'perceptron', value: 'Mistake-driven: w ← w + η·yᵢ·xᵢ only when yᵢ·sᵢ ≤ 0 — fixed increment, no loss, no gradient' },
          { topic: 'logistic-regression', value: 'Gradient step on cross-entropy every point, every epoch: w ← w − η·∇L (probabilistic confidence feeds the update)' },
        ],
      },
      {
        axis: 'Output',
        entries: [
          { topic: 'perceptron', value: 'Hard class only (s > 0 → class 1); no probability, no confidence calibration' },
          { topic: 'logistic-regression', value: 'p = σ(w·x + b) ∈ (0, 1) — a calibrated probability that doubles as confidence' },
        ],
      },
      {
        axis: 'Learning-rate role',
        entries: [
          { topic: 'perceptron', value: 'η is inert (η-invariance: same 4 updates for η = 1 and η = 0.5, measured)' },
          { topic: 'logistic-regression', value: 'η directly controls step size and stability — must be tuned (too large → divergence)' },
        ],
      },
      {
        axis: 'Boundary at convergence',
        entries: [
          { topic: 'perceptron', value: 'Stops at the first separating line it stumbles on (γ = 0.047 on the default draw)' },
          { topic: 'logistic-regression', value: 'Keeps improving the fit as epochs run; boundary sits where the log-odds balance the data' },
        ],
      },
    ],
    notes: [
      'Both produce a linear boundary, and both are trained online on the same score s = w·x + b — the difference is the learning SIGNAL: mistakes only (perceptron) vs calibrated probabilities everywhere (logistic).',
      'Logistic regression is "smooth perceptron": replace the 0-1-style mistake trigger with a smooth loss and every point contributes.',
    ],
  },
  {
    id: 'perceptron-vs-knn',
    title: 'Perceptron vs k-NN (parametric learner vs lazy memorizer)',
    topics: ['perceptron', 'knn'],
    axes: [
      {
        axis: 'Training phase',
        entries: [
          { topic: 'perceptron', value: 'Explicit online training: mistakes accumulate into (w, b) — a compact 3-number model' },
          { topic: 'knn', value: 'No training at all — the "model" IS the stored dataset (lazy / instance-based learning)' },
        ],
      },
      {
        axis: 'Inference cost',
        entries: [
          { topic: 'perceptron', value: 'O(d) — one dot product per prediction; the boundary is a fixed line' },
          { topic: 'knn', value: 'O(n·d) per prediction — distance to every stored point; cost grows with the dataset' },
        ],
      },
      {
        axis: 'Model form',
        entries: [
          { topic: 'perceptron', value: 'Parametric, linear: class is the SIGN of w·x + b — cannot express curved boundaries' },
          { topic: 'knn', value: 'Non-parametric, flexible: any shape the point cloud implies (Voronoi cells); no explicit boundary' },
        ],
      },
      {
        axis: 'Data memorization',
        entries: [
          { topic: 'perceptron', value: 'Keeps only the mistake sum; old points vanish from the model once covered' },
          { topic: 'knn', value: 'Keeps every point forever — the model can answer exactly where the data was, but has no summarization' },
        ],
      },
    ],
    notes: [
      'Perceptron trades away expressiveness for a tiny model and O(1) inference; k-NN trades away training to keep the raw data. Perceptron generalizes only if the true boundary is (near-)linear; k-NN adapts to any boundary but pays per prediction.',
      'Both are classic Wave-1 baselines: the perceptron is the ancestor of deep linear layers, k-NN is the default non-parametric baseline.',
    ],
  },
  {
    id: 'perceptron-vs-sgd',
    title: 'Perceptron vs Gradient Descent (mistake-correction vs gradient descent)',
    topics: ['perceptron', 'gradient-descent'],
    axes: [
      {
        axis: 'Update trigger',
        entries: [
          { topic: 'perceptron', value: 'Mistake-driven: w ← w + η·yᵢ·xᵢ fires ONLY when yᵢ·(w·xᵢ+b) ≤ 0 — correct points are examined and skipped (measured: 4 updates on the default seed)' },
          { topic: 'gradient-descent', value: 'Gradient-driven: steps x ← x − η·∇f(x) on EVERY iteration — no mistake condition; the run advances as long as the gradient is non-zero' },
        ],
      },
      {
        axis: 'Objective',
        entries: [
          { topic: 'perceptron', value: 'No loss function is minimized — a pure correction rule; the 0-1 error it targets has gradient 0 almost everywhere, so no gradient could drive it' },
          { topic: 'gradient-descent', value: 'Minimizes an explicit loss f(x) by following −∇f; "convergence" means the gradient is ≈ 0 (a minimum), not that the data is separated' },
        ],
      },
      {
        axis: 'Learning-rate role',
        entries: [
          { topic: 'perceptron', value: 'η is inert in the classic rule (η-invariance: measured 4 updates for both η = 1 and η = 0.5) — it only scales the final weights' },
          { topic: 'gradient-descent', value: 'η IS the step size: too large → overshoot/divergence, too small → glacial progress; tuning it is the central practical skill' },
        ],
      },
      {
        axis: 'Data usage (online vs batch)',
        entries: [
          { topic: 'perceptron', value: 'Online: ONE point per update — the point whose mistake fired; the final w is the running sum of past mistakes' },
          { topic: 'gradient-descent', value: 'Batch: the gradient averages over the whole objective, so each step sees every point (full-batch); per-example "stochastic" GD is the online variant' },
        ],
      },
    ],
    notes: [
      'The perceptron update LOOKS like an SGD step but is not a gradient step: no loss, no gradient, updates only on mistakes, never shrinks the weights. GD on a smooth loss updates on every point, has a real objective, and is critically η-dependent — exactly the properties the perceptron lacks.',
      'The historical bridge: the perceptron is the ancestor of SGD-trained neural networks; modern "perceptron as SGD" readings bolt on a surrogate loss the classic rule never had.',
    ],
  },
];