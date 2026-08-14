// src/topics/decision-trees-regression/failures.ts
// Failure demos — every number below is MEASURED from an actual computeRun on
// the given params (the HONESTY rule: no fabricated values). Params use the
// test-only `xys` override for hand-crafted datasets (whole dataset = train).
import type { FailureDemo } from '../../engine/types';

export const dtrFailureDemos: FailureDemo[] = [
  {
    id: 'dtr-fail-extrapolation',
    title: 'Extrapolation: the prediction is a flat constant outside the observed range',
    scenario: 'extrapolation',
    params: { xys: '[[0,0],[1,2],[2,2],[3,2.5],[4,1],[5,1.2]]', maxDepth: 3, minLeaf: 1, seed: 42 },
    narration: 'The tree fits the six points exactly (train error 0, 5 leaves whose constants are 0.00, 1.00, 1.20, 2.00, 2.50). But the predictions beyond the data are the leaf constants of the edge regions: x = −2 → 0.00, and x = 5.5, 7 → 1.20. The fitted step function does not continue any trend — it simply stops, flat, at the last observed values.',
    whyItBreaks: 'A regression tree is a piecewise-constant model over the observed feature range; there is no slope to extrapolate. Where a linear regression would keep climbing with its fitted slope, the tree guesses "the last leaf\'s mean" forever. Use regression trees only for interpolation inside the training domain, or combine with a trend model for the tails.',
  },
  {
    id: 'dtr-fail-deep-variance',
    title: 'Deep trees overfit: train error → 0 while test error blows up',
    scenario: 'high-variance',
    params: { n: 25, noise: 0.8, maxDepth: 6, minLeaf: 1, seed: 42 },
    narration: 'Measured on this run (n = 25, σ = 0.8, depth 6, min-leaf 1): train error falls monotonically 1.394 → 1.049 → 0.547 → 0.164 → 0.056 → 0.000 (17 leaves, every leaf memorizing 1–2 points). The held-out test error bottomed at 0.147 right after the FIRST split, then swung through 0.38 → 0.42 → 0.64 as the tree kept splitting — ending at 0.415. The two loss curves diverge sharply: perfect memorization, poor generalization.',
    whyItBreaks: 'Each additional split trades bias for variance: with min-leaf 1 the leaves shrink to single points, so the fitted step function interpolates the training noise (the jagged staircase in the scatter view). Test points landing between the noise spikes get the wrong constant. The fix the tree itself offers: stop early (maxDepth / minLeaf) or prune — cost-complexity pruning collapses leaves whose added splits do not pay off out-of-sample.',
  },
  {
    id: 'dtr-fail-outlier-isolation',
    title: 'A single dominant outlier hijacks a whole region',
    scenario: 'outlier',
    params: { xys: '[[0,1],[1,1.2],[2,1.1],[3,1.3],[4,8],[5,1.2]]', maxDepth: 3, minLeaf: 1, seed: 42 },
    narration: 'Measured: the root splits at x < 3.5 — SSE drops 39.04 → 23.17 (reduction 15.87) to isolate the point (4, 8). The tree then carves it its own leaf with constant 8.00, so the whole region x ∈ [3.5, 4.5) predicts 8 even though the underlying values around x = 4 are ≈ 1.2 (prediction at x = 4.2 is 8.00; at x = 4.9 it snaps back to 1.20). The squared loss is what forces this: one residual of 5.7² (8 − 2.3) dominates the small residuals of the well-behaved points.',
    whyItBreaks: 'SSE squares residuals, so a single extreme target dominates the split search and the leaf means. The tree "spends" splits isolating the outlier instead of modeling the signal — a 6-point dataset wastes half its structure on one bad point. Robust alternatives: cap extreme targets (winsorize), use MAE-based criteria, or lower minLeaf tolerance only after cleaning the data.',
  },
  {
    id: 'dtr-fail-tiny-spurious',
    title: 'Tiny + noisy data: the first split is pure noise',
    scenario: 'small-data',
    params: { n: 8, noise: 1.5, maxDepth: 6, minLeaf: 1, seed: 42 },
    narration: 'Measured (n = 8, σ = 1.5, seed 42): the global mean predicts with test error 0.80 — then the FIRST split (x < 0.883) drives train error down to 1.279 but jacks test error up to 2.144, and the fully grown 5-leaf tree reaches train error 0.000 while test error sits at 2.00. The split that "found structure" actually locked onto noise: with 2–3 points per candidate region, any threshold looks like a pattern.',
    whyItBreaks: 'With tiny samples the variance of the split search is enormous: the greedy criterion maximizes a noisy in-sample SSE reduction, so the chosen threshold is essentially random. The tree confidently overfits noise (train 0.000, test ≈ 2.0 — the test error is 2.5× the naive mean\'s). Minimum leaf sizes, depth caps, or a larger sample are the only honest fixes.',
  },
];
