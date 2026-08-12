// src/topics/svm-hard-margin/failures.ts
// Every numeric claim below is measured against the simulator:
//   - F1 (margin 0.5, noise 1.5, seed 42): NO separable seed in the 200-seed
//     search — the run fails cleanly via telemetry (converged: false).
//   - F2 (margin 1.0, noise 0.9, seed 42): seeds 42–47 are all non-separable;
//     the search lands on seed 48 → margin 0.633, ½‖w‖² = 4.99, 3 SVs.
//   - F3 (margin 0.5, noise 0.45, seed 42): lands on seed 48 → margin 0.316,
//     ½‖w‖² = 19.97, 3 SVs — the razor-thin bottom of the slider range.
import type { FailureDemo } from '../../engine/types';

export const svmFailureDemos: FailureDemo[] = [
  {
    id: 'svm-fail-nonseparable',
    title: 'Non-separable data: the hard-margin problem has NO solution',
    scenario: 'non-separable',
    params: { nPerClass: 12, margin: 0.5, noise: 1.5, seed: 42 },
    narration: 'With separation margin 0.5 and spread σ = 1.5 the two Gaussian clusters overlap massively (σ is three times the separation), so the convex hulls intersect and NO straight line separates them. The simulator searches 200 deterministic seeds and every draw is non-separable: the run fails cleanly via telemetry with an honest message instead of producing a bogus separator.',
    whyItBreaks: 'The primal constraints yᵢ(w·xᵢ+b) ≥ 1 are INFEASIBLE — no (w, b) can put every class-0 point on the −1 side and every class-1 point on the +1 side. The margin 2/‖w‖ is undefined because no valid band exists. The fix is the soft-margin SVM: introduce slack variables ξᵢ ≥ 0, relax the constraints to yᵢ(w·xᵢ+b) ≥ 1 − ξᵢ, and add a penalty C·Σξᵢ to the objective — then every dataset, separable or not, yields a solution (future wave).',
  },
  {
    id: 'svm-fail-outlier',
    title: 'A single boundary outlier collapses the margin',
    scenario: 'outliers',
    params: { nPerClass: 12, margin: 1.0, noise: 0.9, seed: 42 },
    narration: 'Raise the spread to σ = 0.9 with separation 1.0: seeds 42–47 all produce overlapping (non-separable) draws — one or two points from each cluster drift across the midline. The bounded seed search deterministically lands on seed 48, where the draw barely separates: the max-margin solution has margin 0.633 with THREE support vectors crammed into a thin band, and the objective balloons to ½‖w‖² = 4.99 (vs 1.228 at the default separation).',
    whyItBreaks: 'Support vectors are the extremes of the data, so the closest pair of points across the gap sets the entire margin. A single near-boundary outlier is the closest point, so it alone shrinks the margin to half its segment length — and with tight clusters the band is defined by three touching points, not two. Hard margin has no safety valve: it must obey every point exactly, and outliers dictate the geometry. Soft margin or removing the outlier are the fixes.',
  },
  {
    id: 'svm-fail-tight-margin',
    title: 'Razor-thin margin at the bottom of the separation slider',
    scenario: 'tiny-margin',
    params: { nPerClass: 12, margin: 0.5, noise: 0.45, seed: 42 },
    narration: 'At the slider\'s minimum separation (margin 0.5) with σ = 0.45 the clusters barely avoid overlapping: seeds 42–47 are non-separable, and the search lands on seed 48. The surviving separator is a hairline — margin 0.316 and ½‖w‖² = 19.97, sixteen times the default objective — with 3 support vectors squeezed together.',
    whyItBreaks: 'The margin is set by the gap between the closest cross-class points; when the clusters are almost touching, that gap is tiny, so ‖w‖ = 2/margin explodes and the objective ½‖w‖² = 2/margin² blows up quadratically. The separator still classifies every point correctly (train error 0 — hard margin guarantees it), but its robustness is gone: a perturbation of ~0.16 units flips a support vector to the wrong side. This is the geometric reason soft margin and regularization exist.',
  },
];
