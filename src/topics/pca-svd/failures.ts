// src/topics/pca-svd/failures.ts
// Failure demos — every params set is MEASURED and reproduces the described
// failure on the real module (verified in testCases.test.ts):
//   F1 missing data:  points '[[1,2],[3,4],null,[7,8]]'  → telemetry failure at
//                     step 0 ("object null is not iterable" — the missing ROW
//                     breaks before any math runs). A [null,null] row would
//                     silently coerce to 0 — narrated as the JS pitfall.
//   F2 sign flips:    rotDeg 140, seed 2 → raw v₁ = (−0.9978, 0.0667) at
//                     176.18°; the sign convention flips it to (0.9978, −0.0667).
//   F3 degenerate:    collinear '[[1,2],[3,4],[5,6],[7,8]]' → σ₁ = 6.324555320336759,
//                     σ₂ = 0, rankDeficient = 1, ratio saturated at 1e9; the run
//                     COMPLETES (rank-1 reconstruction is already exact).
import type { FailureDemo } from '../../engine/types';

export const svdFailureDemos: FailureDemo[] = [
  {
    id: 'svd-fail-missing-data',
    title: 'Missing Data Breaks the SVD — and Silently, if You Are Not Careful',
    scenario: 'missing-data',
    params: {
      n: 40, corr: 0.7, rotDeg: 30, noise: 0.15, seed: 42, rank: 2,
      points: '[[1,2],[3,4],null,[7,8]]',
    },
    narration:
      'One observation comes back as a missing row (null) instead of a point. The SVD requires a COMPLETE rectangular matrix — there is no partial information. Here the run fails cleanly at step 0: the row-to-point mapping throws before any math runs, and the telemetry records the failure instead of a garbage factorization.',
    whyItBreaks:
      'SVD needs every entry of X: the Gram matrix XᵀX sums over ALL rows, the eigenvalues λ = σ² are computed from it, and a single NaN anywhere would poison λ → σ → U and the entire reconstruction. The honest failure is thrown up front. The silent variant is the real danger: in JavaScript a row of [null, null] COERCES to [0, 0] — a valid-looking point at the origin that distorts σ₁, σ₂ and every direction without ever raising an error. Real pipelines must validate completeness BEFORE the SVD, or fall back to a missing-data method (e.g. EM / matrix completion) that the SVD itself does not provide.',
  },
  {
    id: 'svd-fail-sign-flips',
    title: 'Numerical Sign Flips: ±v Are Both Valid — and Both Wrong Without a Convention',
    scenario: 'sign-flips',
    params: {
      n: 40, corr: 0.7, rotDeg: 140, noise: 0.15, seed: 2, rank: 2,
    },
    narration:
      'This draw (rotDeg 140, seed 2) returns a raw right singular vector v₁ = (−0.9978, 0.0667) at 176.18° — the eigen-solver\'s arbitrary orientation. Flipping the sign of a singular vector never changes the factorization: σ₁u₁v₁ᵀ = σ₁(−u₁)(−v₁)ᵀ exactly. The module\'s deterministic convention (largest |component| positive) flips it to v₁ = (0.9978, −0.0667), so every run, every seed, and every comparison uses the same orientation.',
    whyItBreaks:
      'Eigenvectors are defined only up to sign, so a raw solver returns EITHER orientation arbitrarily. The failure is real: without a convention, PCA loadings and singular-vector plots flip sign between runs, two libraries disagree about v₁, and downstream sign-sensitive uses (interpreting loadings, comparing directions) silently break. The fix is a documented, deterministic rule — here "largest |component| positive, ties → first component positive" — applied identically to every singular vector, which is what the module does (and tests).',
  },
  {
    id: 'svd-fail-degenerate-rank',
    title: 'Degenerate Rank: σ₂ = 0 and the Second Singular Vector Is Undefined',
    scenario: 'collinearity',
    params: {
      n: 40, corr: 0.7, rotDeg: 30, noise: 0.15, seed: 42, rank: 2,
      points: '[[1,2],[3,4],[5,6],[7,8]]',
    },
    narration:
      'The points lie exactly on a line: the data matrix has rank 1, so σ₁ = 6.3246 and σ₂ = 0. The second left singular vector u₂ = Xv₂/σ₂ is a 0/0 — NOT determined by the data. The module handles it honestly: it completes u₂ deterministically (Gram–Schmidt against u₁), flags rankDeficient = 1, and saturates the singular ratio at 1e9 so every metric stays finite. The run completes, and the rank-1 reconstruction is already exact (measured error 7.0e-16) — the second dimension carries ZERO information.',
    whyItBreaks:
      'When σ_k = 0 the direction u_k is undefined: infinitely many unit vectors are orthogonal to the range, and X = UΣVᵀ holds for any of them (they contribute nothing, since σ_k = 0). A naive implementation divides by zero and produces NaN. The failure is not the mathematics — it is the honest handling: the module documents the completion convention instead of crashing, and every metric stays finite in the sandbox. The takeaway: rank-deficiency does not stop the SVD (it is exactly the case the pseudoinverse exists for), but it makes the extra singular vectors meaningless — the data simply has no second direction.',
  },
];