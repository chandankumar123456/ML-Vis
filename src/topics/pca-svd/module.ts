// src/topics/pca-svd/module.ts
// Task 17 (Wave 4): pca-svd — the EXACT 2×2 SVD of the centered correlated
// Gaussian design matrix, factorized entirely by closed forms (no iterative
// solver), with the mandated factorization build-up sweep + rank sweep and the
// Eckart-Young low-rank story.
//
// Design decisions (deviations from the plan are documented in the report):
//  - Data conventions mirror the sibling pca topic EXACTLY (same params n /
//    corr / rotDeg / noise / seed, same mulberry32 stream, same Box–Muller
//    draws, same DATA_OFFSET): the same params produce byte-identical point
//    clouds in both topics, so the "SVD vs PCA" comparison is apples-to-apples.
//    The factorized object is the CENTERED design matrix X (n×2, rows = points,
//    columns = features) — the matrix whose covariance is (1/n)XᵀX.
//  - EXACT 2×2 SVD via closed forms (all exported for hand-verified tests):
//      1. Gram matrix G = XᵀX (2×2, symmetric PSD);
//      2. eigen-decomposition of G via the rotation-angle closed form
//         (λ₁,₂ = (a+c ± √((a−c)²+4b²))/2, θ = ½·atan2(2b, a−c),
//         v₁ = (cosθ, sinθ), v₂ = (−sinθ, cosθ)) — V = [v₁ v₂], σ_k = √λ_k;
//      3. U column-wise: u_k = X·v_k / σ_k, renormalized. Because the v's are
//         orthonormal, u₁·u₂ = 0 holds to machine precision BY CONSTRUCTION.
//  - Deterministic sign convention (documented; the plan's "largest |component|
//    positive" suggestion): each right singular vector v_k is oriented so its
//    largest-|component| entry is positive; exact ties are broken by making the
//    FIRST component positive. U inherits the sign via u_k = Xv_k/σ_k (flipping
//    v_k flips u_k — the pairing that preserves X = UΣVᵀ). rawV (pre-convention)
//    is exposed for the sign-flip failure demo + tests.
//  - Rank-deficient singular values handled honestly (plan: "handled honestly,
//    documented; telemetry or warning"): when σ₂ ≤ 1e-9·σ₁ the second left
//    singular vector is NOT determined by the data (u₂ = Xv₂/σ₂ would be 0/0).
//    We complete it deterministically (Gram–Schmidt of the first standard-basis
//    vector against u₁), emit rankDeficient: 1 and narrate the convention; the
//    run completes because the reconstruction is still EXACT (σ₂ = 0).
//    Zero-variance data (all points identical → G = 0) THROWS from getSweep →
//    honest telemetry failure (the pca precedent). The singularRatio metric is
//    SATURATED at 1e9 when σ₂ ≈ 0 so the sandbox never sees a non-finite value.
//  - Step model (mandated "sweep + final exact snapshot", the pca/lda
//    precedent): 4 FACTORIZATION BUILD-UP snapshots — X/XᵀX → V, λ → Σ =
//    diag(σ₁,σ₂) (relation XᵀX = VΣ²Vᵀ shown side-by-side) → U = XV/Σ with the
//    full X = UΣVᵀ assembled — followed by a RANK SWEEP k = 1..params.rank (one
//    reconstruction snapshot per k; the LAST snapshot is the slider rank
//    exactly, the ridge λ-sweep convention). Rank snapshots carry the ECONOMY
//    factors U_k (n×k), Σ_k (k×k), V_kᵀ (k×d) with the exact dimensions.
// - lossMetricKey = 'reconstructionError' (lower-better; documented in the
  //    layer title, the lda 'jFisher' precedent): the Eckart-Young mean squared
  //    error per sample, (1/n)·‖X−X̂_k‖_F² = σ_{k+1}²/n, measured FROM the actual
  //    reconstructed matrix. The metric is emitted on EVERY snapshot so the
  //    loss curve renders a continuous series (the LossCurve draws NaN gaps as
  //    empty): the four build-up snapshots carry the rank-0 baseline
  //    ‖X‖²_F/n = (λ₁+λ₂)/n (keeping NO singular vectors costs the full data
  //    energy), and the rank snapshots override it with the actual rank-k
  //    error. The curve therefore reads as a flat plateau during build-up, a
  //    drop to σ₂²/n at rank 1, and EXACTLY 0 at full rank (Eckart-Young).
//  - "Singular value bars" + reconstruction residuals via registry-only views:
//    the eigenviewer renders the σ²-explained fraction bars from the point
//    cloud and its reconstruct mode draws the per-point error lines from the
//    projection commands (no new registry components — the pca precedent).
//  - Unsupervised: no labels, no classifier registration. The `points` override
//    (JSON '[[x,y],…]') is test-only and is what the failure demos use for
//    hand-crafted datasets (zero-variance, rank-deficient).
import type { TopicModule, Params, SimState, VisualCommand, ParamValue, MathStep } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { svdTestCases } from './testCases';
import { svdFormulas } from './formulas';
import { svdDerivations } from './derivations';
import { svdMistakes } from './mistakes';
import { svdQuestions } from './questions';
import { svdComparisons } from './comparisons';
import { svdFailureDemos } from './failures';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Build-up phase: X/XᵀX → V, λ → Σ → U (full factorization). Then the rank sweep.
export const BUILDUP_STEPS = 4;
// The rank slider domain (d = 2 columns ⇒ k ∈ {1, 2}).
export const MAX_RANK = 2;
// σ₂ ≤ RANK_TOL·σ₁ ⇒ the second singular value is numerically zero (rank 1).
export const RANK_TOL = 1e-9;
// singularRatio saturation for rank-deficient data (keeps every metric finite).
export const RATIO_CAP = 1e9;

// Fixed NONZERO data mean (pca's convention — the centered/uncentered contrast
// is a real, measurable mistake in both topics; the SVD target is the centered
// matrix, so the offset cancels in X_c but dominates an UNcentered X).
export const DATA_OFFSET: [number, number] = [2.5, 1.5];

// Visual-semantic colors (data cloud + singular-vector axes).
export const POINT_COLOR = '#2563eb';    // single unlabeled cloud (unsupervised)
export const AXIS_COLOR = '#64748b';     // build-up candidate axis (v₁)
export const V1_COLOR = '#dc2626';       // first right singular vector (PC1 line)
export const V2_COLOR = '#16a34a';       // second right singular vector (PC2 line)

// ---------------------------------------------------------------------------
// Deterministic PRNG + data synthesis (byte-identical to the pca topic)
// ---------------------------------------------------------------------------

export interface SvdPoint { x: number; y: number; }

/** Mulberry32 — deterministic PRNG (matches every Wave-1/2/3/4 topic). */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller standard normal (with the spare-draw trick). */
function makeNormal(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0;
    let v = 0;
    do { u = rng(); } while (u <= 1e-12);
    v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

/**
 * Deterministic data synthesis — a single correlated Gaussian cloud:
 *   xᵢ ~ N(DATA_OFFSET, Σ),  Σ = R(φ)·[[1,ρ],[ρ,1]]·R(φ)ᵀ + σ²I,
 * drawn exactly via the Cholesky factor L = R(φ)·chol([[1,ρ],[ρ,1]]) with
 * σ·N(0,1) added per axis. IDENTICAL draw order and arithmetic to the pca
 * topic's generateData — same params ⇒ same points in both topics.
 * Test-only override: `points` (JSON '[[x,y],…]') → a hand-crafted dataset.
 */
export function generateData(p: Params): SvdPoint[] {
  if (typeof p.points === 'string') {
    const rows: [number, number][] = JSON.parse(p.points);
    return rows.map(([x, y]) => ({ x, y }));
  }
  const n = (p.n as number) ?? 40;
  const corr = (p.corr as number) ?? 0.7;
  const rotDeg = (p.rotDeg as number) ?? 30;
  const noise = (p.noise as number) ?? 0.15;
  const seed = (p.seed as number) ?? 42;
  const rng = mulberry32(seed);
  const normal = makeNormal(rng);
  const phi = (rotDeg * Math.PI) / 180;
  const cos = Math.cos(phi), sin = Math.sin(phi);
  const s = Math.sqrt(Math.max(0, 1 - corr * corr));
  // L = R(φ)·chol([[1,ρ],[ρ,1]]) = [[cos−ρ·sin, −s·sin],[sin+ρ·cos, s·cos]]
  const l11 = cos - corr * sin, l12 = -s * sin;
  const l21 = sin + corr * cos, l22 = s * cos;
  const pts: SvdPoint[] = [];
  for (let i = 0; i < n; i++) {
    const g1 = normal(), g2 = normal();
    const x = l11 * g1 + l12 * g2 + (noise > 0 ? noise * normal() : 0);
    const y = l21 * g1 + l22 * g2 + (noise > 0 ? noise * normal() : 0);
    pts.push({ x: x + DATA_OFFSET[0], y: y + DATA_OFFSET[1] });
  }
  return pts;
}

// ---------------------------------------------------------------------------
// SVD math core (all mutation-free, exported for hand-verified tests)
// ---------------------------------------------------------------------------

/** Arithmetic mean of the points; [0, 0] for an empty list. */
export function meanOf(points: SvdPoint[]): [number, number] {
  const n = Math.max(points.length, 1);
  let sx = 0, sy = 0;
  for (const d of points) { sx += d.x; sy += d.y; }
  return [sx / n, sy / n];
}

/** The CENTERED design matrix X (n×2): rows = points, columns = features. */
export function centeredDesignMatrix(points: SvdPoint[], mu: [number, number]): number[][] {
  return points.map((d) => [d.x - mu[0], d.y - mu[1]]);
}

/** Gram matrix XᵀX (d×d symmetric PSD) of an n×d design matrix. */
export function gramMatrix(X: number[][]): number[][] {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const G: number[][] = Array.from({ length: d }, () => Array(d).fill(0));
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < d; a++) {
      for (let b = 0; b < d; b++) G[a][b] += X[i][a] * X[i][b];
    }
  }
  return G;
}

/** Matrix–vector product X·v (n-vector). */
export function matVec(X: number[][], v: number[]): number[] {
  return X.map((row) => row[0] * v[0] + row[1] * v[1]);
}

/** Euclidean normalization; throws on a degenerate (zero) vector. */
function unit(v: number[]): number[] {
  let s = 0;
  for (const x of v) s += x * x;
  const norm = Math.sqrt(s);
  if (norm < 1e-15) throw new Error('pca-svd: degenerate vector in unit() — internal invariant violated');
  return v.map((x) => x / norm);
}

export interface Eigen2x2Symmetric {
  lambda1: number; lambda2: number;        // sorted descending
  rawV1: [number, number]; rawV2: [number, number]; // pre-sign-fix eigenvectors
  angleDeg: number;                        // raw v₁ line angle ∈ [0, 180)
}

/**
 * EXACT 2×2 symmetric eigen-decomposition via the rotation-angle closed form:
 *   λ₁,₂ = (a+c ± √((a−c)² + 4b²))/2,   θ = ½·atan2(2b, a−c),
 *   v₁ = (cosθ, sinθ),  v₂ = (−sinθ, cosθ) (exact 90° rotation — orthogonal by
 *   construction). Throws an honest error when the matrix is the zero matrix
 *   (λ₁ + λ₂ < 1e-12) — the zero-variance telemetry failure path.
 */
export function eigen2x2Symmetric(M: number[][]): Eigen2x2Symmetric {
  const a = M[0][0], b = M[0][1], c = M[1][1];
  // Explicit rejection of non-finite entries: NaN/Infinity would otherwise flow
  // silently through the closed form (NaN < 1e-12 is false, so the zero-matrix
  // guard below would NOT fire) and produce a garbage "factorization" instead
  // of an honest failure. SVD requires a complete, finite design matrix.
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) {
    throw new Error(
      'pca-svd: non-finite Gram matrix XᵀX (missing or corrupt data) — the SVD requires a complete, finite design matrix',
    );
  }
  const t = a + c;                                        // trace
  const delta = Math.max(0, (a - c) * (a - c) + 4 * b * b);
  const root = Math.sqrt(delta);
  const lambda1 = (t + root) / 2;
  const lambda2 = (t - root) / 2;
  if (lambda1 + lambda2 < 1e-12) {
    throw new Error(
      'pca-svd: zero-variance data — the Gram matrix XᵀX is the zero matrix (all points identical); ' +
      'the SVD needs some spread to find singular directions',
    );
  }
  let angle = 0.5 * Math.atan2(2 * b, a - c);            // major-axis angle
  if (angle < 0) angle += Math.PI;                       // line angle in [0, π)
  return {
    lambda1, lambda2,
    rawV1: [Math.cos(angle), Math.sin(angle)],
    rawV2: [-Math.sin(angle), Math.cos(angle)],
    angleDeg: (angle * 180) / Math.PI,
  };
}

/**
 * Deterministic sign convention (documented in the module header): orient v so
 * the component with the largest |value| is positive; exact ties are broken by
 * making the FIRST component positive. Unique for every unit vector except the
 * measure-zero tie case, which the tie-break also resolves deterministically.
 */
export function signFix(v: [number, number]): [number, number] {
  if (Math.abs(v[0]) >= Math.abs(v[1])) {
    return v[0] < 0 ? [-v[0], -v[1]] : [v[0], v[1]];
  }
  return v[1] < 0 ? [-v[0], -v[1]] : [v[0], v[1]];
}

/**
 * Deterministic null-space completion for a rank-deficient U column (σ_k ≈ 0):
 * the k-th left singular vector is NOT determined by the data (u_k = Xv_k/σ_k
 * is 0/0), so we pick the canonical unit vector orthogonal to the range:
 * Gram–Schmidt of the FIRST standard-basis vector not parallel to u₁. Any unit
 * vector orthogonal to the range is a valid left singular vector for σ = 0;
 * this rule is deterministic and documented (rankDeficient is flagged).
 */
export function nullSpaceCompletion(u1: number[]): number[] {
  const n = u1.length;
  for (let j = 0; j < n; j++) {
    if (Math.abs(u1[j]) < 1 - 1e-12) {
      const e: number[] = new Array(n).fill(0);
      e[j] = 1;
      const w = e.map((v, i) => v - u1[j] * u1[i]);
      return unit(w);
    }
  }
  throw new Error('pca-svd: null-space completion failed — u₁ spans every standard basis vector');
}

export interface SvdCore {
  X: number[][];                  // n×2 centered design matrix
  Gram: number[][];               // 2×2 XᵀX
  gramLambda1: number; gramLambda2: number; // eigenvalues of XᵀX (= σ²)
  covLambda1: number; covLambda2: number;   // eigenvalues of (1/n)XᵀX (= σ²/n)
  sigma1: number; sigma2: number;           // σ₁ ≥ σ₂ ≥ 0
  V1: [number, number]; V2: [number, number];  // right singular vectors (sign-fixed)
  Vt: number[][];                 // 2×2: rows [V1ᵀ; V2ᵀ]
  U1: number[]; U2: number[];     // left singular vectors (n-vectors, columns of U)
  Sigma: number[][];              // 2×2 diag(σ₁, σ₂)
  angleDeg: number;               // sign-fixed v₁ line angle ∈ [0, 180)
  rawV1: [number, number]; rawV2: [number, number]; // pre-convention (sign demo)
  rawAngleDeg: number;            // pre-convention v₁ angle
  rankDeficient: boolean;
}

/**
 * The EXACT 2×2 SVD of X = UΣVᵀ built from closed forms (module header):
 * Gram eigen-decomposition → σ = √λ → U = XV/Σ column-wise, with the
 * deterministic sign convention and the honest rank-deficient completion.
 * Throws on zero-variance data (telemetry failure path).
 */
export function computeSvd(X: number[][]): SvdCore {
  const n = X.length;
  const Gram = gramMatrix(X);
  const eig = eigen2x2Symmetric(Gram);
  const sigma1 = Math.sqrt(Math.max(0, eig.lambda1));
  const sigma2 = Math.sqrt(Math.max(0, eig.lambda2));
  const V1 = signFix(eig.rawV1);
  const V2 = signFix(eig.rawV2);
  const rankDeficient = sigma2 <= RANK_TOL * sigma1;
  const Xv1 = matVec(X, V1);
  const Xv2 = matVec(X, V2);
  // U column-wise: u_k = X·v_k / σ_k (exactly unit: ‖Xv_k‖ = √λ_k = σ_k).
  // For σ_k ≈ 0 the quotient is undefined → deterministic completion (above).
  const U1 = unit(Xv1.map((v) => v / sigma1));
  const U2 = sigma2 > RANK_TOL * sigma1
    ? unit(Xv2.map((v) => v / sigma2))
    : nullSpaceCompletion(U1);
  const covLambda1 = eig.lambda1 / n;
  const covLambda2 = eig.lambda2 / n;
  return {
    X,
    Gram,
    gramLambda1: eig.lambda1, gramLambda2: eig.lambda2,
    covLambda1, covLambda2,
    sigma1, sigma2,
    V1, V2,
    Vt: [[V1[0], V1[1]], [V2[0], V2[1]]],
    U1, U2,
    Sigma: [[sigma1, 0], [0, sigma2]],
    angleDeg: (Math.atan2(V1[1], V1[0]) * 180) / Math.PI,
    rawV1: eig.rawV1, rawV2: eig.rawV2,
    rawAngleDeg: eig.angleDeg,
    rankDeficient,
  };
}

/** Spectral norm (largest singular value) of an n×2 matrix via its Gram matrix. */
export function spectralNorm2x2(R: number[][]): number {
  const G = gramMatrix(R);
  const t = G[0][0] + G[1][1];
  if (t < 1e-24) return 0;
  const delta = Math.max(0, (G[0][0] - G[1][1]) ** 2 + 4 * G[0][1] * G[0][1]);
  return Math.sqrt((t + Math.sqrt(delta)) / 2);
}

export interface RankKRecon {
  k: number;
  Xhat: number[][];      // n×2 reconstruction X̂_k = Σ_{j≤k} σ_j u_j v_jᵀ
  errFro: number;        // ‖X − X̂_k‖_F = σ_{k+1} (Eckart-Young, Frobenius)
  errSpectral: number;   // largest singular value of X − X̂_k = σ_{k+1}
  errMse: number;        // (1/n)·‖X − X̂_k‖_F² = σ_{k+1}²/n (loss-curve metric)
}

/** Reconstruct X̂_k = Σ_{j≤k} σ_j u_j v_jᵀ and MEASURE the residual errors. */
export function rankKRecon(core: SvdCore, k: number): RankKRecon {
  const n = core.X.length;
  const Xhat: number[][] = Array.from({ length: n }, () => [0, 0]);
  const pairs: [number, number[], [number, number]][] = [];
  if (k >= 1) pairs.push([core.sigma1, core.U1, core.V1]);
  if (k >= 2) pairs.push([core.sigma2, core.U2, core.V2]);
  for (const [s, u, v] of pairs) {
    for (let i = 0; i < n; i++) {
      Xhat[i][0] += s * u[i] * v[0];
      Xhat[i][1] += s * u[i] * v[1];
    }
  }
  const R = core.X.map((row, i) => [row[0] - Xhat[i][0], row[1] - Xhat[i][1]]);
  let fro2 = 0;
  for (const row of R) fro2 += row[0] * row[0] + row[1] * row[1];
  const fro = Math.sqrt(fro2);
  return { k, Xhat, errFro: fro, errSpectral: spectralNorm2x2(R), errMse: fro2 / n };
}

// ---------------------------------------------------------------------------
// The sweep (build-up + rank) — the run's single source
// ---------------------------------------------------------------------------

export interface SvdSweep {
  data: SvdPoint[];
  dataSeed: number;
  mu: [number, number];
  core: SvdCore;
  ranks: number[];        // [1, .., min(params.rank, MAX_RANK)]
  recon: RankKRecon[];    // recon[r-1] = reconstruction at rank r
}

export function getSweep(p: Params): SvdSweep {
  const data = generateData(p);
  const dataSeed = (p.seed as number) ?? 42;
  const mu = meanOf(data);
  const X = centeredDesignMatrix(data, mu);
  const core = computeSvd(X);
  const maxRank = Math.min((p.rank as number) ?? 2, MAX_RANK);
  const ranks = Array.from({ length: maxRank }, (_, i) => i + 1);
  const recon = ranks.map((k) => rankKRecon(core, k));
  return { data, dataSeed, mu, core, ranks, recon };
}

// Bounded memoization (the svm/pca precedent): initialState/step stay O(1)
// after the first evaluation of a params key.
const SWEEP_CACHE = new Map<string, SvdSweep>();
const SWEEP_CACHE_MAX = 16;

function sweepKey(p: Params): string {
  return JSON.stringify([p.n, p.corr, p.rotDeg, p.noise, p.seed, p.rank, p.points ?? null]);
}

export function cachedSweep(p: Params): SvdSweep {
  const key = sweepKey(p);
  let sw = SWEEP_CACHE.get(key);
  if (!sw) {
    sw = getSweep(p);
    if (SWEEP_CACHE.size >= SWEEP_CACHE_MAX) SWEEP_CACHE.clear();
    SWEEP_CACHE.set(key, sw);
  }
  return sw;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

function baseMetrics(sweep: SvdSweep, step: number, rank: number, isFinal: boolean): Record<string, number> {
  const c = sweep.core;
  // σ₁/σ₂ saturated at RATIO_CAP when σ₂ ≈ 0 — keeps every metric finite in
  // the sandbox (the rankDeficient flag carries the honest meaning).
  const ratio = c.sigma2 > RANK_TOL * c.sigma1 ? c.sigma1 / c.sigma2 : RATIO_CAP;
  return {
    step,
    rank,                                     // displayed rank: 0 = build-up, k otherwise
    sigma1: c.sigma1,
    sigma2: c.sigma2,
    singularRatio: ratio,
    lambda1: c.gramLambda1,                   // eigenvalues of XᵀX (= σ²)
    lambda2: c.gramLambda2,
    covLambda1: c.covLambda1,                 // eigenvalues of (1/n)XᵀX (= σ²/n)
    covLambda2: c.covLambda2,
    n: sweep.data.length,
    dataSeed: sweep.dataSeed,
    rankDeficient: c.rankDeficient ? 1 : 0,
    isOptimal: isFinal ? 1 : 0,
    // rank-0 baseline (kept on every snapshot; rank snapshots override it with
    // the actual rank-k error): the error of reconstructing with NOTHING —
    // ‖X‖²_F/n = trace(XᵀX)/n = (λ₁ + λ₂)/n, exactly the full data energy.
    reconstructionError: (c.gramLambda1 + c.gramLambda2) / sweep.data.length,
  };
}

// ---------------------------------------------------------------------------
// Visuals
// ---------------------------------------------------------------------------

/** Clip the ray center + t·dir (unit dir) to the data bbox (Liang–Barsky). */
function clipRay(center: [number, number], dir: [number, number],
  x0: number, x1: number, y0: number, y1: number): [number, number][] | null {
  let tMin = -Infinity, tMax = Infinity;
  for (const [lo, hi, pc, dc] of [[x0, x1, center[0], dir[0]], [y0, y1, center[1], dir[1]]] as const) {
    if (Math.abs(dc) < 1e-12) continue;
    const t1 = (lo - pc) / dc, t2 = (hi - pc) / dc;
    tMin = Math.max(tMin, Math.min(t1, t2));
    tMax = Math.min(tMax, Math.max(t1, t2));
  }
  if (tMin > tMax || !Number.isFinite(tMin) || !Number.isFinite(tMax)) return null;
  return [[center[0] + tMin * dir[0], center[1] + tMin * dir[1]],
    [center[0] + tMax * dir[0], center[1] + tMax * dir[1]]] as [number, number][];
}

function dataBounds(points: SvdPoint[]): { x0: number; x1: number; y0: number; y1: number } {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const d of points) {
    x0 = Math.min(x0, d.x); x1 = Math.max(x1, d.x);
    y0 = Math.min(y0, d.y); y1 = Math.max(y1, d.y);
  }
  return { x0, x1, y0, y1 };
}

/**
 * Scatter-plot commands: the data cloud, the current v₁ axis through the data
 * mean, a direction arrow; the final snapshot adds both singular-vector axes.
 */
function buildScatter(sweep: SvdSweep, isFinal: boolean): VisualCommand[] {
  const { data, mu, core } = sweep;
  const cmd: VisualCommand[] = data.map((d, i) => ({
    type: 'point', id: `d${i}`, x: d.x, y: d.y, color: POINT_COLOR,
  }));
  const { x0, x1, y0, y1 } = dataBounds(data);
  const line = clipRay(mu, core.V1, x0, x1, y0, y1);
  if (line) cmd.push({ type: 'line', id: 'axis', points: line, color: AXIS_COLOR });
  cmd.push({
    type: 'arrow', id: 'dir', color: AXIS_COLOR,
    x1: mu[0], y1: mu[1], x2: mu[0] + core.V1[0] * 1.1, y2: mu[1] + core.V1[1] * 1.1,
  });
  if (isFinal) {
    const p1 = clipRay(mu, core.V1, x0, x1, y0, y1);
    const p2 = clipRay(mu, core.V2, x0, x1, y0, y1);
    if (p1) cmd.push({ type: 'line', id: 'v1', points: p1, color: V1_COLOR });
    if (p2) cmd.push({ type: 'line', id: 'v2', points: p2, color: V2_COLOR });
    cmd.push({ type: 'arrow', id: 'v1-arrow', color: V1_COLOR, x1: mu[0], y1: mu[1], x2: mu[0] + core.V1[0] * 1.1, y2: mu[1] + core.V1[1] * 1.1 });
    cmd.push({ type: 'arrow', id: 'v2-arrow', color: V2_COLOR, x1: mu[0], y1: mu[1], x2: mu[0] + core.V2[0] * 1.1, y2: mu[1] + core.V2[1] * 1.1 });
  }
  return cmd;
}

/**
 * Eigenviewer commands: the data cloud, the candidate axis along v₁ (angle in
 * radians through the centroid) and per-point orthogonal projections with
 * residuals. The view renders the σ²-explained bars itself and its reconstruct
 * mode draws the error lines — the low-rank reconstruction visual.
 */
function buildEigenviewer(sweep: SvdSweep): VisualCommand[] {
  const { data, mu, core } = sweep;
  const cmd: VisualCommand[] = data.map((d, i) => ({
    type: 'point', id: `d${i}`, x: d.x, y: d.y, color: POINT_COLOR,
  }));
  const angle = Math.atan2(core.V1[1], core.V1[0]);
  cmd.push({ type: 'axis', id: 'axis', angle, color: AXIS_COLOR });
  data.forEach((d, i) => {
    const t = core.V1[0] * (d.x - mu[0]) + core.V1[1] * (d.y - mu[1]);
    const px = mu[0] + core.V1[0] * t;
    const py = mu[1] + core.V1[1] * t;
    cmd.push({
      type: 'projection', id: `proj${i}`,
      point: [d.x, d.y], onto: [px, py],
      residual: Math.hypot(d.x - px, d.y - py),
    });
  });
  return cmd;
}

/** k×k diagonal matrix from the first k singular values. */
export function diagMatrix(vals: number[], k: number): number[][] {
  return Array.from({ length: k }, (_, r) => Array.from({ length: k }, (_, c) => (r === c ? vals[r] : 0)));
}

/** Economy U_k: n×k matrix whose columns are the first k left singular vectors. */
export function economyU(core: SvdCore, k: number): number[][] {
  const cols = [core.U1];
  if (k >= 2) cols.push(core.U2);
  const n = core.U1.length;
  return Array.from({ length: n }, (_, i) => cols.slice(0, k).map((c) => c[i]));
}

/** Economy V_kᵀ: k×d matrix whose rows are the first k right singular vectors. */
export function economyVt(core: SvdCore, k: number): number[][] {
  const rows = [[core.V1[0], core.V1[1]]];
  if (k >= 2) rows.push([core.V2[0], core.V2[1]]);
  return rows.slice(0, k);
}

/** Matrix story per snapshot phase (ids carry the exact dimensions). */
function buildMatrices(sweep: SvdSweep, phase: number): VisualCommand[] {
  const c = sweep.core;
  const cmds: VisualCommand[] = [];
  if (phase === 1) {
    cmds.push(
      { type: 'matrix', id: 'X (n×2)', rows: c.X.length, cols: 2, cells: c.X },
      { type: 'matrix', id: 'XᵀX (2×2)', rows: 2, cols: 2, cells: c.Gram },
    );
  }
  if (phase === 2) {
    cmds.push(
      { type: 'matrix', id: 'V (2×2) = [v₁ v₂]', rows: 2, cols: 2, cells: c.Vt },
      { type: 'matrix', id: 'λ (2×1) = eig(XᵀX)', rows: 2, cols: 1, cells: [[c.gramLambda1], [c.gramLambda2]] },
    );
  }
  if (phase === 3) {
    cmds.push(
      { type: 'matrix', id: 'V (2×2)', rows: 2, cols: 2, cells: c.Vt },
      { type: 'matrix', id: 'Σ² (2×2) = diag(λ)', rows: 2, cols: 2, cells: [[c.gramLambda1, 0], [0, c.gramLambda2]] },
      { type: 'matrix', id: 'Σ (2×2) = diag(σ₁,σ₂)', rows: 2, cols: 2, cells: c.Sigma },
    );
  }
  if (phase === 4) {
    cmds.push(
      { type: 'matrix', id: 'X (n×2) = UΣVᵀ', rows: c.X.length, cols: 2, cells: c.X },
      { type: 'matrix', id: 'U (n×2)', rows: c.U1.length, cols: 2, cells: economyU(c, 2) },
      { type: 'matrix', id: 'Σ (2×2)', rows: 2, cols: 2, cells: c.Sigma },
      { type: 'matrix', id: 'Vᵀ (2×2)', rows: 2, cols: 2, cells: c.Vt },
    );
  }
  return cmds;
}

/** Economy-factor + error matrices for a rank-k reconstruction snapshot. */
function buildRankMatrices(sweep: SvdSweep, k: number): VisualCommand[] {
  const c = sweep.core;
  const r = sweep.recon[k - 1];
  const cmds: VisualCommand[] = [
    { type: 'matrix', id: `U_${k} (n×${k})`, rows: c.U1.length, cols: k, cells: economyU(c, k) },
    { type: 'matrix', id: `Σ_${k} (${k}×${k})`, rows: k, cols: k, cells: diagMatrix([c.sigma1, c.sigma2], k) },
    { type: 'matrix', id: `V_${k}ᵀ (${k}×2)`, rows: k, cols: 2, cells: economyVt(c, k) },
    { type: 'matrix', id: `X̂_${k} (n×2)`, rows: r.Xhat.length, cols: 2, cells: r.Xhat },
    { type: 'matrix', id: `error (σ_${k + 1} = ${c.sigma1 ? (k === 1 ? c.sigma2 : 0) : 0})`, rows: 1, cols: 1, cells: [[r.errFro]] },
  ];
  return cmds;
}

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

function fmtMat(M: number[][]): string {
  return `[[${M[0][0].toFixed(3)}, ${M[0][1].toFixed(3)}],[${M[1][0].toFixed(3)}, ${M[1][1].toFixed(3)}]]`;
}

/** Build-up snapshot (phase 1..4): X/XᵀX → V, λ → Σ → U, full factorization. */
function buildupSnapshot(sweep: SvdSweep, phase: number, first: boolean): SimState {
  const c = sweep.core;
  const m = baseMetrics(sweep, phase, 0, false);
  const events: SimState['events'] = first
    ? [
        { type: 'init', label: 'svd-seeded-correlated-gaussian', step: phase },
        { type: 'compute', label: 'gram-matrix-XᵀX', step: phase },
      ]
    : phase === 2
      ? [{ type: 'compute', label: 'XᵀX-eigendecomposition', step: phase }]
      : phase === 3
        ? [{ type: 'compute', label: 'singular-values-extracted', step: phase }]
        : [{ type: 'compute', label: 'factorization-complete', step: phase }];

  const math: MathStep[] = [
    { latex: 'X = U \\Sigma V^T, \\quad X \\in \\mathbb{R}^{n \\times 2}', id: 'svd-factorization' },
    { latex: 'X^T X = V \\Sigma^2 V^T \\;\\Rightarrow\\; \\sigma_k = \\sqrt{\\lambda_k(X^T X)}', id: 'svd-gram-eigen' },
    { latex: 'U_k = \\frac{X v_k}{\\sigma_k}', id: 'svd-u-construction' },
  ];

  const narrations: Record<number, string> = {
    1: `Build-up 1/4: the cloud is centered at μ = (${sweep.mu[0].toFixed(2)}, ${sweep.mu[1].toFixed(2)}) and its centered design matrix ` +
       `X (n×2) is formed. The SVD factorizes X = UΣVᵀ; because XᵀX is a symmetric 2×2 matrix, its eigen-decomposition gives us the right ` +
       `singular vectors and the SQUARED singular values directly: XᵀX = VΣ²Vᵀ.`,
    2: `Build-up 2/4: the exact 2×2 eigen closed form solves XᵀX = VΣ²Vᵀ — λ₁ = ${c.gramLambda1.toFixed(4)}, λ₂ = ${c.gramLambda2.toFixed(4)}, ` +
       `v₁ = (${c.rawV1[0].toFixed(4)}, ${c.rawV1[1].toFixed(4)}), v₂ ⊥ v₁ (exact 90° rotation). The eigenvalues of XᵀX are the SQUARED singular values: σ_k = √λ_k.`,
    3: `Build-up 3/4: take square roots — σ₁ = ${c.sigma1.toFixed(4)}, σ₂ = ${c.sigma2.toFixed(4)} — and pack them into the diagonal matrix ` +
       `Σ = diag(σ₁, σ₂). The relation XᵀX = VΣ²Vᵀ now shows the eigendecomposition (V, λ) side-by-side with the SVD's own factors (V, Σ², Σ): same V, λ = σ².`,
    4: `Build-up 4/4: left singular vectors column-wise — u_k = X·v_k / σ_k (each u_k is exactly unit because ‖X·v_k‖ = √λ_k = σ_k) — and the full ` +
       `factorization X = UΣVᵀ is assembled. UᵀU = I and VᵀV = I hold to machine precision by construction.`,
  };

  const whys: Record<number, string> = {
    1: `The SVD factorizes the data matrix itself (X = UΣVᵀ) rather than a covariance. The build-up starts where the math does: form XᵀX, the 2×2 Gram matrix, ` +
       `whose eigen-decomposition provides the entire SVD (V and Σ²) at closed-form cost.`,
    2: `The eigen closed form (characteristic polynomial) is exact and deterministic — no iteration, no approximate solver. λ_k(XᵀX) = σ_k² because ` +
       `XᵀX = VΣ²Vᵀ with V orthonormal: left-multiplying by Vᵀ and right-multiplying by V diagonalizes XᵀX into Σ².`,
    3: `Singular values are non-negative square roots of the Gram eigenvalues (a Gram matrix is positive semidefinite, so λ ≥ 0). Sorting is inherited: ` +
       `λ₁ ≥ λ₂ forces σ₁ ≥ σ₂, and the V columns are ordered to match.`,
    4: `u_k = Xv_k/σ_k is the exact left singular vector: ‖Xv_k‖ = √(v_kᵀXᵀXv_k) = √λ_k = σ_k, so u_k is unit, and orthogonality of the v's makes the u's ` +
       `orthogonal. The factorization X = UΣVᵀ reconstructs the data EXACTLY at full rank.`,
  };

  return {
    algorithm: {
      mode: 'svd-build', step: phase, phase, rank: 0,
      v1x: c.V1[0], v1y: c.V1[1], v2x: c.V2[0], v2y: c.V2[1],
      sigma1: c.sigma1, sigma2: c.sigma2, dataSeed: sweep.dataSeed,
    } as Record<string, ParamValue>,
    visuals: [
      ...buildEigenviewer(sweep),
      ...buildScatter(sweep, false),
      ...buildMatrices(sweep, phase),
    ],
    math: phase === 1 ? [math[0], math[1]] : phase === 2 ? [math[1]] : phase === 3 ? [math[1]] : [math[0], math[2]],
    narration: narrations[phase],
    explanation: {
      changed: first
        ? ['data centered', `XᵀX = ${fmtMat(c.Gram)}`, `σ₁ = ${c.sigma1.toFixed(4)}, σ₂ = ${c.sigma2.toFixed(4)}`]
        : phase === 2
          ? [`V = ${fmtMat(c.Vt)}`, `λ = (${c.gramLambda1.toFixed(4)}, ${c.gramLambda2.toFixed(4)})`]
          : phase === 3
            ? [`Σ = diag(${c.sigma1.toFixed(4)}, ${c.sigma2.toFixed(4)})`, `σ² = λ (side-by-side)`]
            : [`U (n×2) built`, `X = UΣVᵀ assembled`],
      why: whys[phase],
      formulaRef: phase === 2 ? 'svd-gram-eigen' : phase === 3 ? 'svd-gram-eigen' : phase === 4 ? 'svd-u-construction' : 'svd-factorization',
      dependsOn: ['linear-algebra', 'probability', 'statistics'],
      gateConcepts: ['SVD', 'singular value', 'singular vector', 'Gram matrix', 'eigen-decomposition'],
    },
    highlights: first ? [] : phase === 3
      ? [{ panel: 'matrix', id: 'Σ² (2×2) = diag(λ):0,0', intensity: 1 }, { panel: 'matrix', id: 'Σ (2×2) = diag(σ₁,σ₂):0,0', intensity: 1 }]
      : phase === 4
        ? [{ panel: 'matrix', id: 'Σ (2×2):0,0', intensity: 1 }]
        : [],
    metrics: m,
    events,
    timeline: phase === 1
      ? ['Data', 'Center', 'Gram']
      : phase === 2
        ? ['Gram', 'Eigen', 'Singular Values']
        : phase === 3
          ? ['Singular Values', 'Sigma']
          : ['Sigma', 'U', 'SVD Complete'],
  };
}

/** Rank-k reconstruction snapshot (the rank sweep). */
function rankSnapshot(sweep: SvdSweep, k: number, isFinal: boolean): SimState {
  const c = sweep.core;
  const r = sweep.recon[k - 1];
  const step = BUILDUP_STEPS + k;
  const m = baseMetrics(sweep, step, k, isFinal);
  m.reconstructionError = r.errMse;      // σ_{k+1}²/n — the loss-curve metric
  m.frobErrK = r.errFro;                 // σ_{k+1} (Eckart-Young, Frobenius)
  m.spectralErrK = r.errSpectral;        // σ_{k+1} (spectral norm)
  const events: SimState['events'] = [
    { type: 'rank', label: `rank-${k}-reconstruction`, step },
    ...(isFinal ? [{ type: 'converged', label: 'svd-rank-sweep-complete', step }] : []),
  ];
  const math: MathStep[] = [
    { latex: '\\hat{X}_k = \\sum_{j \\le k} \\sigma_j u_j v_j^T = U_k \\Sigma_k V_k^T', id: 'svd-eckart-young' },
    { latex: '\\|X - \\hat{X}_k\\|_F = \\sigma_{k+1}, \\quad \\|X - \\hat{X}_k\\|_2 = \\sigma_{k+1}', id: 'svd-eckart-young-error' },
  ];
  const narration =
    k === 1
      ? `Rank 1: keep only σ₁u₁v₁ᵀ — X̂₁ = σ₁·u₁·v₁ᵀ is the best rank-1 approximation (Eckart–Young). ` +
        `Measured residual: ‖X − X̂₁‖_F = ${r.errFro.toFixed(6)} = σ₂ = ${c.sigma2.toFixed(6)} (spectral norm ${r.errSpectral.toFixed(6)}); ` +
        `per-sample mean squared error = ${r.errMse.toFixed(6)} = σ₂²/n.`
      : `Rank 2 (full): X̂₂ = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ spans the whole plane — the reconstruction is EXACT: ` +
        `‖X − X̂₂‖_F = ${r.errFro.toFixed(9)} (≈ 0 to machine precision), error = σ₃ = 0.`;
  return {
    algorithm: {
      mode: k === sweep.ranks.length ? 'svd-full-rank' : 'svd-rank-1', step, rank: k, isOptimal: isFinal ? 1 : 0,
      v1x: c.V1[0], v1y: c.V1[1], v2x: c.V2[0], v2y: c.V2[1],
      sigma1: c.sigma1, sigma2: c.sigma2, dataSeed: sweep.dataSeed,
    } as Record<string, ParamValue>,
    visuals: [
      ...buildEigenviewer(sweep),
      ...buildScatter(sweep, isFinal),
      ...buildMatrices(sweep, 4),
      ...buildRankMatrices(sweep, k),
    ],
    math,
    narration,
    explanation: {
      changed: k === 1
        ? [`X̂₁ (n×2) built`, `‖X−X̂₁‖_F → ${r.errFro.toFixed(6)} = σ₂`, `error → ${r.errMse.toFixed(6)}`]
        : [`X̂₂ (n×2) built`, `‖X−X̂₂‖_F → ${r.errFro.toFixed(9)}`, `error → 0`],
      why: k === 1
        ? `By Eckart–Young, the rank-k matrix minimizing ‖X − X̂‖ over all rank-k matrices is the truncated SVD X̂_k = U_kΣ_kV_kᵀ, and the residual norm is ` +
          `exactly the dropped singular value σ_{k+1}. The error is measured FROM the actual reconstructed matrix (not just the closed form) — ` +
          `the numeric identity the tests assert.`
        : `At full rank the truncated SVD IS the exact factorization, so the reconstruction is lossless — the residual is zero to machine precision. ` +
          `This is the SVD's version of PCA's "keeping both PCs reconstructs exactly".`,
      formulaRef: 'svd-eckart-young',
      dependsOn: ['linear-algebra', 'probability', 'statistics'],
      gateConcepts: ['SVD', 'Eckart-Young theorem', 'low-rank approximation', 'reconstruction'],
    },
    highlights: k === 1
      ? [
          { panel: 'matrix', id: `Σ_1 (1×1):0,0`, intensity: 1 },
          { panel: 'equation', id: 'svd-eckart-young-error', intensity: 1 },
        ]
      : [{ panel: 'matrix', id: 'X̂_2 (n×2):0,0', intensity: 0.5 }],
    metrics: m,
    events,
    timeline: k === 1 ? ['Reconstruct', 'Rank 1'] : ['Reconstruct', 'Full Rank'],
  };
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

export const simulation = {
  /**
   * Snapshot 1 = build-up phase 1 (X, XᵀX). Throws (honest telemetry failure)
   * when the data has zero variance (the `points` override degenerate case) —
   * computeRun records failedAtStep, mirroring pca's zero-variance path.
   */
  initialState: (p: Params): SimState => {
    const sweep = cachedSweep(p);
    return buildupSnapshot(sweep, 1, true);
  },

  /**
   * Advance the build-up phases 2..4, then the rank sweep k = 1..params.rank
   * (the LAST rank snapshot is the slider rank exactly); then null.
   */
  step: (p: Params, s: SimState): SimState | null => {
    const sweep = cachedSweep(p);
    const current = (s.algorithm.step as number) ?? 1;
    const next = current + 1;
    if (next <= BUILDUP_STEPS) {
      return buildupSnapshot(sweep, next, false);
    }
    const kIdx = next - BUILDUP_STEPS - 1; // 0-based rank index
    if (kIdx >= 0 && kIdx < sweep.ranks.length) {
      const k = sweep.ranks[kIdx];
      return rankSnapshot(sweep, k, kIdx === sweep.ranks.length - 1);
    }
    return null;
  },
};

// ---------------------------------------------------------------------------
// Topic module + registration
// ---------------------------------------------------------------------------

export const svdModule: TopicModule = {
  id: 'pca-svd',
  title: 'Singular Value Decomposition (SVD)',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 4, mathematical: 5, coding: 3, visualization: 4, gateFrequency: 5 },
    estimatedHours: 7,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['linear-algebra', 'probability', 'statistics', 'pca'],
    relatedTopics: ['pca', 'lda', 'linear-regression', 'linear-algebra'],
    revision: { quick: '20m', standard: '1h', deep: '2h', mastery: '4h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'eigenviewer', title: 'Data Cloud, v₁ Axis, Projections & σ² Bars — reconstruct mode draws the low-rank error lines' },
      { slot: 'primary', component: 'scatter-plot', title: 'Data & Singular-Vector Axes — final step shows v₁ ⊥ v₂' },
      { slot: 'primary', component: 'loss-curve', title: 'Reconstruction Error vs Rank — lower = better; build-up shows the rank-0 baseline (full energy ‖X‖²_F/n), full rank → exactly 0 (Eckart–Young)' },
      { slot: 'sidebar', component: 'matrix-animator', title: 'X → XᵀX → V, Σ, U (with dims); rank steps add U_k, Σ_k, V_kᵀ and the reconstruction' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivations: XᵀX = VΣ²Vᵀ, U = XV/Σ, Eckart–Young' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Data → Center → Gram → Eigen → Sigma → U → Reconstruct → Full Rank' },
      { slot: 'sidebar', component: 'explain-step', title: 'Step Explanation' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'n', label: 'Points n', type: 'number', min: 3, max: 80, step: 1, default: 40 },
    { id: 'corr', label: 'Correlation ρ', type: 'number', min: -0.95, max: 0.95, step: 0.05, default: 0.7 },
    { id: 'rotDeg', label: 'Correlation angle φ (deg)', type: 'number', min: 0, max: 170, step: 5, default: 30 },
    { id: 'noise', label: 'Isotropic noise σ', type: 'number', min: 0, max: 1.5, step: 0.05, default: 0.15 },
    { id: 'rank', label: 'Reconstruction rank k', type: 'number', min: 1, max: 2, step: 1, default: 2 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
  ],
  simulation,
  formulas: svdFormulas,
  derivations: svdDerivations,
  questions: svdQuestions,
  comparisons: svdComparisons,
  failureDemos: svdFailureDemos,
  mistakes: svdMistakes,
  testCases: svdTestCases,
  lossMetricKey: 'reconstructionError',

  validateParams: (p) => {
    const issues: string[] = [];
    const n = p.n as number | undefined;
    if (n !== undefined) {
      // The plan's "nPerClass ≥ 2" guard maps to n ≥ 3 — the pca convention:
      // ≥ 3 points keep the sample covariance full-rank (2 points always give
      // a rank-1 design matrix and a meaningless second singular value).
      if (!Number.isInteger(n) || n < 3) {
        issues.push('n must be an integer ≥ 3 — SVD needs at least 3 non-collinear points so the design matrix has a meaningful second singular value (2 points always give rank 1)');
      }
      if (n > 80) issues.push('n > 80 exceeds the lightweight demo size (keep n ≤ 80 for smooth scrubbing)');
    }
    const corr = p.corr as number | undefined;
    if (corr !== undefined && !Number.isFinite(corr)) issues.push('corr must be a finite number');
    if (corr !== undefined && !(corr > -1 && corr < 1)) {
      issues.push('corr must be in (−1, 1) — |ρ| = 1 makes the base covariance singular (perfect correlation → σ₂ → 0, a degenerate second singular value)');
    }
    const rot = p.rotDeg as number | undefined;
    if (rot !== undefined && !(rot >= 0 && rot < 180)) issues.push('rotDeg must be in [0, 180) — the correlation rotation is defined mod 180°');
    const noise = p.noise as number | undefined;
    // Plan's "noise > 0" guard delivered as noise ≥ 0 (the pca convention —
    // noise = 0 exposes the near-degenerate σ₂ → 0 case below, a valuable
    // teaching moment for SVD too). Documented drift.
    if (noise !== undefined && !(noise >= 0)) issues.push('noise must be ≥ 0 (0 is allowed — it exposes the near-degenerate σ₂ ≈ 0 case)');
    if (noise !== undefined && noise > 1.5) issues.push('noise > 1.5 overwhelms the correlation structure (σ²I dominates Σ and the singular directions become meaningless)');
    if (noise === 0 && corr !== undefined && Math.abs(corr) >= 0.98) {
      issues.push('WARNING: noise = 0 with |corr| ≥ 0.98 gives a nearly degenerate covariance (σ₂ ≈ 0) — the second singular value carries no information; the SVD is still exact but rank 1 already reconstructs everything');
    }
    const rank = p.rank as number | undefined;
    if (rank !== undefined && (!Number.isInteger(rank) || rank < 1 || rank > MAX_RANK)) {
      issues.push(`rank must be an integer in [1, ${MAX_RANK}] — the design matrix has 2 columns, so the reconstruction rank is 1 or 2`);
    }
    const seed = p.seed as number | undefined;
    if (seed !== undefined && (!Number.isInteger(seed) || seed < 0 || seed > 9999)) issues.push('seed must be an integer in [0, 9999]');
    if (typeof p.points === 'string') {
      try {
        const rows = JSON.parse(p.points) as unknown;
        if (!Array.isArray(rows) || rows.length < 3 || !rows.every((r) => Array.isArray(r) && r.length === 2 && r.every(Number.isFinite))) {
          issues.push('points must be a JSON array of ≥ 3 [x, y] pairs');
        } else {
          const pts = (rows as [number, number][]).map(([x, y]) => ({ x, y }));
          const mu0 = pts.reduce((a, d) => a + d.x, 0) / pts.length;
          const mu1 = pts.reduce((a, d) => a + d.y, 0) / pts.length;
          let v = 0;
          for (const d of pts) v += (d.x - mu0) ** 2 + (d.y - mu1) ** 2;
          if (v < 1e-12) issues.push('points have zero variance (all points identical) — the SVD is undefined without any spread');
        }
      } catch {
        issues.push('points must be a valid JSON array of [x, y] pairs');
      }
    }
    return issues;
  },
};

export function register() {
  registerTopic(svdModule);
  // SVD is unsupervised — no classifier to register (the pca convention).
}
