// src/topics/mle/module.ts
// Task 18 (Wave 4): mle — Maximum Likelihood Estimation: point estimation for
// three dataset families (Bernoulli coin flips, Gaussian samples, linear
// regression with Gaussian noise) via EXACT closed-form MLEs (no iterative
// solver), animated by a deterministic growing-n sweep that demonstrates
// consistency and the ÷n-vs-÷(n−1) variance bias with measured numbers.
//
// Design decisions (deviations from the plan are documented in the report):
//  - Dataset families ('coin' | 'gaussian' | 'linear') selected by the `family`
//    param; `n` (schema 10..1000, guard n ≥ 2) and `seed` drive a deterministic
//    mulberry32 stream. The stream generates exactly `n` samples; every
//    snapshot's MLE is computed on the PREFIX of size n_k, so snapshot k's data
//    is a strict subset of snapshot k+1's — the consistency story is honest
//    (same draws, strictly more data; the PRNG is never re-seeded mid-run).
//  - Step model: GROWING-N SWEEP. The run visits the SWEEP_SIZES = [10, 30,
//    100, 300, 1000] that are ≤ the requested n, then (if n is off-grid) one
//    final snapshot at exactly n. n = 1000 → 5 snapshots, n = 10 → 1 snapshot.
//    The final snapshot is ALWAYS exactly the requested n (test cases pin the
//    endpoint). lossMetricKey = 'nllPerSample' — the average negative
//    log-likelihood at the MLE (= the cross-entropy of the fitted model on the
//    sample), which descends toward the true entropy as n grows.
//  - Math: exact closed forms — p̂ = k/n (Bernoulli); μ̂ = x̄ and
//    σ̂² = Σ(x−μ̂)²/n (Gaussian, the BIASED ÷n MLE — the plan's required
//    distinction; the unbiased ÷(n−1) estimator is computed alongside so the
//    gap σ̂²u − σ̂² = σ²/n (small n) can be measured and watched shrink);
//    θ̂ = (XᵀX)⁻¹Xᵀy via the 2×2 adjugate (linear, x ~ U(−3, 3), noise
//    N(0, noiseSigma²)), with MLE noise variance σ̂² = RSS/n and the unbiased
//    RSS/(n−2). Score functions: S(p) = k/p − (n−k)/(1−p), S(μ) = n(μ̂−μ)/σ²,
//    S(σ²) = −n/(2σ²) + Σ(x−μ̂)²/(2σ⁴), S(θ) = Xᵀ(y−Xθ); tests assert the
//    score vanishes at the MLE (numeric gradient check on ℓ vs L).
//  - LIKELIHOOD SURFACE (plan): the plan's "ℓ(μ, σ²) 2D contour + gradient
//    arrows" needs loss-landscape (Wave 6) — NOT available. Self-contained
//    substitution (documented drift): the gaussian family emits the ℓ(μ, σ²)
//    grid as a matrix-animator command (9 μ-rows × 9 σ²-cols, numeric
//    heatmap-style grid; the ℓ(μ, σ²) argmax over μ is exactly the center row
//    μ̂ for every σ², asserted in the tests). The distribution-view renders
//    Gaussian densities ONLY, so (a) the Bernoulli PMF cannot be drawn there —
//    the coin family instead shows the raw flips plus p̂/pTrue reference lines
//    (drift), and (b) ℓ(μ) "curve slices" are not emitted as distributions —
//    the numeric grid is the single honest surface channel.
//  - Non-identifiability (failure): the linear family with a hand-crafted
//    degenerate design (all x equal → det XᵀX = 0) makes initialState THROW
//    the honest telemetry failure (lda's singular-S_W precedent); the failure
//    demo prints the measured flat-likelihood numbers (any θ on the line
//    β₀ + β₁·c = const gives identical predictions, RSS and ℓ).
//  - Test-only params (cf. lda's toy/points precedent): `points` JSON string
//    '[[x,y],…]' overrides the linear data (failure demos); not in the schema.
import type { TopicModule, Params, SimState, VisualCommand, ParamValue, MathStep } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { mleTestCases } from './testCases';
import { mleFormulas } from './formulas';
import { mleDerivations } from './derivations';
import { mleMistakes } from './mistakes';
import { mleQuestions } from './questions';
import { mleComparisons } from './comparisons';
import { mleFailureDemos } from './failures';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Growing-n sweep sizes (visits every size ≤ requested n in ascending order). */
export const SWEEP_SIZES = [10, 30, 100, 300, 1000] as const;
/** Schema cap for n (also the stream cap — matches the largest sweep size). */
export const MAX_N = 1000;
/** Singularity floor for det(XᵀX) — a degenerate linear design is not identifiable. */
export const DET_FLOOR = 1e-12;
/** Clamp for log(p) / log(1−p) at the Bernoulli boundary (k = 0 or k = n). */
export const P_LOG_FLOOR = 1e-12;
/** Default dataset family. */
export const DEFAULT_FAMILY = 'coin';

export const FITTED_COLOR = '#3b82f6';   // fitted / estimate color (blue)
export const TRUE_COLOR = '#ef4444';     // true parameter / line color (red)
export const HEADS_COLOR = '#3b82f6';    // coin heads
export const TAILS_COLOR = '#94a3b8';    // coin tails (slate)
export const GRID_COLOR = '#0f172a';     // neutral (axis / narration accents)

// ---------------------------------------------------------------------------
// Deterministic PRNG + data stream
// ---------------------------------------------------------------------------

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

export interface MleStream {
  family: 'coin' | 'gaussian' | 'linear';
  n: number;                 // total draws available (= requested n, or points length)
  coin: number[];            // 0/1 flips (family 'coin')
  gauss: number[];           // Gaussian samples (family 'gaussian')
  linX: number[];            // linear family x (U(−3, 3)) / crafted override
  linY: number[];            // linear family y = b0 + b1·x + noise / crafted override
}

/**
 * Deterministic data synthesis. The stream has exactly `n` draws; snapshots
 * consume prefixes of it. For the linear family the test-only `points` JSON
 * string ('[[x,y],…]') overrides the generated design (failure demos).
 */
export function generateStream(p: Params): MleStream {
  const family = (p.family as string) ?? DEFAULT_FAMILY;
  if (typeof p.points === 'string') {
    const rows = JSON.parse(p.points) as [number, number][];
    return {
      family: 'linear', n: rows.length,
      coin: [], gauss: [],
      linX: rows.map((r) => r[0]), linY: rows.map((r) => r[1]),
    };
  }
  const n = Math.max(2, Math.round((p.n as number) ?? 100));
  const seed = (p.seed as number) ?? 42;
  const rng = mulberry32(seed);
  const normal = makeNormal(rng);
  if (family === 'gaussian') {
    const mu = (p.muTrue as number) ?? 1;
    const sigma = (p.sigmaTrue as number) ?? 1.5;
    const gauss: number[] = [];
    for (let i = 0; i < n; i++) gauss.push(mu + sigma * normal());
    return { family: 'gaussian', n, coin: [], gauss, linX: [], linY: [] };
  }
  if (family === 'linear') {
    const slope = (p.slopeTrue as number) ?? 1.5;
    const intercept = (p.interceptTrue as number) ?? -0.5;
    const noise = (p.noiseSigma as number) ?? 0.8;
    const linX: number[] = [];
    const linY: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = -3 + 6 * rng();            // x ~ U(−3, 3): spread design, well-conditioned XᵀX
      linX.push(x);
      linY.push(intercept + slope * x + noise * normal());
    }
    return { family: 'linear', n, coin: [], gauss: [], linX, linY };
  }
  // coin (default family)
  const pTrue = (p.pTrue as number) ?? 0.7;
  const coin: number[] = [];
  for (let i = 0; i < n; i++) coin.push(rng() < pTrue ? 1 : 0);
  return { family: 'coin', n, coin, gauss: [], linX: [], linY: [] };
}

// ---------------------------------------------------------------------------
// MLE math core (all pure + exported for hand-verified tests)
// ---------------------------------------------------------------------------

export interface BernoulliFit { k: number; n: number; pHat: number; }
export interface GaussianFitStats { n: number; muHat: number; sigmaHatSq: number; sigmaUnbSq: number; }
export interface LinearFit {
  n: number; slopeHat: number; interceptHat: number; rss: number;
  sigmaHatSq: number; sigmaUnbSq: number; detXX: number;
}

/** Bernoulli MLE: p̂ = k/n — the empirical frequency of heads. */
export function bernoulliMle(values: number[]): BernoulliFit {
  const n = values.length;
  let k = 0;
  for (const v of values) k += v;
  return { k, n, pHat: k / n };
}

/**
 * Gaussian MLE: μ̂ = x̄ and σ̂² = Σ(x−μ̂)²/n — the BIASED (÷n) estimator, by
 * construction (∂ℓ/∂σ² = 0). The unbiased ÷(n−1) estimator is reported
 * alongside so the bias gap can be measured; for n = 2 it is exactly 2×σ̂².
 */
export function gaussianMle(values: number[]): GaussianFitStats {
  const n = values.length;
  const muHat = values.reduce((a, b) => a + b, 0) / n;
  let ss = 0;
  for (const v of values) ss += (v - muHat) ** 2;
  const sigmaHatSq = ss / n;
  const sigmaUnbSq = n > 1 ? ss / (n - 1) : sigmaHatSq;
  return { n, muHat, sigmaHatSq, sigmaUnbSq };
}

/**
 * Linear-regression MLE (= OLS) via the exact 2×2 normal equation:
 * θ̂ = (XᵀX)⁻¹Xᵀy with X = [1 x]. det = n·Σx² − (Σx)². The MLE noise variance
 * is σ̂² = RSS/n (÷n); the unbiased estimator RSS/(n−2) is reported alongside.
 * Throws an honest error when the design is degenerate (det ≤ DET_FLOOR) —
 * slope and intercept are not identifiable there (all x equal).
 */
export function linearMle(xs: number[], ys: number[]): LinearFit {
  const n = xs.length;
  let sx = 0, sxx = 0, sy = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i]; sxx += xs[i] * xs[i]; sy += ys[i]; sxy += xs[i] * ys[i];
  }
  const det = n * sxx - sx * sx;
  if (det <= DET_FLOOR) {
    throw new Error(
      `mle: design matrix XᵀX is singular (det = ${det.toExponential(3)}) — slope and intercept are not ` +
      'identifiable (all x equal, or too few distinct x values); the likelihood is flat along the direction ' +
      'β₀ + β₁·x̄ = const. Add spread to x or more samples.',
    );
  }
  const slopeHat = (n * sxy - sx * sy) / det;
  const interceptHat = (sxx * sy - sx * sxy) / det;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - interceptHat - slopeHat * xs[i];
    rss += r * r;
  }
  return {
    n, slopeHat, interceptHat, rss,
    sigmaHatSq: rss / n,
    sigmaUnbSq: n > 2 ? rss / (n - 2) : rss / n,
    detXX: det,
  };
}

// ---- log-likelihood / score / grid (exported for numeric tests) -----------

/** Clamp p into the safe log domain. */
export function clampP(p: number): number {
  return Math.min(Math.max(p, P_LOG_FLOOR), 1 - P_LOG_FLOOR);
}

/** Bernoulli log-likelihood at p for k heads out of n (0·ln 0 handled by clamp). */
export function bernoulliLogLik(k: number, n: number, p: number): number {
  const pc = clampP(p);
  return k * Math.log(pc) + (n - k) * Math.log(1 - pc);
}

/** Bernoulli likelihood L(p) = p^k(1−p)^(n−k) — underflows for large n (failure demo). */
export function bernoulliLikelihood(k: number, n: number, p: number): number {
  return Math.exp(bernoulliLogLik(k, n, p));
}

/** Bernoulli score S(p) = dℓ/dp = k/p − (n−k)/(1−p). */
export function bernoulliScore(k: number, n: number, p: number): number {
  const pc = clampP(p);
  return k / pc - (n - k) / (1 - pc);
}

/** Gaussian log-likelihood of a sample at (μ, σ²). */
export function gaussianLogLik(values: number[], mu: number, sigmaSq: number): number {
  const n = values.length;
  if (!(sigmaSq > 0) || !Number.isFinite(sigmaSq)) return -Infinity;
  let ss = 0;
  for (const v of values) { const d = v - mu; ss += d * d; }
  return -0.5 * n * Math.log(2 * Math.PI * sigmaSq) - ss / (2 * sigmaSq);
}

/** Linear score vector S(θ) = Xᵀ(y − Xθ) — the matrix score (2×1). */
export function linearScoreVec(xs: number[], ys: number[], b0: number, b1: number): [number, number] {
  let s0 = 0, s1 = 0;
  for (let i = 0; i < xs.length; i++) {
    const r = ys[i] - b0 - b1 * xs[i];
    s0 += r;          // X₀ᵀ r = 1ᵀ r = Σ residual
    s1 += xs[i] * r;  // X₁ᵀ r = Σ x·residual
  }
  return [s0, s1];
}

/** Linear log-likelihood at (b0, b1, σ²): −n/2·ln(2πσ²) − RSS/(2σ²). */
export function linearLogLik(xs: number[], ys: number[], b0: number, b1: number, sigmaSq: number): number {
  if (!(sigmaSq > 0) || !Number.isFinite(sigmaSq)) return -Infinity;
  let rss = 0;
  for (let i = 0; i < xs.length; i++) {
    const r = ys[i] - b0 - b1 * xs[i];
    rss += r * r;
  }
  return -0.5 * xs.length * Math.log(2 * Math.PI * sigmaSq) - rss / (2 * sigmaSq);
}

/**
 * The ℓ(μ, σ²) grid over the gaussian sample — the plan's likelihood-surface
 * substitution (loss-landscape is Wave 6). 9 μ-rows centered on μ̂ (±2σ̂) and
 * 9 σ²-cols spanning [max(0.1·σ̂², ε), 3·σ̂²]. Because ∂ℓ/∂μ ∝ (μ̂−μ), the
 * argmax over μ is EXACTLY the center row (μ = μ̂) for every σ² — asserted in
 * the tests, so the grid honestly locates the MLE.
 */
export function likelihoodGrid(values: number[], muHat: number, sigmaHatSq: number): number[][] {
  const sigma = Math.sqrt(sigmaHatSq);
  const muLo = muHat - 2 * sigma;
  const muHi = muHat + 2 * sigma;
  const sigLo = Math.max(0.1 * sigmaHatSq, 1e-12);
  const sigHi = 3 * sigmaHatSq;
  const grid: number[][] = [];
  for (let r = 0; r < 9; r++) {
    const mu = muLo + (muHi - muLo) * (r / 8);
    const row: number[] = [];
    for (let c = 0; c < 9; c++) {
      const sig2 = sigLo + (sigHi - sigLo) * (c / 8);
      row.push(gaussianLogLik(values, mu, sig2));
    }
    grid.push(row);
  }
  return grid;
}

// ---------------------------------------------------------------------------
// Sweep + snapshot data
// ---------------------------------------------------------------------------

/** The growing-n sizes for the requested n: every SWEEP_SIZES ≤ n, then n itself. */
export function sweepSizesOf(p: Params): number[] {
  const n = Math.max(2, Math.round((p.n as number) ?? 100));
  const sizes: number[] = SWEEP_SIZES.filter((s) => s <= n);
  if (sizes[sizes.length - 1] !== n) sizes.push(n);
  return sizes;
}

export interface SweepStep {
  n: number;                       // prefix size
  // coin
  k: number; pHat: number; pTrue: number;
  // gaussian
  muHat: number; sigmaHatSq: number; sigmaUnbSq: number; muTrue: number; sigmaTrueSq: number;
  // linear
  slopeHat: number; interceptHat: number; slopeTrue: number; interceptTrue: number;
  rss: number; sigmaHatSqL: number; sigmaUnbSqL: number; noiseSigmaSq: number;
  // common
  nllPerSample: number;
  grid: number[][];                // gaussian: ℓ(μ, σ²) 9×9; others: empty
}

/** Compute the full MLE story for one prefix size n_k of the stream. */
export function fitPrefix(p: Params, stream: MleStream, n: number): SweepStep {
  const family = stream.family;
  if (family === 'gaussian') {
    const vals = stream.gauss.slice(0, n);
    const f = gaussianMle(vals);
    const nll = 0.5 * Math.log(2 * Math.PI * f.sigmaHatSq) + 0.5;
    return {
      n, k: 0, pHat: 0, pTrue: 0,
      muHat: f.muHat, sigmaHatSq: f.sigmaHatSq, sigmaUnbSq: f.sigmaUnbSq,
      muTrue: (p.muTrue as number) ?? 1, sigmaTrueSq: (((p.sigmaTrue as number) ?? 1.5) ** 2),
      slopeHat: 0, interceptHat: 0, slopeTrue: 0, interceptTrue: 0,
      rss: 0, sigmaHatSqL: 0, sigmaUnbSqL: 0, noiseSigmaSq: 0,
      nllPerSample: nll,
      grid: likelihoodGrid(vals, f.muHat, f.sigmaHatSq),
    };
  }
  if (family === 'linear') {
    const xs = stream.linX.slice(0, n);
    const ys = stream.linY.slice(0, n);
    const f = linearMle(xs, ys);   // throws honestly on a degenerate design
    const nll = 0.5 * Math.log(2 * Math.PI * f.sigmaHatSq) + 0.5;
    return {
      n, k: 0, pHat: 0, pTrue: 0,
      muHat: 0, sigmaHatSq: 0, sigmaUnbSq: 0, muTrue: 0, sigmaTrueSq: 0,
      slopeHat: f.slopeHat, interceptHat: f.interceptHat,
      slopeTrue: (p.slopeTrue as number) ?? 1.5, interceptTrue: (p.interceptTrue as number) ?? -0.5,
      rss: f.rss, sigmaHatSqL: f.sigmaHatSq, sigmaUnbSqL: f.sigmaUnbSq,
      noiseSigmaSq: (((p.noiseSigma as number) ?? 0.8) ** 2),
      nllPerSample: nll,
      grid: [],
    };
  }
  // coin
  const vals = stream.coin.slice(0, n);
  const f = bernoulliMle(vals);
  const nll = -bernoulliLogLik(f.k, f.n, f.pHat) / n;
  return {
    n, k: f.k, pHat: f.pHat, pTrue: (p.pTrue as number) ?? 0.7,
    muHat: 0, sigmaHatSq: 0, sigmaUnbSq: 0, muTrue: 0, sigmaTrueSq: 0,
    slopeHat: 0, interceptHat: 0, slopeTrue: 0, interceptTrue: 0,
    rss: 0, sigmaHatSqL: 0, sigmaUnbSqL: 0, noiseSigmaSq: 0,
    nllPerSample: nll,
    grid: [],
  };
}

// ---------------------------------------------------------------------------
// Metrics + visuals
// ---------------------------------------------------------------------------

function metricsOf(p: Params, st: SweepStep, step: number, family: string): Record<string, number> {
  const m: Record<string, number> = {
    step, n: st.n, family: family === 'coin' ? 0 : family === 'gaussian' ? 1 : 2,
    dataSeed: (p.seed as number) ?? 42,
    nllPerSample: st.nllPerSample,
  };
  if (family === 'coin') {
    m.k = st.k;
    m.pHat = st.pHat;
    m.pTrue = st.pTrue;
    m.pErr = Math.abs(st.pHat - st.pTrue);
    m.entropyTrue = -(st.pTrue * Math.log(st.pTrue) + (1 - st.pTrue) * Math.log(1 - st.pTrue));
  } else if (family === 'gaussian') {
    m.muHat = st.muHat;
    m.muTrue = st.muTrue;
    m.muErr = Math.abs(st.muHat - st.muTrue);
    m.sigmaHatSq = st.sigmaHatSq;       // BIASED ÷n — the MLE
    m.sigmaUnbSq = st.sigmaUnbSq;       // unbiased ÷(n−1)
    m.sigmaTrueSq = st.sigmaTrueSq;
    m.biasGap = st.sigmaUnbSq - st.sigmaHatSq;       // the ÷n vs ÷(n−1) gap
    m.biasGapRel = st.sigmaUnbSq > 0 ? (st.sigmaUnbSq - st.sigmaHatSq) / st.sigmaUnbSq : 0;
    m.sigmaErr = Math.abs(st.sigmaHatSq - st.sigmaTrueSq);
  } else {
    m.slopeHat = st.slopeHat;
    m.interceptHat = st.interceptHat;
    m.slopeTrue = st.slopeTrue;
    m.interceptTrue = st.interceptTrue;
    m.slopeErr = Math.abs(st.slopeHat - st.slopeTrue);
    m.interceptErr = Math.abs(st.interceptHat - st.interceptTrue);
    m.rss = st.rss;
    m.sigmaHatSq = st.sigmaHatSqL;      // MLE noise variance RSS/n (÷n)
    m.sigmaUnbSq = st.sigmaUnbSqL;      // unbiased RSS/(n−2)
    m.noiseSigmaSq = st.noiseSigmaSq;
    m.biasGap = st.sigmaUnbSqL - st.sigmaHatSqL;
    m.biasGapRel = st.sigmaUnbSqL > 0 ? (st.sigmaUnbSqL - st.sigmaHatSqL) / st.sigmaUnbSqL : 0;
  }
  return m;
}

function buildVisualsCoin(stream: MleStream, st: SweepStep): VisualCommand[] {
  const cmds: VisualCommand[] = stream.coin.slice(0, st.n).map((v, i) => ({
    type: 'point', id: `d${i}`, x: i, y: v, color: v === 1 ? HEADS_COLOR : TAILS_COLOR,
  }));
  cmds.push({ type: 'line', id: 'pHat-line', points: [[0, st.pHat], [st.n - 1, st.pHat]], color: FITTED_COLOR });
  cmds.push({ type: 'line', id: 'pTrue-line', points: [[0, st.pTrue], [st.n - 1, st.pTrue]], color: TRUE_COLOR });
  cmds.push({
    type: 'matrix', id: 'heads/tails', rows: 1, cols: 2,
    cells: [[st.k, st.n - st.k]],
  });
  cmds.push({ type: 'matrix', id: 'p̂ = k/n', rows: 1, cols: 1, cells: [[st.pHat]] });
  return cmds;
}

function buildVisualsGaussian(stream: MleStream, st: SweepStep): VisualCommand[] {
  const vals = [...stream.gauss.slice(0, st.n)].sort((a, b) => a - b);
  const cmds: VisualCommand[] = vals.map((v, i) => ({
    type: 'point', id: `d${i}`, x: i, y: v, color: GRID_COLOR,
  }));
  cmds.push({
    type: 'distribution', id: 'fitted-dist',
    label: `fitted N(μ̂=${st.muHat.toFixed(3)}, σ̂²=${st.sigmaHatSq.toFixed(3)})`,
    mean: st.muHat, variance: Math.max(st.sigmaHatSq, 1e-9), color: FITTED_COLOR,
  });
  cmds.push({
    type: 'distribution', id: 'true-dist',
    label: `true N(μ=${st.muTrue.toFixed(3)}, σ²=${st.sigmaTrueSq.toFixed(3)})`,
    mean: st.muTrue, variance: Math.max(st.sigmaTrueSq, 1e-9), color: TRUE_COLOR,
  });
  // the ℓ(μ, σ²) grid — the likelihood-surface substitution (loss-landscape is
  // Wave 6): rows = μ (μ̂−2σ̂ … μ̂+2σ̂), cols = σ² (0.1σ̂² … 3σ̂²)
  cmds.push({
    type: 'matrix', id: 'ℓ(μ,σ²) grid (rows: μ · cols: σ²)', rows: 9, cols: 9,
    cells: st.grid,
  });
  cmds.push({ type: 'matrix', id: 'μ̂ (MLE)', rows: 1, cols: 1, cells: [[st.muHat]] });
  cmds.push({ type: 'matrix', id: 'σ̂² = Σ(x−μ̂)²/n  (÷n)', rows: 1, cols: 1, cells: [[st.sigmaHatSq]] });
  cmds.push({ type: 'matrix', id: 'σ̂²ᵤ = Σ(x−μ̂)²/(n−1) (unbiased)', rows: 1, cols: 1, cells: [[st.sigmaUnbSq]] });
  return cmds;
}

function buildVisualsLinear(stream: MleStream, st: SweepStep): VisualCommand[] {
  const cmds: VisualCommand[] = stream.linX.slice(0, st.n).map((x, i) => ({
    type: 'point', id: `d${i}`, x, y: stream.linY[i], color: GRID_COLOR,
  }));
  const xmin = Math.min(...stream.linX.slice(0, st.n));
  const xmax = Math.max(...stream.linX.slice(0, st.n));
  const yHat = (x: number) => st.interceptHat + st.slopeHat * x;
  const yTrue = (x: number) => st.interceptTrue + st.slopeTrue * x;
  cmds.push({ type: 'line', id: 'fitted-line', points: [[xmin, yHat(xmin)], [xmax, yHat(xmax)]], color: FITTED_COLOR });
  cmds.push({ type: 'line', id: 'true-line', points: [[xmin, yTrue(xmin)], [xmax, yTrue(xmax)]], color: TRUE_COLOR });
  // XᵀX and Xᵀy for the normal equation θ̂ = (XᵀX)⁻¹Xᵀy
  let sx = 0, sxx = 0, sy = 0, sxy = 0;
  for (let i = 0; i < st.n; i++) {
    sx += stream.linX[i]; sxx += stream.linX[i] ** 2; sy += stream.linY[i]; sxy += stream.linX[i] * stream.linY[i];
  }
  cmds.push({
    type: 'matrix', id: 'XᵀX', rows: 2, cols: 2,
    cells: [[st.n, sx], [sx, sxx]],
  });
  cmds.push({ type: 'matrix', id: 'Xᵀy', rows: 2, cols: 1, cells: [[sy], [sxy]] });
  cmds.push({
    type: 'matrix', id: 'θ̂ = (XᵀX)⁻¹Xᵀy', rows: 2, cols: 1,
    cells: [[st.interceptHat], [st.slopeHat]],
  });
  cmds.push({ type: 'matrix', id: 'σ̂² = RSS/n  (÷n)', rows: 1, cols: 1, cells: [[st.sigmaHatSqL]] });
  cmds.push({ type: 'matrix', id: 'σ̂²ᵤ = RSS/(n−2) (unbiased)', rows: 1, cols: 1, cells: [[st.sigmaUnbSqL]] });
  return cmds;
}

function mathOf(family: string, _st: SweepStep): MathStep[] {
  if (family === 'coin') {
    return [
      { latex: 'L(p) = p^k (1-p)^{n-k}', id: 'mle-likelihood' },
      { latex: '\\ell(p) = k\\ln p + (n-k)\\ln(1-p)', id: 'mle-loglik' },
      { latex: 'S(p) = \\frac{k}{p} - \\frac{n-k}{1-p} = 0 \\Rightarrow \\hat p = \\frac{k}{n}', id: 'mle-bernoulli' },
    ];
  }
  if (family === 'gaussian') {
    return [
      { latex: '\\ell(\\mu, \\sigma^2) = -\\frac{n}{2}\\ln(2\\pi\\sigma^2) - \\frac{1}{2\\sigma^2}\\sum_i (x_i - \\mu)^2', id: 'mle-loglik' },
      { latex: '\\frac{\\partial\\ell}{\\partial\\mu} = 0 \\Rightarrow \\hat\\mu = \\bar x', id: 'mle-gaussian-mean' },
      { latex: '\\frac{\\partial\\ell}{\\partial\\sigma^2} = 0 \\Rightarrow \\hat\\sigma^2 = \\frac{1}{n}\\sum_i (x_i-\\hat\\mu)^2', id: 'mle-gaussian-var' },
    ];
  }
  return [
    { latex: '\\ell(\\theta) = -\\frac{n}{2}\\ln(2\\pi\\sigma^2) - \\frac{\\|y - X\\theta\\|^2}{2\\sigma^2}', id: 'mle-loglik' },
    { latex: 'S(\\theta) = X^T(y - X\\theta) = 0 \\Rightarrow \\hat\\theta = (X^T X)^{-1} X^T y', id: 'mle-score-matrix' },
    { latex: '\\hat\\theta = (X^T X)^{-1}X^T y = \\text{normal equation (OLS)}', id: 'mle-ols' },
  ];
}

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

function narrationOf(family: string, st: SweepStep, isFinal: boolean): string {
  if (family === 'coin') {
    return `The likelihood L(p) = p^k(1−p)^{n−k} is maximized where its derivative (the score) vanishes. ` +
      `With n = ${st.n} flips the score k/p − (n−k)/(1−p) = 0 gives p̂ = k/n = ${st.pHat.toFixed(4)} ` +
      `(${st.k} heads). The true p = ${st.pTrue.toFixed(2)}; the estimate ${isFinal ? 'at the requested n ' : ''}` +
      `is ${Math.abs(st.pHat - st.pTrue) < 0.02 ? 'already' : 'still'} tracking it.`;
  }
  if (family === 'gaussian') {
    return `Setting both partial scores to zero: μ̂ = x̄ = ${st.muHat.toFixed(4)} and ` +
      `σ̂² = Σ(x−μ̂)²/n = ${st.sigmaHatSq.toFixed(4)} — note the ÷n (the MLE's small-sample bias; ` +
      `the unbiased ÷(n−1) estimator would be ${st.sigmaUnbSq.toFixed(4)}, gap ${(st.sigmaUnbSq - st.sigmaHatSq).toFixed(4)}). ` +
      `True parameters μ = ${st.muTrue.toFixed(2)}, σ² = ${st.sigmaTrueSq.toFixed(2)}; the fitted curve ${isFinal ? 'now ' : ''}` +
      `sits ${Math.abs(st.sigmaUnbSq - st.sigmaTrueSq) < 0.05 ? 'close to' : 'on top of / beside'} the true one.`;
  }
  return `The matrix score Xᵀ(y − Xθ) = 0 is the stationarity condition — solving the normal equation ` +
    `θ̂ = (XᵀX)⁻¹Xᵀy exactly: ŷ = ${st.interceptHat.toFixed(4)} + ${st.slopeHat.toFixed(4)}·x vs the true ` +
    `${st.interceptTrue.toFixed(2)} + ${st.slopeTrue.toFixed(2)}·x. Residual variance σ̂² = RSS/n = ` +
    `${st.sigmaHatSqL.toFixed(4)} (unbiased ÷(n−2): ${st.sigmaUnbSqL.toFixed(4)}); OLS and MLE are the same estimator.`;
}

function snapshotAt(p: Params, stream: MleStream, st: SweepStep, step: number, isFinal: boolean): SimState {
  const family = stream.family;
  const metrics = metricsOf(p, st, step, family);
  const changed: string[] = [];
  if (family === 'coin') {
    changed.push(`n → ${st.n}`, `p̂ → ${st.pHat.toFixed(4)}`);
  } else if (family === 'gaussian') {
    changed.push(`n → ${st.n}`, `μ̂ → ${st.muHat.toFixed(4)}`, `σ̂² → ${st.sigmaHatSq.toFixed(4)}`);
  } else {
    changed.push(`n → ${st.n}`, `ŷ = ${st.interceptHat.toFixed(3)} + ${st.slopeHat.toFixed(3)}x`);
  }
  const events: SimState['events'] = step === 1
    ? [
        { type: 'init', label: 'mle-data-generated', step },
        { type: 'candidate', label: `n-${st.n}-samples`, step },
      ]
    : isFinal
      ? [{ type: 'candidate', label: `n-${st.n}-samples`, step }, { type: 'converged', label: 'mle-at-requested-n', step }]
      : [{ type: 'candidate', label: `n-${st.n}-samples`, step }];

  const visuals = family === 'coin'
    ? buildVisualsCoin(stream, st)
    : family === 'gaussian' ? buildVisualsGaussian(stream, st) : buildVisualsLinear(stream, st);

  const why = family === 'coin'
    ? `MLE sets the score S(p) = dℓ/dp to zero. S(p) = k/p − (n−k)/(1−p) = 0 ⟺ p = k/n, the empirical frequency. Each sweep step adds strictly more flips from the SAME seeded stream, so p̂ is a running frequency that tightens toward p.`
    : family === 'gaussian'
      ? `The log-likelihood is a concave function of (μ, σ²); its unique maximizer is the closed form μ̂ = x̄, σ̂² = Σ(x−μ̂)²/n. The ÷n (not ÷(n−1)) is forced by the MLE stationarity condition itself.`
      : `Under Gaussian noise the log-likelihood is −‖y−Xθ‖²/(2σ²) plus constants, so its maximizer is the least-squares solution — the normal equation. The score Xᵀ(y−Xθ) is exactly zero at θ̂, which the tests verify numerically.`;

  return {
    algorithm: {
      family, step, n: st.n,
      // fitted state (decision-boundary-style param merge is not used here —
      // kept as plain metrics for tests + narration)
      muHat: st.muHat, sigmaHatSq: st.sigmaHatSq, sigmaUnbSq: st.sigmaUnbSq,
      slopeHat: st.slopeHat, interceptHat: st.interceptHat, pHat: st.pHat, k: st.k,
    } as Record<string, ParamValue>,
    visuals,
    math: mathOf(family, st),
    narration: narrationOf(family, st, isFinal),
    explanation: {
      changed,
      why,
      formulaRef: family === 'coin' ? 'mle-bernoulli' : family === 'gaussian' ? 'mle-gaussian-var' : 'mle-score-matrix',
      dependsOn: ['probability', 'calculus', 'statistics'],
      gateConcepts: ['maximum likelihood estimation', 'log-likelihood', 'score function', 'consistency', 'OLS'],
    },
    highlights: family === 'coin'
      ? [{ panel: 'canvas', id: 'pHat-line', intensity: 1 }]
      : family === 'gaussian'
        ? [{ panel: 'canvas', id: 'fitted-dist', intensity: 1 }]
        : [{ panel: 'canvas', id: 'fitted-line', intensity: 1 }],
    metrics,
    events,
    timeline: step === 1
      ? ['Data', 'MLE Fit', 'Evaluate']
      : isFinal ? ['MLE Fit', 'Evaluate', 'MLE at n'] : ['Data', 'MLE Fit', 'Evaluate'],
  };
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

export const simulation = {
  /**
   * Snapshot 1 = the smallest sweep size ≤ requested n. Throws (honest
   * telemetry failure) when the linear design is degenerate — the
   * non-identifiable-parameters path (svm-hard-margin / lda precedent).
   */
  initialState: (p: Params): SimState => {
    const stream = generateStream(p);
    const sizes = sweepSizesOf(p);
    const st = fitPrefix(p, stream, sizes[0]);
    return snapshotAt(p, stream, st, 1, sizes.length === 1);
  },

  /** Advance the growing-n sweep; null when the requested n is reached. */
  step: (p: Params, s: SimState): SimState | null => {
    const stream = generateStream(p);
    const sizes = sweepSizesOf(p);
    const idx = (s.algorithm.step as number) ?? 1;
    const next = idx + 1;
    if (next > sizes.length) return null;
    const st = fitPrefix(p, stream, sizes[next - 1]);
    return snapshotAt(p, stream, st, next, next === sizes.length);
  },
};

// ---------------------------------------------------------------------------
// Topic module + registration
// ---------------------------------------------------------------------------

export const mleModule: TopicModule = {
  id: 'mle',
  title: 'Maximum Likelihood Estimation',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 4, mathematical: 4, coding: 2, visualization: 3, gateFrequency: 5 },
    estimatedHours: 6,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['probability', 'calculus', 'statistics', 'gaussian-distribution'],
    relatedTopics: ['simple-linear-regression', 'cross-entropy-loss', 'ridge-regression', 'naive-bayes', 'logistic-regression'],
    revision: { quick: '20m', standard: '1h', deep: '2h', mastery: '4h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Data, MLE Marker vs True Parameter' },
      { slot: 'primary', component: 'distribution-view', title: 'Fitted vs True Distribution (Gaussian family)' },
      { slot: 'primary', component: 'loss-curve', title: 'Average Negative Log-Likelihood at the MLE (CE link; lower = better)' },
      { slot: 'sidebar', component: 'matrix-animator', title: 'ℓ(μ, σ²) Grid, Estimates & the ÷n Bias' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivations: Bernoulli, Gaussian (μ, σ²), Linear via Score = 0' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Data → MLE Fit → Evaluate (growing n)' },
      { slot: 'sidebar', component: 'explain-step', title: 'Step Explanation' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    {
      id: 'family', label: 'Dataset family', type: 'select', default: 'coin',
      options: [
        { value: 'coin', label: 'Coin flips (Bernoulli)' },
        { value: 'gaussian', label: 'Gaussian samples N(μ, σ²)' },
        { value: 'linear', label: 'Linear regression + Gaussian noise' },
      ],
    },
    { id: 'n', label: 'Sample count n', type: 'number', min: 10, max: 1000, step: 10, default: 100 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
    { id: 'pTrue', label: 'True p (coin)', type: 'number', min: 0.05, max: 0.95, step: 0.05, default: 0.7 },
    { id: 'muTrue', label: 'True μ (gaussian)', type: 'number', min: -3, max: 3, step: 0.25, default: 1 },
    { id: 'sigmaTrue', label: 'True σ (gaussian)', type: 'number', min: 0.5, max: 3, step: 0.25, default: 1.5 },
    { id: 'slopeTrue', label: 'True slope (linear)', type: 'number', min: -2, max: 2, step: 0.25, default: 1.5 },
    { id: 'interceptTrue', label: 'True intercept (linear)', type: 'number', min: -3, max: 3, step: 0.25, default: -0.5 },
    { id: 'noiseSigma', label: 'Noise σ (linear)', type: 'number', min: 0.2, max: 2, step: 0.1, default: 0.8 },
  ],
  simulation,
  formulas: mleFormulas,
  derivations: mleDerivations,
  questions: mleQuestions,
  comparisons: mleComparisons,
  failureDemos: mleFailureDemos,
  mistakes: mleMistakes,
  testCases: mleTestCases,
  lossMetricKey: 'nllPerSample',

  validateParams: (p) => {
    const issues: string[] = [];
    const family = (p.family as string | undefined) ?? DEFAULT_FAMILY;
    if (family !== undefined && !['coin', 'gaussian', 'linear'].includes(family)) {
      issues.push('family must be one of "coin", "gaussian", "linear"');
    }
    const n = p.n as number | undefined;
    if (n !== undefined) {
      if (!Number.isInteger(n) || n < 2) issues.push('n must be an integer ≥ 2 — MLE needs at least 2 samples');
      if (n > MAX_N) issues.push(`n > ${MAX_N} exceeds the lightweight demo size (keep n ≤ ${MAX_N})`);
      if (family === 'linear' && n < 3) issues.push('linear family needs n ≥ 3 (slope + intercept + residual variance need ≥ 3 points)');
    }
    const seed = p.seed as number | undefined;
    if (seed !== undefined && (!Number.isInteger(seed) || seed < 0 || seed > 9999)) issues.push('seed must be an integer in [0, 9999]');
    const pTrue = p.pTrue as number | undefined;
    if (pTrue !== undefined && !(pTrue > 0 && pTrue < 1)) issues.push('pTrue must be in (0, 1) — at p = 0 or 1 the Bernoulli likelihood is degenerate (k/n or (n−k)/n is 1)');
    const muTrue = p.muTrue as number | undefined;
    if (muTrue !== undefined && !Number.isFinite(muTrue)) issues.push('muTrue must be finite');
    const sigmaTrue = p.sigmaTrue as number | undefined;
    if (sigmaTrue !== undefined && !(sigmaTrue > 0)) issues.push('sigmaTrue must be positive');
    const slopeTrue = p.slopeTrue as number | undefined;
    if (slopeTrue !== undefined && !Number.isFinite(slopeTrue)) issues.push('slopeTrue must be finite');
    const interceptTrue = p.interceptTrue as number | undefined;
    if (interceptTrue !== undefined && !Number.isFinite(interceptTrue)) issues.push('interceptTrue must be finite');
    const noise = p.noiseSigma as number | undefined;
    if (noise !== undefined && !(noise > 0)) issues.push('noiseSigma must be positive');
    if (typeof p.points === 'string') {
      try {
        const rows = JSON.parse(p.points) as unknown;
        if (!Array.isArray(rows) || rows.length < 3 ||
            !rows.every((r) => Array.isArray(r) && r.length === 2 && r.every(Number.isFinite))) {
          issues.push('points must be a JSON array of [x, y] pairs with ≥ 3 rows (test-only linear override)');
        }
      } catch {
        issues.push('points must be a valid JSON array of [x, y] pairs (test-only linear override)');
      }
    }
    return issues;
  },
};

export function register() {
  registerTopic(mleModule);
}
