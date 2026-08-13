// src/topics/svm-hard-margin/module.ts
// Task 11 (Wave 3): svm-hard-margin — 2D binary classification by the hard-margin
// (maximum-margin) SVM separator, solved with an EXACT 2D geometric solver
// (no QP, no gradient descent — the plan's mandated approach).
//
// Design decisions (deviations from the plan are documented in the report):
//  - Solver: candidate-direction enumeration. In 2D the optimal max-margin
//    hyperplane's normal is determined by its support vectors:
//      * 2 support vectors (one per class)  → w is PARALLEL to the cross-class
//        segment joining them (the boundary is its perpendicular bisector);
//      * 3 support vectors (two from one class) → w is PERPENDICULAR to the
//        same-class segment joining those two (both sit on one margin line).
//    So enumerating every pair direction (cross-class: parallel; same-class:
//    perpendicular) provably contains the optimal direction. For a fixed
//    direction the best canonical separator is the bisector of the extreme
//    projections: gap = min over class 1 of (ŵ·x) − max over class 0 of (ŵ·x);
//    the separator separates with band width = gap, and w = 2ŵ/gap gives
//    ‖w‖ = 2/gap, margin = 2/‖w‖ = gap, support vectors at distance 1/‖w‖.
//    The global optimum is the direction with the LARGEST gap — exact.
//  - Step model: CANDIDATE SWEEP (mirrors knn's k-sweep / ridge's λ-sweep).
//    One snapshot per displayed candidate (top-40 by gap, SWEEP_CAP), ordered
//    WEAKEST → STRONGEST margin, so the running-best separator and the loss
//    ½‖w‖² (lossMetricKey, lower-better) improve monotonically and the FINAL
//    snapshot is exactly the global max-margin solution. The sweep list is
//    computed once per params key and memoized (bounded cache), so
//    initialState/step are O(1) after the first.
//  - Data: two Gaussian clusters (class 0 left at −margin, class 1 right at
//    +margin), `margin` slider = cluster separation. Hard margin needs separable
//    data: generation runs a bounded deterministic seed search (seed, seed+1, …)
//    until separation holds (checked with the solver's own gap test: separable
//    ⇔ some direction has gap > 0). At the default settings the search exits
//    immediately at the requested seed; for tight/noisy in-range draws it may
//    advance to a later separable seed (see failures.ts). If none of
//    200 candidate seeds separates, initialState throws an honest error →
//    computeRun telemetry records it (converged: false). The effective seed is
//    surfaced in metrics + algorithm (dataSeed).
//  - Scaling invariance is built in: scaling all points by c rescales the optimal
//    weights by 1/c (w' = w/c, b' = b), so ‖w'‖ = ‖w‖/c, margin' = c·margin, and
//    the invariant margin·‖w‖ = 2 is preserved. The test-only `scale` param
//    (not in the schema — cf. logistic's nClass1/nonLinear precedent) multiplies
//    generated points. The plan's "scaling data by 2 doubles ‖w‖" is the
//    shrinking framing (scale ½); scaling UP by 2 halves ‖w‖ — both are asserted.
//  - No `points` override param (the plan says to skip it): hand-crafted datasets
//    are unsupported, so validateParams has no points branch.
import type { TopicModule, Params, SimState, VisualCommand, ParamValue, MathStep } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { registerClassifier } from '../../registry/viewRegistry';
import { svmTestCases } from './testCases';
import { svmFormulas } from './formulas';
import { svmDerivations } from './derivations';
import { svmMistakes } from './mistakes';
import { svmQuestions } from './questions';
import { svmComparisons } from './comparisons';
import { svmFailureDemos } from './failures';

export interface SvmPoint { x: number; y: number; cls: number; } // cls ∈ {0, 1}

// Data hard-clip: keeps the canvas sane even at extreme (test-only) scale values.
// At the default range (scale 1–2, noise ≤ 0.45) no point is ever clipped, so
// the clip never disturbs the scaling-invariance measurements.
const DOMAIN = 8;
// Bounded deterministic seed search for separability (see header design notes).
const MAX_SEED_SEARCH = 200;
// Max candidates shown per run (ascending-gap sweep always ENDS at the optimum).
const SWEEP_CAP = 40;

/** Mulberry32 — deterministic PRNG (matches every Wave-1/2 topic). */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller Gaussian from two uniform draws (deterministic given the PRNG). */
function gaussian(rng: () => number): number {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Deterministic data synthesis. Two clusters: class 0 centred at (−margin, 0),
 * class 1 at (+margin, 0), each point jittered by N(0, noise). The `scale`
 * (test-only) param multiplies every generated point — the scaling-invariance
 * dial. No `points` override is supported (hard margin needs guaranteed
 * separability; see findSeparableSeed).
 */
export function generatePoints(p: Params): SvmPoint[] {
  const nPerClass = (p.nPerClass as number) ?? 12;
  const margin = (p.margin as number) ?? 1.5;
  const noise = (p.noise as number) ?? 0.45;
  const scale = (p.scale as number) ?? 1;
  const rng = mulberry32((p.seed as number) ?? 42);
  const pts: SvmPoint[] = [];
  for (let c = 0; c < 2; c++) {
    const cx = c === 0 ? -margin : margin;
    for (let i = 0; i < nPerClass; i++) {
      const x = Math.max(-DOMAIN, Math.min(DOMAIN, scale * (cx + gaussian(rng) * noise)));
      const y = Math.max(-DOMAIN, Math.min(DOMAIN, scale * (gaussian(rng) * noise)));
      pts.push({ x, y, cls: c });
    }
  }
  return pts;
}

// ---- geometric max-margin solver ------------------------------------------

export interface SvmSolution {
  w1: number; w2: number; b: number;      // canonical separator: yᵢ(w·xᵢ+b) ≥ 1
  normW: number;                          // ‖w‖
  margin: number;                         // 2/‖w‖ — the band width
  gamma: number;                          // 1/‖w‖ — per-point geometric margin
  halfWSq: number;                        // ½‖w‖² — the SVM objective (loss)
  gap: number;                            // band width in unit-normal terms (= margin)
  sv: number[];                           // support-vector point indices
  separable: boolean;
}

export interface SvmCandidate extends SvmSolution {
  pair: [number, number];                 // representative point pair for the direction
}

/** Projection extremes of class 1 (min) and class 0 (max) along the unit normal. */
function orientedGap(points: SvmPoint[], ux0: number, uy0: number): {
  ux: number; uy: number; gap: number; p1: SvmPoint; p0: SvmPoint;
} {
  let m1 = Infinity, M0 = -Infinity;
  let p1: SvmPoint | null = null, p0: SvmPoint | null = null;
  for (const pt of points) {
    const proj = ux0 * pt.x + uy0 * pt.y;
    if (pt.cls === 1) {
      if (proj < m1) { m1 = proj; p1 = pt; }
    } else if (proj > M0) {
      M0 = proj; p0 = pt;
    }
  }
  // Precondition: both classes present (generatePoints guarantees it; the guard
  // keeps the type honest for reuse with single-class input).
  if (p1 === null || p0 === null) {
    return { ux: ux0, uy: uy0, gap: -Infinity, p1: points[0], p0: points[0] };
  }
  return { ux: ux0, uy: uy0, gap: m1 - M0, p1, p0 };
}

/**
 * Support vectors: points whose canonical functional margin yᵢ(w·xᵢ+b) is 1
 * (i.e. on a margin line, at distance 1/‖w‖ from the boundary). The two extreme
 * points that defined the direction always qualify; ties (extra collinear
 * extreme points) are caught by the tolerance.
 */
export function supportVectorIndices(points: SvmPoint[], w1: number, w2: number, b: number, tol = 1e-6): number[] {
  const out: number[] = [];
  for (let i = 0; i < points.length; i++) {
    const y = points[i].cls === 1 ? 1 : -1;
    const fm = y * (w1 * points[i].x + w2 * points[i].y + b);
    if (Math.abs(fm - 1) <= tol) out.push(i);
  }
  return out;
}

/**
 * All feasible candidate separators. For every point pair:
 *   - cross-class pair → direction ŵ ∥ the pair segment (2-SV optimum family);
 *   - same-class pair   → direction ŵ ⊥ the pair segment (3-SV optimum family).
 * Each direction's best canonical separator is the bisector of the extreme
 * projections; a direction is FEASIBLE when gap > 0 (it separates). The global
 * optimum is the feasible candidate with the largest gap, and its direction is
 * generated by the optimal support-vector pair itself — so the enumeration is
 * EXACT (see module header). Deterministic: pair order + stable sort.
 */
export function buildCandidates(points: SvmPoint[]): SvmCandidate[] {
  // O(n²) point pairs × O(n) per orientedGap = O(n³); with n ≤ 40 that is
  // ≈ 64K ops — trivially fast, computed once per params key (cached).
  const n = points.length;
  const seen = new Set<string>();
  const cands: SvmCandidate[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = points[j].x - points[i].x;
      const dy = points[j].y - points[i].y;
      // cross-class → parallel to the segment; same-class → perpendicular
      const bx = points[i].cls === points[j].cls ? -dy : dx;
      const by = points[i].cls === points[j].cls ? dx : dy;
      const len = Math.hypot(bx, by);
      if (len < 1e-12) continue;
      const ux0 = bx / len, uy0 = by / len;
      let r = orientedGap(points, ux0, uy0);
      if (r.gap < 0) r = orientedGap(points, -ux0, -uy0); // orientedGap(−u) = −orientedGap(u)
      if (r.gap <= 1e-9) continue; // not a separating direction
      // exact-dedup of the (signed) unit direction — identical directions give
      // identical solutions; near-identical ones (float noise) stay distinct so
      // the true optimum direction is always evaluated exactly.
      const key = `${r.ux.toFixed(10)},${r.uy.toFixed(10)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const w1 = (2 * r.ux) / r.gap;
      const w2 = (2 * r.uy) / r.gap;
      const midx = (r.p1.x + r.p0.x) / 2;
      const midy = (r.p1.y + r.p0.y) / 2;
      const b = -(w1 * midx + w2 * midy);
      const normW = Math.hypot(w1, w2);
      cands.push({
        w1, w2, b,
        normW,
        margin: 2 / normW,
        gamma: 1 / normW,
        halfWSq: 0.5 * normW * normW,
        gap: r.gap,
        sv: supportVectorIndices(points, w1, w2, b),
        separable: true,
        pair: [i, j],
      });
    }
  }
  cands.sort((a, b) => a.gap - b.gap); // ascending — sweep order (weak → strong)
  return cands;
}

const NOT_SEPARABLE: SvmSolution = {
  w1: 0, w2: 0, b: 0, normW: 0, margin: 0, gamma: 0, halfWSq: 0, gap: 0, sv: [], separable: false,
};

/** Exact hard-margin SVM for the given points (see module header). */
export function solveHardMargin(points: SvmPoint[]): SvmSolution {
  const cands = buildCandidates(points);
  if (cands.length === 0) return NOT_SEPARABLE;
  return cands[cands.length - 1]; // max gap (ascending sort) — the optimum
}

/** Largest gap over all pair directions — separability test (gap > 0 ⇔ separable). */
export function maxGap(points: SvmPoint[]): number {
  const cands = buildCandidates(points);
  return cands.length > 0 ? cands[cands.length - 1].gap : 0;
}

export interface SweepResult {
  data: SvmPoint[];
  dataSeed: number;
  solution: SvmSolution;      // the exact optimum (final snapshot model)
  candidates: SvmCandidate[]; // ascending by gap, capped at SWEEP_CAP — last = optimum
}

/**
 * Deterministic seed search: try seed, seed+1, … until the generated data is
 * linearly separable (gap > 0). Bounded — if nothing separates, the requested
 * seed's data is returned and initialState throws an honest telemetry error.
 */
export function findSeparableData(p: Params): { points: SvmPoint[]; seed: number } {
  const base = (p.seed as number) ?? 42;
  for (let s = base; s < base + MAX_SEED_SEARCH; s++) {
    const pts = generatePoints({ ...p, seed: s });
    if (maxGap(pts) > 1e-9) return { points: pts, seed: s };
  }
  return { points: generatePoints({ ...p, seed: base }), seed: base };
}

export function buildSweep(p: Params): SweepResult {
  const { points, seed } = findSeparableData(p);
  const all = buildCandidates(points);
  const candidates = all.length > SWEEP_CAP ? all.slice(all.length - SWEEP_CAP) : all;
  const solution = candidates.length > 0
    ? (candidates[candidates.length - 1] as SvmSolution)
    : NOT_SEPARABLE;
  return { data: points, dataSeed: seed, solution, candidates };
}

// ---- memoization -----------------------------------------------------------
// getSweep() is the SINGLE source of truth for initialState, step and the
// classifier fallback — one deterministic SweepResult per params key (bounded).
const SWEEP_CACHE = new Map<string, SweepResult>();
const SWEEP_CACHE_MAX = 16;

function sweepKey(p: Params): string {
  return JSON.stringify([p.nPerClass, p.margin, p.noise, p.seed, p.scale]);
}

export function getSweep(p: Params): SweepResult {
  const key = sweepKey(p);
  let sw = SWEEP_CACHE.get(key);
  if (!sw) {
    sw = buildSweep(p);
    if (SWEEP_CACHE.size >= SWEEP_CACHE_MAX) SWEEP_CACHE.clear();
    SWEEP_CACHE.set(key, sw);
  }
  return sw;
}

// ---- visuals ---------------------------------------------------------------

// Class colors mirror the decision-boundary palette (class 0 blue / class 1 red).
const CLS_COLORS = ['#3b82f6', '#ef4444'];
const BOUNDARY_COLOR = '#3b82f6';
const MARGIN_COLOR = '#f59e0b';
const NORMAL_COLOR = '#0f172a';

/** Clip the implicit line w1·x + w2·y + c = 0 to the data bbox (Liang–Barsky style). */
function clipImplicitLine(w1: number, w2: number, c: number, bbox: { x0: number; x1: number; y0: number; y1: number }):
  [number, number][] | null {
  const normSq = w1 * w1 + w2 * w2;
  if (normSq < 1e-12) return null;
  const p0x = -c * w1 / normSq, p0y = -c * w2 / normSq;
  const dx = -w2, dy = w1;
  let tMin = -Infinity, tMax = Infinity;
  for (const [lo, hi, pc, d] of [[bbox.x0, bbox.x1, p0x, dx], [bbox.y0, bbox.y1, p0y, dy]] as const) {
    if (Math.abs(d) < 1e-12) continue;
    const t1 = (lo - pc) / d, t2 = (hi - pc) / d;
    tMin = Math.max(tMin, Math.min(t1, t2));
    tMax = Math.min(tMax, Math.max(t1, t2));
  }
  if (tMin > tMax || !Number.isFinite(tMin) || !Number.isFinite(tMax)) return null;
  return [[p0x + tMin * dx, p0y + tMin * dy], [p0x + tMax * dx, p0y + tMax * dy]] as [number, number][];
}

function bboxOf(points: SvmPoint[]): { x0: number; x1: number; y0: number; y1: number } {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const pt of points) {
    x0 = Math.min(x0, pt.x); x1 = Math.max(x1, pt.x);
    y0 = Math.min(y0, pt.y); y1 = Math.max(y1, pt.y);
  }
  return { x0, x1, y0, y1 };
}

function buildVisuals(points: SvmPoint[], c: SvmCandidate): VisualCommand[] {
  const pts: VisualCommand[] = points.map((pt, i) => ({
    type: 'point', id: `d${i}`, x: pt.x, y: pt.y, color: CLS_COLORS[pt.cls],
  }));
  const bbox = bboxOf(points);
  const boundary = clipImplicitLine(c.w1, c.w2, c.b, bbox);
  const out: VisualCommand[] = [...pts];
  if (boundary) {
    out.push({ type: 'line', id: 'boundary', points: boundary, color: BOUNDARY_COLOR });
  }
  // margin band: the two canonical margin lines w·x + b = ±1 at distance 1/‖w‖
  const m1 = clipImplicitLine(c.w1, c.w2, c.b - 1, bbox);
  const m2 = clipImplicitLine(c.w1, c.w2, c.b + 1, bbox);
  if (m1) out.push({ type: 'line', id: 'marginPos', points: m1, color: MARGIN_COLOR });
  if (m2) out.push({ type: 'line', id: 'marginNeg', points: m2, color: MARGIN_COLOR });
  // normal arrow: from the boundary midpoint along +ŵ (the separating direction)
  const ux = c.w1 / c.normW, uy = c.w2 / c.normW;
  const mid = boundary ? boundary[0] : [0, 0];
  out.push({
    type: 'arrow', id: 'normal',
    x1: mid[0], y1: mid[1], x2: mid[0] + ux * 1.2, y2: mid[1] + uy * 1.2,
    color: NORMAL_COLOR,
  });
  return out;
}

/** Highlight the support vectors (ScatterPlot renders them larger + amber). */
function svHighlights(sv: number[]): SimState['highlights'] {
  return sv.map((i) => ({ panel: 'canvas', id: `d${i}`, intensity: 1 }));
}

// ---- simulation ------------------------------------------------------------

function snapshotAt(
  sweep: SweepResult, cand: SvmCandidate, rank: number,
  isFinal: boolean, first: boolean, prev?: SimState,
): SimState {
  const total = sweep.candidates.length;
  const [iA, iB] = cand.pair;
  const metrics: Record<string, number> = {
    halfWSq: cand.halfWSq,
    margin: cand.margin,
    gamma: cand.gamma,
    normW: cand.normW,
    svCount: cand.sv.length,
    nPoints: sweep.data.length,
    iteration: rank,
    totalCandidates: total,
    dataSeed: sweep.dataSeed,
    trainError: 0, // hard margin: perfectly separable by construction
  };
  const changed: string[] = [];
  if (!first && prev) {
    changed.push(`candidate → ${rank}/${total}`);
    if (prev.metrics.halfWSq !== cand.halfWSq) changed.push(`½‖w‖² → ${cand.halfWSq.toFixed(4)}`);
    if (prev.metrics.margin !== cand.margin) changed.push(`margin → ${cand.margin.toFixed(3)}`);
    if (prev.metrics.svCount !== cand.sv.length) changed.push(`support vectors → ${cand.sv.length}`);
  }
  const math: MathStep[] = [
    { latex: 'w \\cdot x + b = 0', id: 'svm-hyperplane' },
    { latex: '\\gamma_i = \\frac{y_i (w \\cdot x_i + b)}{\\|w\\|}', id: 'svm-geometric-margin' },
    { latex: '\\text{margin} = \\frac{2}{\\|w\\|} = \\text{gap}', id: 'svm-margin' },
    { latex: '\\min_{w,b}\\; \\tfrac{1}{2}\\|w\\|^2 \\;\\; \\text{s.t. } y_i(w \\cdot x_i + b) \\ge 1', id: 'svm-primal' },
  ];
  const events = first
    ? [{ type: 'init', label: `seeded separable data (seed ${sweep.dataSeed})`, step: rank }]
    : [{ type: 'candidate', label: `evaluated candidate ${rank}/${total}`, step: rank }];
  if (isFinal) events.push({ type: 'converged', label: 'max-margin separator found', step: rank });

  return {
    algorithm: {
      mode: 'svm-hard-margin',
      iteration: rank,
      w1: cand.w1, w2: cand.w2, b: cand.b,
      normW: cand.normW, svCount: cand.sv.length,
      dataSeed: sweep.dataSeed,
    } as Record<string, ParamValue>,
    visuals: buildVisuals(sweep.data, cand),
    math,
    narration: `Candidate ${rank}/${total}: direction from points d${iA}–d${iB} → gap ${cand.gap.toFixed(3)}; ` +
      `margin = ${cand.margin.toFixed(3)} (2/‖w‖), ½‖w‖² = ${cand.halfWSq.toFixed(4)}, support vectors = ${cand.sv.length}${isFinal ? ' — max-margin separator (no candidate improves on it)' : ''}`,
    explanation: {
      changed,
      why: first
        ? `Optimization sweep starts at the weakest displayed separator (the top-40 by gap, SWEEP_CAP): every point pair defines a candidate direction (cross-class pair → normal ∥ the segment, the 2-support-vector optimum family; same-class pair → normal ⊥ the segment, the 3-support-vector family), and for that direction the best canonical separator is the bisector of the extreme projections (band width = gap). Candidates are examined weakest → strongest margin, so ½‖w‖² falls monotonically.`
        : `Candidate ${rank}/${total}: direction from points d${iA}–d${iB} gives band width ${cand.gap.toFixed(3)} (${cand.sv.length} support vectors at distance 1/‖w‖ = ${cand.gamma.toFixed(3)} from the boundary). The running best ½‖w‖² = ${cand.halfWSq.toFixed(4)}${isFinal ? ' — this is the global max-margin separator: the largest gap over all pair directions, exact for 2D' : '.'}`,
      formulaRef: 'svm-margin',
      dependsOn: ['linear-algebra', 'convex-optimization', 'gradient-descent'],
      gateConcepts: ['SVM', 'support vectors', 'max-margin hyperplane', 'geometric margin'],
    },
    highlights: svHighlights(cand.sv),
    metrics,
    events,
    timeline: first
      ? ['Data', 'Candidate', 'Evaluate']
      : isFinal ? ['Candidate', 'Evaluate', 'Optimal'] : ['Candidate', 'Evaluate'],
  };
}

export const simulation = {
/**
   * Snapshot 1 = the weakest displayed candidate (top-40 by gap, SWEEP_CAP);
   * each step advances to the next (stronger) candidate in the sweep; the final
   * snapshot is the exact optimum.
   * Non-separable data (pathological params) throws here → computeRun telemetry
   * records an honest failure (converged: false).
   */
  initialState: (p: Params): SimState => {
    const sweep = getSweep(p);
    if (!sweep.solution.separable) {
      throw new Error(
        `svm-hard-margin: no linearly separable data found for seed ${(p.seed as number) ?? 42} after ${MAX_SEED_SEARCH} ` +
        'candidate seeds — increase the "separation" slider or reduce "noise" (hard margin needs separable clusters)',
      );
    }
    const cand = sweep.candidates[0];
    return snapshotAt(sweep, cand, 1, sweep.candidates.length === 1, true);
  },

  /** Advance the candidate sweep. Returns null when the sweep is complete. */
  step: (p: Params, s: SimState): SimState | null => {
    const sweep = getSweep(p);
    const current = (s.algorithm.iteration as number) ?? 1;
    const next = current + 1;
    if (next > sweep.candidates.length) return null; // sweep complete
    const cand = sweep.candidates[next - 1];
    return snapshotAt(sweep, cand, next, next === sweep.candidates.length, false, s);
  },
};

// ---- classifier for the decision-boundary view ----------------------------
// Contract (landed by the Wave-2 decision-boundary task):
//   registerClassifier(id, (x, y, params) => class index 0/1)
// The view merges the current snapshot's algorithm state into params before each
// grid call, so classifyByParams reads the CURRENT candidate's (w1, w2, b)
// straight off params — the boundary tracks the exact step being scrubbed with
// no re-solving. Before any run exists it falls back to the deterministically
// re-solved optimum (memoized via getSweep).
export function classifyByParams(x: number, y: number, p: Params): number {
  const w1 = p.w1 as number | undefined;
  const w2 = p.w2 as number | undefined;
  const b = p.b as number | undefined;
  if (typeof w1 === 'number' && typeof w2 === 'number' && typeof b === 'number' && Number.isFinite(w1 + w2 + b)) {
    return w1 * x + w2 * y + b > 0 ? 1 : 0;
  }
  const sol = getSweep(p).solution;
  if (!sol.separable) return 0;
  return sol.w1 * x + sol.w2 * y + sol.b > 0 ? 1 : 0;
}

export const svmModule: TopicModule = {
  id: 'svm-hard-margin',
  title: 'SVM: Hard Margin',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 4, mathematical: 5, coding: 2, visualization: 3, gateFrequency: 5 },
    estimatedHours: 7,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['linear-algebra', 'convex-optimization', 'logistic-regression', 'perceptron'],
    relatedTopics: ['svm', 'logistic-regression', 'perceptron', 'lda', 'knn'],
    revision: { quick: '20m', standard: '1h', deep: '2h', mastery: '4h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Data, Max-Margin Band & Support Vectors' },
      { slot: 'primary', component: 'decision-boundary', title: 'Decision Boundary: Max-Margin Hyperplane' },
      { slot: 'primary', component: 'loss-curve', title: 'SVM Objective ½‖w‖² over the Candidate Sweep' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivations: margin = 2/‖w‖, Primal→Dual, KKT' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Sweep Stages' },
      { slot: 'sidebar', component: 'explain-step', title: 'Step Explanation' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'nPerClass', label: 'Points per class', type: 'number', min: 2, max: 20, step: 1, default: 12 },
    { id: 'margin', label: 'Cluster separation', type: 'number', min: 0.5, max: 3.0, step: 0.1, default: 1.5 },
    { id: 'noise', label: 'Cluster spread σ', type: 'number', min: 0.1, max: 1.5, step: 0.05, default: 0.45 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
  ],
  simulation,
  formulas: svmFormulas,
  derivations: svmDerivations,
  questions: svmQuestions,
  comparisons: svmComparisons,
  failureDemos: svmFailureDemos,
  mistakes: svmMistakes,
  testCases: svmTestCases,
  lossMetricKey: 'halfWSq',

  validateParams: (p) => {
    const issues: string[] = [];
    const nPerClass = (p.nPerClass as number) ?? 12;
    const margin = (p.margin as number) ?? 1.5;
    const noise = (p.noise as number) ?? 0.45;
    if (!Number.isFinite(nPerClass) || nPerClass < 2) {
      issues.push('nPerClass must be ≥ 2 — hard-margin SVM needs at least 2 points per class for a meaningful support set');
    }
    if (nPerClass > 20) issues.push('nPerClass > 20 exceeds the lightweight demo size (keep n ≤ 20 for smooth scrubbing)');
    if (!Number.isFinite(margin) || margin < 0.5) {
      issues.push('margin (cluster separation) must be ≥ 0.5 — hard-margin SVM needs separable clusters; below 0.5 the draw is not guaranteed to separate');
    }
    if (!Number.isFinite(noise) || noise <= 0) issues.push('noise (cluster spread σ) must be positive');
    if (noise >= 0.8 * margin) {
      issues.push(`noise (${noise}) is large relative to the separation (${margin}) — the draw may be non-separable; the simulator searches a bounded set of seeds and fails honestly if none separates`);
    }
    const scale = p.scale as number | undefined;
    if (scale !== undefined && (!Number.isFinite(scale) || scale <= 0)) {
      issues.push('scale must be a positive number (test-only data multiplier)');
    }
    return issues;
  },
};

export function register() {
  registerTopic(svmModule);
  // DecisionBoundary resolves this via getClassifier('svm-hard-margin') to paint
  // class regions + the fitted boundary line (2500 calls per snapshot — reading
  // w1/w2/b off the merged params keeps it a score+sign per cell, no re-solving).
  registerClassifier(svmModule.id, classifyByParams);
}
