// src/topics/perceptron/module.ts
// Task 13 (Wave 3): perceptron — 2D binary classification by the classic
// fixed-increment Rosenblatt perceptron (online, linear threshold).
//
// Design decisions (deviations from the plan are documented in the report):
//  - Step model: ONE MISTAKE-DRIVEN UPDATE per step. Each step scans the dataset
//    in a fixed deterministic order from the current sweep pointer for the FIRST
//    misclassified point (yᵢ·(w·xᵢ+b) ≤ 0), applies THE update w ← w+η·yᵢ·xᵢ /
//    b ← b+η·yᵢ, and advances the pointer past it — so the mistake counter
//    advances by exactly 1 per snapshot. Correct points are examined and skipped
//    within the scan. An EPOCH is one full sweep (all n points examined); the
//    `mistakesPerEpoch` metric is the number of updates fired in the most
//    recently COMPLETED epoch. The run converges when an epoch completes with
//    zero updates: the converged snapshot (same weights, clean-sweep event,
//    timeline 'Converge') is emitted, and the NEXT step() returns null (the
//    two-phase termination pattern — the initial snapshot is identical to it,
//    which is what the tests scrub through).
//  - The whole trajectory (update sequence, epoch boundaries, cycle detection)
//    is PRECOMPUTED once per params key and memoized (getPlan — svm-hard-margin's
//    getSweep precedent): initialState/step are O(1) per call, runs are
//    deterministic, and the classifier fallback reuses the same plan's final
//    weights. MAX_UPDATES (5000) is the hard backstop: nothing ever hangs.
//  - NON-SEPARABLE data (`separable: false`) keeps class 0 centered at −margin
//    and moves only class 1 to the origin — heavily overlapping clouds, not
//    linearly separable (the margin and noise sliders keep their meaning for the
//    separable case). The classic
//    cycling phenomenon applies: the weight state (w1,w2,b,pos) — which fully
//    determines the next scan — repeats exactly (each update adds ±η·yᵢ·xᵢ, so a
//    repeated update sequence reproduces bit-identical floats). getPlan detects
//    the first exact state repeat and step() THROWS an honest Error when the
//    run reaches the repeat → computeRun records telemetry.failedAtStep +
//    failureReason ("perceptron does not converge on non-separable data:
//    weight-state cycle detected (start at update X, length L)").
//  - Classic convergence is η-INVARIANT: with fixed η the update SEQUENCE is
//    scale-invariant (y·(w·x+b) ≤ 0 ⟺ y·(ηw·x+ηb) ≤ 0 for η > 0), so the update
//    COUNT is identical for every η > 0 from zero init (asserted in
//    testCases.test.ts); the final weights scale exactly by η. The
//    convergence-theorem bound updates ≤ (R/γ)² is computed from MEASURED R
//    and γ (the geometric margin) and asserted.
//  - No loss function exists: the perceptron does NOT minimize one (updates fire
//    ONLY on mistakes). lossMetricKey = 'mistakesPerEpoch' is a diagnostic
//    (lower-better) and the loss-curve layer is titled accordingly.
import type { TopicModule, Params, SimState, VisualCommand, ParamValue, MathStep } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { registerClassifier } from '../../registry/viewRegistry';
import { perceptronTestCases } from './testCases';
import { perceptronFormulas } from './formulas';
import { perceptronDerivations } from './derivations';
import { perceptronMistakes } from './mistakes';
import { perceptronQuestions } from './questions';
import { perceptronComparisons } from './comparisons';
import { perceptronFailureDemos } from './failures';

interface PPoint { x: number; y: number; label: number; } // label ∈ {+1, -1}

// Hard backstop: the classic rule converges in O((R/γ)²) updates on separable
// data (measured: 4 on the default seed, 23 on seed 7) and never settles on
// non-separable data. Nothing hangs past this bound.
export const MAX_UPDATES = 5000;

// Non-separable runs are NOT exact-cyclic in any practical sense: each update
// fires on a different point, so the (w, b, scan position) float state drifts
// like a bounded random walk and an EXACT repeat is astronomically unlikely
// (measured: none within MAX_UPDATES on the default overlapping clouds — the
// classical cycling theorem proves oscillation, but the exact float period can
// far exceed any runnable bound). The honest demo bound is therefore a SNAPSHOT
// CAP: a non-converging run terminates after OSCILLATION_CAP updates (181
// snapshots: init + 180 updates, no converge re-emission on non-separable
// runs — under the plan's ~200 cap) with an honest
// oscillation telemetry message. An exact cycle, when it fires (keys on some
// seeds), still terminates early with the precise period.
export const OSCILLATION_CAP = 180;

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

interface PerceptronData { points: PPoint[]; R: number; }

/**
 * Deterministic data. `separable: true` (default): class 0 (y = −1) clustered at
 * (−margin, 0), class 1 (y = +1) at (+margin, 0), each point jittered by
  * N(0, noise) — near-surely linearly separable at the supported slider range
  * (margin ≥ 0.9, noise ≤ 0.6 on the default seed). `separable: false`: class 0
  * stays centered at −margin, only class 1 moves to the origin — heavily
  * overlapping clouds, not linearly separable with probability ~1 for n ≥ 4
  * points in general position.
 * R = maxᵢ‖xᵢ‖ is the data-radius input to the convergence-theorem bound.
 * Fixed point ORDER: class 0 first (indices 0..nPerClass−1), class 1 after
 * (nPerClass..‥) — the scan order and therefore the update sequence are
 * deterministic.
 */
export function generateData(p: Params): PerceptronData {
  const nPerClass = (p.nPerClass as number) ?? 20;
  const margin = (p.margin as number) ?? 1.2;
  const noise = (p.noise as number) ?? 0.5;
  const nClass1 = (p.nClass1 as number) ?? nPerClass; // test-only imbalance override
  const separable = p.separable !== false;
  const rng = mulberry32((p.seed as number) ?? 42);
  const points: PPoint[] = [];
  for (let i = 0; i < nPerClass; i++) {
    points.push({ x: -margin + gaussian(rng) * noise, y: gaussian(rng) * noise, label: -1 });
  }
  const c1x = separable ? margin : 0; // non-separable: class 1 at the origin (class 0 stays at −margin)
  for (let i = 0; i < nClass1; i++) {
    points.push({ x: c1x + gaussian(rng) * noise, y: gaussian(rng) * noise, label: +1 });
  }
  const R = Math.max(...points.map((pt) => Math.hypot(pt.x, pt.y)), 1e-9);
  return { points, R };
}

// ---- data + plan memoization (knn getPoints / svm getSweep precedent) -------
const DATA_CACHE = new Map<string, PerceptronData>();
const PLAN_CACHE = new Map<string, RunPlan>();
const CACHE_MAX = 64;

function paramsKey(p: Params): string {
  return JSON.stringify([p.nPerClass, p.margin, p.noise, p.eta, p.init, p.seed, p.separable, p.nClass1, p.initScale]);
}

export function getData(p: Params): PerceptronData {
  const key = paramsKey(p);
  let d = DATA_CACHE.get(key);
  if (!d) {
    d = generateData(p);
    if (DATA_CACHE.size >= CACHE_MAX) DATA_CACHE.clear();
    DATA_CACHE.set(key, d);
  }
  return d;
}

// ---- the classic update rule (exported for the hand-verified unit tests) ----

/** The single mistake-driven update: w ← w + η·y·x, b ← b + η·y. */
export function perceptronUpdate(w1: number, w2: number, b: number, x: number, y: number, label: number, eta: number) {
  return {
    w1: w1 + eta * label * x,
    w2: w2 + eta * label * y,
    b: b + eta * label,
  };
}

/** Zero weights (classic start) or small seeded random (initScale, seed+1). */
export function initWeights(p: Params): { w1: number; w2: number; b: number } {
  const init = (p.init as string) ?? 'zero';
  if (init === 'zero') return { w1: 0, w2: 0, b: 0 };
  const scale = (p.initScale as number) ?? 0.1;
  const rng = mulberry32(((p.seed as number) ?? 42) + 1);
  return {
    w1: (rng() - 0.5) * 2 * scale,
    w2: (rng() - 0.5) * 2 * scale,
    b: (rng() - 0.5) * 2 * scale,
  };
}

// ---- run plan: precomputed deterministic trajectory -------------------------

/** State after update k (k ≥ 1): the weights plus which point the update fired on. */
interface TraceStep { w1: number; w2: number; b: number; hitIdx: number; }

interface EpochMark { endUpdate: number; mistakes: number; } // completed epoch

interface RunPlan {
  paramsKey: string;
  data: PerceptronData;
  init: { w1: number; w2: number; b: number };
  trace: TraceStep[];              // trace[k-1] = state after update k
  updates: number;                 // total updates until termination
  converged: boolean;              // clean epoch reached (separable) vs cycle/backstop
  cycleStart: number;              // update index where the weight state first repeats (0 if converged)
  cycleLength: number;             // repeat period (0 when converged)
  epochMarks: EpochMark[];         // completed epochs: end-update + mistakes fired
  finalW: { w1: number; w2: number; b: number };
  finalGamma: number;              // geometric margin at termination (0 if ‖w‖ = 0)
  finalAccuracy: number;
  bound: number;                   // (R/γ)² Novikoff theorem bound at the final solution
}

function weightOutcome(data: PerceptronData, w1: number, w2: number, b: number) {
  let ok = 0;
  let gamma = Infinity;
  const norm = Math.hypot(w1, w2);
  for (const pt of data.points) {
    const score = w1 * pt.x + w2 * pt.y + b;
    if (score * pt.label > 0) ok++;
    const d = norm > 1e-12 ? (score * pt.label) / norm : 0;
    gamma = Math.min(gamma, d);
  }
  return { accuracy: ok / data.points.length, gamma: norm > 1e-12 ? gamma : 0 };
}

/**
 * Simulate the FULL deterministic trajectory (bounded by MAX_UPDATES) once per
 * params key. Pure function of params — memoized, never mutated by runs.
 *
 * Scan semantics (the fixed-increment cyclic perceptron): each pass scans the
 * dataset in the FIXED deterministic order starting from the position of the
 * last update, cyclically, for the FIRST misclassified point (yᵢ·(w·xᵢ+b) ≤ 0).
 * The scan examines AT MOST n points — a full rotation of the dataset — so a
 * scan that completes the rotation without firing has examined EVERY point
 * under the CURRENT (unchanged) weights: that is exactly the classic
 * convergence criterion (zero mistakes on a full pass). (An earlier draft
 * capped the scan at the epoch-packet boundary instead of the rotation, which
 * let a PARTIAL sweep masquerade as a clean one: it "converged" with one point
 * still misclassified and accuracy 0.975 — the honest clean-rotation check
 * fixed it.)
 *
 * Epoch-packet bookkeeping (the mistakesPerEpoch metric): packets of n
 * examinations; when a scan crosses a packet boundary the crossing packet is
 * closed with the updates that fired inside it (the update whose exam triggered
 * the boundary is attributed to the closing packet — or to the next one if its
 * exam came after the boundary). On convergence the final clean rotation is the
 * last completed packet with mistakesPerEpoch = 0.
 */
export function buildPlan(p: Params): RunPlan {
  const data = getData(p);
  const eta = (p.eta as number) ?? 1;
  const init = initWeights(p);
  let { w1, w2, b } = init;
  const n = data.points.length;
  let pos = 0;                 // next scan index within the fixed cyclic order
  let epochExamined = 0;       // examinations since the current epoch-packet began
  let epochMistakes = 0;       // updates fired in the current epoch-packet
  let totalUpdates = 0;
  const trace: TraceStep[] = [];
  const epochMarks: EpochMark[] = [];
  const seen = new Map<string, number>(); // exact float state key → first update index
  seen.set(stateKey(w1, w2, b, pos), 0);
  let stoppedByCap = false;

  while (totalUpdates < MAX_UPDATES) {
    // Non-separable runs never settle — bound the emitted snapshots (~plan cap).
    if (totalUpdates >= OSCILLATION_CAP) { stoppedByCap = true; break; }

    // Full-rotation scan from pos for the FIRST misclassified point.
    const e0 = epochExamined;
    let hit = -1;
    let hitExam = 0;           // 1-based exam position of the hit within this scan
    let t = n;                 // exams taken in this scan (n when the rotation is clean)
    for (let k = 0; k < n; k++) {
      const i = (pos + k) % n;
      const pt = data.points[i];
      const score = w1 * pt.x + w2 * pt.y + b;
      epochExamined++;
      if (score * pt.label <= 0) { hit = i; hitExam = k + 1; t = hitExam; break; }
    }

    if (hit === -1) {
      // Full clean rotation: every point correct under the current weights.
      // Close the final epoch-packet with 0 mistakes (the clean sweep).
      epochMarks.push({ endUpdate: totalUpdates, mistakes: 0 });
      break;
    }

    // ONE mistake-driven update on the first misclassified point.
    const pt = data.points[hit];
    const nxt = perceptronUpdate(w1, w2, b, pt.x, pt.y, pt.label, eta);
    w1 = nxt.w1; w2 = nxt.w2; b = nxt.b;
    totalUpdates++;
    epochMistakes++;
    pos = (hit + 1) % n;
    trace.push({ w1, w2, b, hitIdx: hit });

    // Epoch-packet bookkeeping: packets of n examinations. If the scan crossed
    // a packet boundary, close the crossing packet, attributing the update to
    // the packet that contained its triggering exam.
    if (e0 + t >= n) {
      const hitInNewPacket = e0 + hitExam > n;
      epochMarks.push({ endUpdate: totalUpdates, mistakes: hitInNewPacket ? epochMistakes - 1 : epochMistakes });
      epochExamined = (e0 + t) - n;
      epochMistakes = hitInNewPacket ? 1 : 0;
    } else {
      epochExamined = e0 + t;
    }

    // Exact weight-state repeat → classic cycling → never converges.
    const key = stateKey(w1, w2, b, pos);
    const first = seen.get(key);
    if (first !== undefined) {
      const out = weightOutcome(data, w1, w2, b);
      return {
        paramsKey: paramsKey(p), data, init, trace, updates: totalUpdates, converged: false,
        cycleStart: first, cycleLength: totalUpdates - first,
        epochMarks, finalW: { w1, w2, b },
        finalGamma: out.gamma, finalAccuracy: out.accuracy, bound: NaN,
      };
    }
    seen.set(key, totalUpdates);
  }
  // Clean rotation (clean-sweep epoch mark) — or the non-separable cap fired /
  // backstop hit (no exact cycle within the runnable bound → updates == cap).
  const out = weightOutcome(data, w1, w2, b);
  const bound = out.gamma > 1e-12 ? (data.R / out.gamma) ** 2 : NaN;
  return {
    paramsKey: paramsKey(p), data, init, trace, updates: totalUpdates,
    converged: !stoppedByCap && totalUpdates < MAX_UPDATES,
    cycleStart: 0, cycleLength: 0, epochMarks,
    finalW: { w1, w2, b }, finalGamma: out.gamma, finalAccuracy: out.accuracy, bound,
  };
}

function stateKey(w1: number, w2: number, b: number, pos: number): string {
  return `${w1}|${w2}|${b}|${pos}`;
}

export function getPlan(p: Params): RunPlan {
  const key = paramsKey(p);
  let plan = PLAN_CACHE.get(key);
  if (!plan) {
    plan = buildPlan(p);
    if (PLAN_CACHE.size >= CACHE_MAX) PLAN_CACHE.clear();
    PLAN_CACHE.set(key, plan);
  }
  return plan;
}

// ---- visuals ----------------------------------------------------------------

// Class colors mirror the decision-boundary palette (class 0 blue / class 1 red).
const CLS_COLORS = ['#3b82f6', '#ef4444'];
const BOUNDARY_COLOR = '#f59e0b';
const NORMAL_COLOR = '#0f172a';

function clipImplicitLine(w1: number, w2: number, c: number, pts: PPoint[]): [number, number][] | null {
  const normSq = w1 * w1 + w2 * w2;
  if (normSq < 1e-12) return null;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const pt of pts) {
    x0 = Math.min(x0, pt.x); x1 = Math.max(x1, pt.x);
    y0 = Math.min(y0, pt.y); y1 = Math.max(y1, pt.y);
  }
  const p0x = -c * w1 / normSq, p0y = -c * w2 / normSq;
  const dx = -w2, dy = w1;
  let tMin = -Infinity, tMax = Infinity;
  for (const [lo, hi, pc, d] of [[x0, x1, p0x, dx], [y0, y1, p0y, dy]] as const) {
    if (Math.abs(d) < 1e-12) continue;
    const t1 = (lo - pc) / d, t2 = (hi - pc) / d;
    tMin = Math.max(tMin, Math.min(t1, t2));
    tMax = Math.min(tMax, Math.max(t1, t2));
  }
  if (tMin > tMax || !Number.isFinite(tMin) || !Number.isFinite(tMax)) return null;
  return [[p0x + tMin * dx, p0y + tMin * dy], [p0x + tMax * dx, p0y + tMax * dy]] as [number, number][];
}

function buildVisuals(data: PerceptronData, w1: number, w2: number, b: number, hitIdx: number | null, eta: number): VisualCommand[] {
  const pts: VisualCommand[] = data.points.map((pt, i) => ({
    type: 'point', id: `d${i}`, x: pt.x, y: pt.y, color: CLS_COLORS[pt.label === 1 ? 1 : 0],
  }));
  const out: VisualCommand[] = [...pts];
  const boundary = clipImplicitLine(w1, w2, b, data.points);
  if (boundary) out.push({ type: 'line', id: 'boundary', points: boundary, color: BOUNDARY_COLOR });
  // weight vector: normal arrow from the boundary midpoint along ŵ (length 1.2,
  // like svm-hard-margin) — the exact numbers live in the matrix-animator.
  const norm = Math.hypot(w1, w2);
  if (norm > 1e-12) {
    const mid = boundary ? boundary[0] : [0, 0];
    out.push({
      type: 'arrow', id: 'normal',
      x1: mid[0], y1: mid[1], x2: mid[0] + (w1 / norm) * 1.2, y2: mid[1] + (w2 / norm) * 1.2,
      color: NORMAL_COLOR,
    });
  }
  // matrices for matrix-animator: w (2×1), b (1×1) and the applied update term.
  out.push({ type: 'matrix', id: 'w', rows: 2, cols: 1, cells: [[w1], [w2]] });
  out.push({ type: 'matrix', id: 'b', rows: 1, cols: 1, cells: [[b]] });
  const pt = hitIdx !== null ? data.points[hitIdx] : data.points[0];
  out.push({
    type: 'matrix', id: 'Δw = η·y·x (last update)', rows: 2, cols: 1,
    cells: [[eta * pt.label * pt.x], [eta * pt.label * pt.y]],
  });
  return out;
}

function metricsOf(plan: RunPlan, w1: number, w2: number, b: number, epoch: number, updates: number, mistakesPerEpoch: number): Record<string, number> {
  const out = weightOutcome(plan.data, w1, w2, b);
  return {
    mistakes: updates,
    updates,
    epoch,
    mistakesPerEpoch,
    accuracy: out.accuracy,
    normW: Math.hypot(w1, w2),
    gamma: out.gamma,
    R: plan.data.R,
    nPoints: plan.data.points.length,
  };
}

function snapshotAt(
  p: Params, plan: RunPlan, w1: number, w2: number, b: number, hitIdx: number | null,
  epoch: number, updates: number, mistakesPerEpoch: number, first: boolean, terminal: 'init' | 'update' | 'converge',
): SimState {
  const m = metricsOf(plan, w1, w2, b, epoch, updates, mistakesPerEpoch);
  const eta = (p.eta as number) ?? 1;
  const initLabel = (p.init as string) ?? 'zero';
  const events: SimState['events'] = [];
  if (first) events.push({ type: 'init', label: `initialized (${initLabel} weights)`, step: 0 });
  if (terminal === 'converge') events.push({ type: 'converged', label: `converged in ${updates} updates`, step: updates });
  // Non-separable: the honest oscillation verdict lands on the FINAL emitted
  // snapshot (exact cycle repeat, or the oscillation cap) — the next step()
  // throws with the details in telemetry. One event, not one per update.
  if (!plan.converged && updates >= plan.updates) {
    events.push({
      type: 'oscillation',
      label: plan.cycleLength > 0
        ? `oscillation: weight state repeats every ${plan.cycleLength} updates — never converges`
        : `oscillation: still making mistakes after ${plan.updates} updates — run capped (the data may not be linearly separable, or convergence needs more updates)`,
      step: updates,
    });
  }

  const math: MathStep[] = [
    { latex: '\\hat{y} = \\operatorname{sign}(w \\cdot x + b)', id: 'perceptron-rule' },
    { latex: 'w \\leftarrow w + \\eta\\, y_i x_i, \\qquad b \\leftarrow b + \\eta\\, y_i', id: 'perceptron-update' },
    { latex: 'y_i (w \\cdot x_i + b) \\le 0 \\;\\Rightarrow\\; \\text{mistake — fire the update}', id: 'mistake-condition' },
  ];

  const wStr = `w = (${w1.toFixed(3)}, ${w2.toFixed(3)}), b = ${b.toFixed(3)}`;
  let narration: string;
  const why: string[] = [];
  const changed: string[] = [];
  if (terminal === 'init') {
    narration = `Epoch 0: ${wStr} — every score is 0, so all ${plan.data.points.length} points count as misclassified. ` +
      `accuracy ${(m.accuracy * 100).toFixed(1)}%, ‖w‖ = ${m.normW.toFixed(3)}`;
    why.push(`Initialization (${initLabel}): all yᵢ·(w·xᵢ+b) = 0 ≤ 0 — the classic zero-weight start fires on point d0 first.`);
    changed.push('weights initialized', 'scan order fixed from index 0');
  } else if (terminal === 'converge') {
    narration = `Epoch ${epoch} completed with 0 updates — CONVERGED in ${updates} updates. ${wStr} — ` +
      `accuracy ${(m.accuracy * 100).toFixed(1)}%, ‖w‖ = ${m.normW.toFixed(3)}, geometric margin γ = ${m.gamma.toFixed(3)}`;
    why.push(`A full clean sweep: every point was correctly classified at its visit, and since no update fired, no weight ` +
      `changed — so every point is correct everywhere. The convergence-theorem bound (R/γ)² = ` +
      `${Number.isFinite(plan.bound) ? plan.bound.toFixed(0) : '∞'} (measured R = ${plan.data.R.toFixed(3)}, ` +
      `γ = ${m.gamma.toFixed(3)}) is loose as theory predicts: ${updates} updates ≪ bound.`);
    changed.push('converged — clean sweep, 0 mistakes this epoch', `‖w‖ = ${m.normW.toFixed(3)}`, `γ = ${m.gamma.toFixed(3)}`);
  } else {
    const hitPt = hitIdx !== null ? plan.data.points[hitIdx] : null;
    const marginText = hitPt !== null
      ? (hitPt.label * (w1 * hitPt.x + w2 * hitPt.y + b)).toFixed(3)
      : '—';
    narration = `Update ${updates} (epoch ${epoch}): point d${hitIdx ?? '?'} misclassified (y·(w·x+b) = ${marginText} ≤ 0) → ` +
      `w += η·y·x → ${wStr}; ‖w‖ = ${m.normW.toFixed(3)}, accuracy = ${(m.accuracy * 100).toFixed(1)}%, γ = ${m.gamma.toFixed(3)}`;
    why.push(`The mistake condition fired on the first misclassified point d${hitIdx ?? '?'} in the fixed scan order; ` +
      `the update adds η·yᵢ·xᵢ (η = ${eta}) and pulls w toward classifying that point correctly.`);
    changed.push(`w → ${wStr}`, `‖w‖ = ${m.normW.toFixed(3)}`, `accuracy → ${(m.accuracy * 100).toFixed(1)}%`);
    if (hitIdx !== null) changed.push(`point d${hitIdx} updated (first misclassified)`);
  }

  return {
    algorithm: {
      mode: 'perceptron', epoch, pos: hitIdx !== null ? (hitIdx + 1) % plan.data.points.length : 0,
      updates, w1, w2, b, eta, init: initLabel, separable: p.separable !== false,
      converged: terminal === 'converge',
      cycleEnd: plan.converged ? 0 : plan.cycleStart + plan.cycleLength,
    } as Record<string, ParamValue>,
    visuals: buildVisuals(plan.data, w1, w2, b, hitIdx, eta),
    math,
    narration,
    explanation: {
      changed,
      why: why.join(' '),
      formulaRef: terminal === 'converge' ? 'perceptron-convergence' : 'perceptron-update',
      dependsOn: ['linear-algebra', 'linear-classifier'],
      gateConcepts: ['perceptron', 'linear classifier', 'convergence theorem', 'margin'],
    },
    highlights: hitIdx !== null
      ? [{ panel: 'canvas', id: `d${hitIdx}`, intensity: 1 }]
      : [],
    metrics: m,
    events,
    timeline: first
      ? ['Initialize', 'Scan']
      : terminal === 'converge'
        ? ['Scan', 'Converge']
        : plan.converged
          ? (updates === 1 ? ['Mistake', 'Update'] : ['Mistake', 'Update', 'Repeat'])
          // Non-separable: the oscillation verdict lands on the final emitted
          // snapshot (the cap/cycle threshold) — the next step throws.
          : (updates >= plan.updates
            ? ['Mistake', 'Update', 'Repeat', 'Oscillate']
            : ['Mistake', 'Update', 'Repeat']),
  };
}

// ---- simulation ------------------------------------------------------------

function epochStateAt(plan: RunPlan, updates: number): { epoch: number; mistakesPerEpoch: number } {
  let epoch = 0;
  let mistakesPerEpoch = 0;
  for (const em of plan.epochMarks) {
    if (em.endUpdate <= updates) { epoch++; mistakesPerEpoch = em.mistakes; }
    else break;
  }
  return { epoch, mistakesPerEpoch };
}

export const simulation = {
  /** Snapshot 0: the init weights before any update. */
  initialState: (p: Params): SimState => {
    const plan = getPlan(p);
    return snapshotAt(p, plan, plan.init.w1, plan.init.w2, plan.init.b, null, 0, 0, 0, true, 'init');
  },

  /**
   * ONE mistake-driven update per step (see header). Two-phase termination:
   * when a clean epoch is reached, the converged snapshot carries the
   * clean-sweep event and the NEXT call returns null. Non-separable runs
   * THROW an honest Error when the weight cycle repeats → computeRun records
   * telemetry.failedAtStep + failureReason.
   */
  step: (p: Params, s: SimState): SimState | null => {
    if (s.algorithm.converged === true) return null;
    const plan = getPlan(p);
    const cur = (s.algorithm.updates as number) ?? 0;
    if (cur < plan.updates) {
      // Apply the next update from the precomputed trace.
      const t = plan.trace[cur];
      const es = epochStateAt(plan, cur + 1);
      return snapshotAt(p, plan, t.w1, t.w2, t.b, t.hitIdx, es.epoch, cur + 1, es.mistakesPerEpoch, false, 'update');
    }
    if (plan.converged) {
      // All updates applied; emit the converged snapshot (clean final epoch).
      const es = epochStateAt(plan, plan.updates);
      return snapshotAt(p, plan, plan.finalW.w1, plan.finalW.w2, plan.finalW.b, null,
        es.epoch, plan.updates, 0, false, 'converge');
    }
    // Non-separable: the run has reached the oscillation verdict (exact cycle
    // repeat, or the OSCILLATION_CAP) → honest failure via telemetry. An exact
    // cycle is the definitive oscillation signal; the CAP alone only says the
    // run did not settle within the budget — the draw may still be separable
    // and simply need more updates than the cap.
    const cycle = plan.cycleLength > 0
      ? `the weight state (w1, w2, b, scan position) at update ${plan.cycleStart} repeats exactly at update ${plan.updates} ` +
        `(cycle length ${plan.cycleLength}) — the boundary cycles forever`
      : `after ${plan.updates} updates the weights are still changing (mistakes fire on every epoch; no exact cycle within the ` +
        `runnable bound — the draw may be genuinely non-separable or merely slow to converge)`;
    const verdict = plan.cycleLength > 0
      ? 'the data is not linearly separable — the exact weight-state cycle is the classical proof\'s definitive oscillation signal'
      : `the run stopped after ${plan.updates} updates — the data is likely not linearly separable, or convergence requires more updates than the cap`;
    throw new Error(`perceptron does not converge: oscillation detected — ${cycle}. ${verdict}.`);
  },
};

// ---- classifier for the decision-boundary view ------------------------------
// Reads the CURRENT snapshot's (w1, w2, b) off merged params (the view spreads
// snapshot.algorithm over params before each grid call). Fallback (no snapshot
// yet): the plan's final weights, memoized via getPlan — deterministically
// trained, converged or not.
export function classifyByParams(x: number, y: number, p: Params): number {
  const w1 = p.w1 as number | undefined;
  const w2 = p.w2 as number | undefined;
  const b = p.b as number | undefined;
  if (typeof w1 === 'number' && typeof w2 === 'number' && typeof b === 'number' && Number.isFinite(w1 + w2 + b)) {
    return w1 * x + w2 * y + b > 0 ? 1 : 0;
  }
  const plan = getPlan(p);
  return plan.finalW.w1 * x + plan.finalW.w2 * y + plan.finalW.b > 0 ? 1 : 0;
}

export const perceptronModule: TopicModule = {
  id: 'perceptron',
  title: 'Perceptron',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 4, coding: 3, visualization: 4, gateFrequency: 5 },
    estimatedHours: 5,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['linear-algebra', 'linear-classifier', 'gradient-descent'],
    relatedTopics: ['logistic-regression', 'svm-hard-margin', 'svm-soft-margin', 'knn', 'neural-networks'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Data, Boundary, Weight Arrow & the Most Recent Mistake' },
      { slot: 'primary', component: 'decision-boundary', title: 'Decision Boundary: Linear Threshold Classifier' },
      { slot: 'primary', component: 'loss-curve', title: 'Mistakes per Epoch — perceptron has NO loss function' },
      { slot: 'sidebar', component: 'matrix-animator', title: 'Weights w, Bias b and the Last Update η·y·x' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivation: the Convergence Theorem' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Initialize → Mistake → Update → Repeat → Converge' },
      { slot: 'sidebar', component: 'explain-step', title: 'Step Explanation' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'nPerClass', label: 'Points per class', type: 'number', min: 2, max: 40, step: 1, default: 20 },
    { id: 'margin', label: 'Cluster separation', type: 'number', min: 0.4, max: 3.0, step: 0.1, default: 1.2 },
    { id: 'noise', label: 'Cluster spread σ', type: 'number', min: 0.1, max: 1.5, step: 0.05, default: 0.5 },
    { id: 'eta', label: 'Learning rate η', type: 'number', min: 0.01, max: 3, step: 0.01, default: 1 },
    {
      id: 'separable', label: 'Separable data', type: 'toggle',
      default: true,
    },
    {
      id: 'init', label: 'Initialization', type: 'select',
      options: [
        { value: 'zero', label: 'Zero (classic — first update fires on d0)' },
        { value: 'random', label: 'Random (small, seeded)' },
      ],
      default: 'zero',
    },
    { id: 'initScale', label: 'Random init scale', type: 'number', min: 0.01, max: 1, step: 0.01, default: 0.1 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
  ],
  simulation,
  formulas: perceptronFormulas,
  derivations: perceptronDerivations,
  questions: perceptronQuestions,
  comparisons: perceptronComparisons,
  failureDemos: perceptronFailureDemos,
  mistakes: perceptronMistakes,
  testCases: perceptronTestCases,
  lossMetricKey: 'mistakesPerEpoch',

  validateParams: (p) => {
    const issues: string[] = [];
    const nPerClass = (p.nPerClass as number) ?? 20;
    const margin = (p.margin as number) ?? 1.2;
    const noise = (p.noise as number) ?? 0.5;
    const eta = (p.eta as number) ?? 1;
    if (!Number.isFinite(nPerClass) || nPerClass < 2) {
      issues.push('nPerClass must be ≥ 2 — each class needs at least 2 points for a meaningful class region');
    }
    if (nPerClass > 200) issues.push('nPerClass > 200 exceeds the lightweight demo size (keep n ≤ ~200 for smooth scrubbing)');
    if (!Number.isFinite(margin) || margin < 0) issues.push('margin must be non-negative (the signed cluster-center offset)');
    if (!Number.isFinite(noise) || noise <= 0) issues.push('noise (cluster spread σ) must be positive');
    if (!Number.isFinite(eta) || eta <= 0) issues.push('Learning rate η must be positive');
    if (eta >= 1e3) issues.push('η ≥ 1000 risks numerical overflow of the weights (each update adds η·y·x)');
    if (p.separable === false && nPerClass < 4) {
      issues.push('With separable off (overlapping clusters), use nPerClass ≥ 4 so the clouds are genuinely inseparable');
    }
    // test-only imbalance override (failure demo); keep it in a sane range
    if (typeof p.nClass1 === 'number' && (p.nClass1 < 2 || p.nClass1 > 200)) {
      issues.push('nClass1 (test-only class-1 override) must be in [2, 200] — extremes would make the demo degenerate');
    }
    if (p.separable !== false && Number.isFinite(margin) && Number.isFinite(noise) && noise > 0 && margin / noise < 2.4) {
      issues.push('Noise is large relative to separation (margin/noise < 2.4): non-separability risk — the drawn clusters may not be linearly separable, ' +
        'and the perceptron then oscillates forever instead of converging (measured: seed 42 at margin 0.9 / noise 0.6)');
    }
    return issues;
  },
};

export function register() {
  registerTopic(perceptronModule);
  // DecisionBoundary resolves this via getClassifier('perceptron') to paint
  // class regions + the fitted boundary line (2500 calls per snapshot — reading
  // w1/w2/b off the merged params keeps it one score + sign per cell).
  registerClassifier(perceptronModule.id, classifyByParams);
}