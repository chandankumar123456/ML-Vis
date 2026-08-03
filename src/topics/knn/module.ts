// src/topics/knn/module.ts
// Task 9 (Wave 2): k-nearest-neighbors — 2D binary classification, lazy learner.
//
// Design decisions (deviations from the plan documented in the report):
//  - Step model: K-SWEEP mirroring ridge's λ-sweep EXACTLY — one snapshot per k on
//    [1, 2, …, params.k], last snapshot exactly the slider's k. Scrubbing the run IS
//    the boundary-smoothing animation (region count falls over the mid range as k rises)
//    + the growing distance rings around the query point.
//  - 'error' metric = LEAVE-ONE-OUT error per k (honest: k=1 is NOT zero under LOO,
//    unlike train error — the "k=1 overfits" story). 'trainError' = self-classification
//    error (k=1 → exactly 0: each point is its own nearest neighbor).
//  - Tie-break: among classes tied on vote count, the class whose nearest neighbor
//    (smallest distance inside the k-neighborhood) wins; on equal distance, lower class
//    index. Deterministic, documented, and tested with crafted exact ties.
//  - `points` param: OPTIONAL serialized custom dataset (JSON string) REPLACING seeded
//    generation when present — the module-side hook for click-to-add/drag-to-move
//    (ScatterPlot onDragPoint prop lands in this wave; TopicPage wiring is polish-wave).
//    Validated: even count, equal class counts, within [−5,5]², ≥ 2 points.
//  - Distance rings rendered as 'circle' visual commands (additive ScatterPlot case).
import type { TopicModule, Params, SimState, VisualCommand, MathStep, ParamValue } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { registerClassifier } from '../../registry/viewRegistry';
import { knnTestCases } from './testCases';
import { knnFormulas } from './formulas';
import { knnDerivations } from './derivations';
import { knnMistakes } from './mistakes';
import { knnQuestions } from './questions';
import { knnComparisons } from './comparisons';
import { knnFailureDemos } from './failures';

export type KnnMetric = 'euclidean' | 'manhattan';

export interface KnnPoint { x: number; y: number; cls: number; }

// Fixed jitter spread for seeded generation; margins + spread keep points in [−5,5]².
const SPREAD = 1.5;
const DOMAIN = 5;

/** Mulberry32 — deterministic PRNG so runs are reproducible. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Parse the optional serialized custom dataset `points` (JSON [[x,y,cls], …]). */
export function parsePoints(s: unknown): KnnPoint[] {
  if (typeof s !== 'string' || s.trim() === '') return [];
  try {
    const raw = JSON.parse(s) as unknown;
    if (!Array.isArray(raw)) return [];
    const pts: KnnPoint[] = [];
    for (const row of raw) {
      if (!Array.isArray(row) || row.length < 3) return [];
      const [x, y, cls] = row as number[];
      if (!Number.isFinite(x) || !Number.isFinite(y) || (cls !== 0 && cls !== 1)) return [];
      pts.push({ x, y, cls });
    }
    return pts;
  } catch {
    return [];
  }
}

/**
 * Deterministic dataset: two jittered clusters at x = ±margin (class 0 left, class 1
 * right) — overlapping at moderate margin so k visibly smooths the boundary. When the
 * optional `points` param is present (validated: even count, equal class counts, within
 * [−5,5]²), it REPLACES the seeded generation entirely — the drag/click-add hook.
 */
export function generatePoints(p: Params): KnnPoint[] {
  const custom = parsePoints(p.points);
  if (custom.length > 0) return custom;
  const nPerClass = (p.nPerClass as number) ?? 12;
  const margin = (p.margin as number) ?? 1.0;
  const rng = mulberry32((p.seed as number) ?? 42);
  const pts: KnnPoint[] = [];
  for (let c = 0; c < 2; c++) {
    const cx = c === 0 ? -margin : margin;
    for (let i = 0; i < nPerClass; i++) {
      const x = Math.max(-DOMAIN, Math.min(DOMAIN, cx + (rng() - 0.5) * 2 * SPREAD));
      const y = Math.max(-DOMAIN, Math.min(DOMAIN, (rng() - 0.5) * 2 * SPREAD));
      pts.push({ x, y, cls: c });
    }
  }
  return pts;
}

// --- Point-set memoization ----------------------------------------------
// DecisionBoundary resolves the classifier 2500× per snapshot (50×50 grid); before this
// cache every call re-ran the O(n) generation. getPoints() is now the SINGLE source of
// truth: initialState, step and the classifier all reach the SAME cached array for a
// given params key. The key captures every input that changes the set — seed, nPerClass,
// margin and the optional `points` JSON override (a drag/click edit changes the JSON
// string, so the key changes and the cache invalidates naturally). Deterministic:
// same key → same array instance.
const POINTS_CACHE = new Map<string, KnnPoint[]>();
const POINTS_CACHE_MAX = 64;

function pointsCacheKey(p: Params): string {
  const custom = p.points;
  if (typeof custom === 'string' && custom.trim() !== '') return `points:${custom}`;
  return `seed:${(p.seed as number) ?? 42}|nPerClass:${(p.nPerClass as number) ?? 12}|margin:${(p.margin as number) ?? 1.0}`;
}

/** Cached generatePoints — same params key ⇒ same array (never mutated). */
export function getPoints(p: Params): KnnPoint[] {
  const key = pointsCacheKey(p);
  let pts = POINTS_CACHE.get(key);
  if (!pts) {
    pts = generatePoints(p);
    if (POINTS_CACHE.size >= POINTS_CACHE_MAX) POINTS_CACHE.clear(); // bounded memory
    POINTS_CACHE.set(key, pts);
  }
  return pts;
}

/** Distance between a stored point and a query (Euclidean or Manhattan). */
export function distance(pt: KnnPoint, qx: number, qy: number, metric: KnnMetric): number {
  const dx = pt.x - qx;
  const dy = pt.y - qy;
  return metric === 'manhattan' ? Math.abs(dx) + Math.abs(dy) : Math.hypot(dx, dy);
}

export interface Neighbor { idx: number; d: number; cls: number; }

/**
 * The k nearest neighbors of (qx,qy), sorted by (distance, index) — index tie-break
 * keeps the k-BOUNDARY deterministic when several points sit at the k-th distance.
 */
export function nearestBy(points: KnnPoint[], qx: number, qy: number, k: number, metric: KnnMetric): Neighbor[] {
  const kk = Math.max(1, Math.min(Math.floor(k) || 1, points.length));
  return points
    .map((pt, i) => ({ idx: i, d: distance(pt, qx, qy, metric), cls: pt.cls }))
    .sort((a, b) => (a.d - b.d) || (a.idx - b.idx))
    .slice(0, kk);
}

/**
 * Deterministic majority vote. Votes per class over the k neighbors; on a vote tie the
 * class whose nearest neighbor (smallest distance INSIDE the k-neighborhood) wins; on an
 * equal nearest distance, the lower class index wins. This is documented + tested with
 * crafted exact ties (both at d=1, k=2 → class 0).
 */
export function majorityClass(neighbors: Neighbor[]): number {
  const votes = [0, 0];
  for (const nb of neighbors) votes[nb.cls]++;
  const maxV = Math.max(votes[0], votes[1]);
  const tied = votes
    .map((v, c) => ({ c, v }))
    .filter((t) => t.v === maxV)
    .map((t) => t.c);
  tied.sort((c1, c2) => {
    const d1 = Math.min(...neighbors.filter((nb) => nb.cls === c1).map((nb) => nb.d));
    const d2 = Math.min(...neighbors.filter((nb) => nb.cls === c2).map((nb) => nb.d));
    return (d1 - d2) || (c1 - c2);
  });
  return tied[0];
}

/** Classify (qx,qy) by k-NN over the dataset (lazy learner — no training phase). */
export function knnClassify(points: KnnPoint[], qx: number, qy: number, k: number, metric: KnnMetric): number {
  return majorityClass(nearestBy(points, qx, qy, k, metric));
}

/** Self-classification error: k=1 → exactly 0 (each point is its own nearest neighbor). */
export function trainErrorOf(points: KnnPoint[], k: number, metric: KnnMetric): number {
  let err = 0;
  for (const pt of points) {
    if (knnClassify(points, pt.x, pt.y, k, metric) !== pt.cls) err++;
  }
  return err / points.length;
}

/** Leave-one-out error per k — the honest 'error' metric (k=1 does NOT score 0 here). */
export function looErrorOf(points: KnnPoint[], k: number, metric: KnnMetric): number {
  let err = 0;
  for (let i = 0; i < points.length; i++) {
    const others = points.filter((_, j) => j !== i);
    if (knnClassify(others, points[i].x, points[i].y, k, metric) !== points[i].cls) err++;
  }
  return err / points.length;
}

/**
 * Decision-region complexity: classify a G×G grid over [−5,5]² and count adjacent label
 * transitions (horizontal + vertical). Deterministic; falls OVERALL as k smooths the
 * boundary (51 → 35 on the default seed at k=1 vs k=15) — a trend, not a strict monotone
 * (finite-sample wobbles, and the vote saturates past k ≈ n/2).
 */
export function regionCount(points: KnnPoint[], k: number, metric: KnnMetric, G = 31): number {
  let changes = 0;
  const grid: number[][] = [];
  for (let gx = 0; gx < G; gx++) {
    const row: number[] = [];
    const x = -DOMAIN + (2 * DOMAIN * gx) / (G - 1);
    for (let gy = 0; gy < G; gy++) {
      const y = -DOMAIN + (2 * DOMAIN * gy) / (G - 1);
      row.push(knnClassify(points, x, y, k, metric));
    }
    grid.push(row);
  }
  for (let gx = 0; gx < G; gx++) {
    for (let gy = 0; gy < G - 1; gy++) if (grid[gx][gy] !== grid[gx][gy + 1]) changes++;
  }
  for (let gy = 0; gy < G; gy++) {
    for (let gx = 0; gx < G - 1; gx++) if (grid[gx][gy] !== grid[gx + 1][gy]) changes++;
  }
  return changes;
}

// Class colors mirror the decision-boundary palette (class 0 blue / class 1 red).
const CLS_COLORS = ['#3b82f6', '#ef4444'];
const QUERY_COLOR = '#0f172a';
const RING_COLOR = '#94a3b8';

function metricOf(p: Params): KnnMetric {
  return p.metric === 'manhattan' ? 'manhattan' : 'euclidean';
}
function kOf(p: Params): number {
  return Math.max(1, Math.floor((p.k as number) ?? 5) || 1);
}

function buildVisuals(points: KnnPoint[], qx: number, qy: number, k: number, metric: KnnMetric): VisualCommand[] {
  const pts = points.map((pt, i) => ({
    type: 'point', id: `d${i}`, x: pt.x, y: pt.y, color: CLS_COLORS[pt.cls],
  }));
  const nbs = nearestBy(points, qx, qy, k, metric);
  // Nested distance rings: one circle per neighbor distance — scrub k and the rings
  // visibly grow outward (the "expanding neighborhood" animation).
  const rings: VisualCommand[] = nbs.map((nb, j) => ({
    type: 'circle', id: `ring${j}`, x: qx, y: qy, r: nb.d, color: RING_COLOR,
  }));
  const query: VisualCommand = { type: 'point', id: 'query', x: qx, y: qy, color: QUERY_COLOR };
  return [...pts, ...rings, query];
}

/** Highlights for the k nearest neighbors (ScatterPlot renders them larger + amber). */
function nearestHighlights(nbs: Neighbor[], first: boolean): SimState['highlights'] {
  const hl: SimState['highlights'] = nbs.map((nb) => ({ panel: 'canvas', id: `d${nb.idx}`, intensity: 1 }));
  if (first) hl.push({ panel: 'canvas', id: 'query', intensity: 1 });
  return hl;
}

function snapshotAt(p: Params, points: KnnPoint[], k: number, first: boolean): SimState {
  const metric = metricOf(p);
  const qx = (p.queryX as number) ?? 0;
  const qy = (p.queryY as number) ?? 0;
  const train = trainErrorOf(points, k, metric);
  const loo = looErrorOf(points, k, metric);
  const regions = regionCount(points, k, metric);
  const qClass = knnClassify(points, qx, qy, k, metric);
  const nbs = nearestBy(points, qx, qy, k, metric);
  const kthDist = nbs.length > 0 ? nbs[nbs.length - 1].d : NaN;

  const math: MathStep[] = [
    { latex: 'd(p, q) = \\sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2}', id: 'knn-euclidean' },
    { latex: '\\hat{y}(q) = \\arg\\max_c \\sum_{i \\in N_k(q)} [y_i = c]', id: 'knn-majority-vote' },
  ];

  return {
    algorithm: { mode: 'knn', k, metric } as Record<string, ParamValue>,
    visuals: buildVisuals(points, qx, qy, k, metric),
    math,
    narration: `k = ${k} (${metric}): train error = ${train.toFixed(3)}, LOO error = ${loo.toFixed(3)}, ` +
      `regions = ${regions}, query (${qx.toFixed(1)}, ${qy.toFixed(1)}) → class ${qClass}, k-th neighbor at d = ${Number.isFinite(kthDist) ? kthDist.toFixed(2) : '—'}`,
    explanation: {
      changed: first ? [] : [`k → ${k}`, `train error → ${train.toFixed(3)}`, `LOO error → ${loo.toFixed(3)}`, `regions → ${regions}`],
      why: first
        ? `k = 1 (${metric}): overfit signature — train error = ${train.toFixed(3)} (memorization: each point is its own nearest neighbor) while LOO error = ${loo.toFixed(3)} sits at the high end of the honest curve. regions = ${regions}`
        : `k = ${k} (${metric}): the vote widens — train error ${train.toFixed(3)} rises from 0 (memorization fades) as LOO error ${loo.toFixed(3)} falls from its k=1 high (≈ 0.42 on the default seed); past k ≈ n/2 = ${(points.length / 2).toFixed(0)} the majority vote saturates and the honest error stops improving (both curves sit ≈ 0.21 at k = 15–18, then LOO creeps back up at k = 19–20). regions = ${regions}`,
      formulaRef: 'knn-majority-vote',
      dependsOn: ['distance-metrics', 'majority-vote'],
      gateConcepts: ['k-NN', 'nearest-neighbor', metric === 'manhattan' ? 'L1 distance' : 'L2 distance'],
    },
    highlights: nearestHighlights(nbs, first),
    metrics: { k, trainError: train, error: loo, regions, queryClass: qClass, nPoints: points.length },
    events: [{ type: 'fit', label: 'knn-classify', step: 0 }],
    timeline: first ? ['Data', 'Neighbors', 'Vote', 'Evaluate'] : ['Neighbors', 'Vote', 'Evaluate'],
  };
}

export const simulation = {
  /**
   * One snapshot per k on [1, 2, …, params.k]; the first snapshot is the k = 1
   * reference and each step advances k by 1 up to the slider value. Scrubbing the run
   * IS the boundary-smoothing + ring-expansion animation. Single-shot per k (lazy
   * learner — no training epochs to step through).
   */
  initialState: (p: Params): SimState => {
    const points = getPoints(p);
    return snapshotAt(p, points, 1, true);
  },

  step: (p: Params, s: SimState): SimState | null => {
    const points = getPoints(p);
    const target = kOf(p);
    const current = (s.algorithm.k as number) ?? 1;
    const next = current + 1;
    if (next > target) return null; // sweep complete
    return snapshotAt(p, points, next, false);
  },
};

export const knnModule: TopicModule = {
  id: 'knn',
  title: 'K-Nearest Neighbors',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 2, mathematical: 2, coding: 3, visualization: 3, gateFrequency: 4 },
    estimatedHours: 4,
    revisionPriority: 'P1',
    examFrequency: 'Frequent',
    prerequisites: ['distance-metrics', 'probability', 'linear-algebra'],
    relatedTopics: ['naive-bayes', 'decision-tree', 'svm', 'logistic-regression'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Data, Query Point, k-Neighbors & Distance Rings' },
      { slot: 'primary', component: 'decision-boundary', title: 'Decision Regions (k-NN classifier)' },
      { slot: 'sidebar', component: 'loss-curve', title: 'Train vs Leave-One-Out Error vs k' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivation: Majority Vote & Boundaries' },
      { slot: 'primary', component: 'explain-step', title: 'Step Explanation' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'k', label: 'Neighbors (k)', type: 'number', min: 1, max: 20, step: 1, default: 5 },
    {
      id: 'metric', label: 'Distance metric', type: 'select',
      options: [
        { value: 'euclidean', label: 'Euclidean (L2)' },
        { value: 'manhattan', label: 'Manhattan (L1)' },
      ],
      default: 'euclidean',
    },
    { id: 'nPerClass', label: 'Points per class', type: 'number', min: 2, max: 15, step: 1, default: 12 },
    { id: 'margin', label: 'Cluster separation', type: 'number', min: 0.5, max: 2.5, step: 0.1, default: 1.0 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
    { id: 'queryX', label: 'Query x', type: 'number', min: -5, max: 5, step: 0.1, default: 0 },
    { id: 'queryY', label: 'Query y', type: 'number', min: -5, max: 5, step: 0.1, default: 0 },
  ],
  simulation,
  formulas: knnFormulas,
  derivations: knnDerivations,
  questions: knnQuestions,
  comparisons: knnComparisons,
  failureDemos: knnFailureDemos,
  mistakes: knnMistakes,
  testCases: knnTestCases,
  lossMetricKey: 'error',
  lossMetricKey2: 'trainError',

  validateParams: (p) => {
    const issues: string[] = [];
    const k = (p.k as number) ?? 5;
    const nPerClass = (p.nPerClass as number) ?? 12;
    const metric = (p.metric as string) ?? 'euclidean';
    if (!Number.isFinite(k) || k < 1) issues.push('k must be ≥ 1');
    const custom = parsePoints(p.points);
    if (custom.length > 0) {
      if (custom.length < 2) issues.push('Custom points need at least 2 points');
      if (custom.length % 2 !== 0) issues.push('Custom points must have an even count (equal classes)');
      if (custom.filter((pt) => pt.cls === 0).length !== custom.filter((pt) => pt.cls === 1).length) {
        issues.push('Custom points must have equal class counts (balanced classes)');
      }
      for (const pt of custom) {
        if (Math.abs(pt.x) > DOMAIN || Math.abs(pt.y) > DOMAIN) {
          issues.push('Custom points must lie within [−5,5]² (the visualization domain)');
          break;
        }
      }
      if (k > custom.length) issues.push(`k = ${k} exceeds the custom dataset size (${custom.length})`);
    } else {
      if (!Number.isFinite(nPerClass) || nPerClass < 1) issues.push('nPerClass must be ≥ 1');
      if (k > 2 * nPerClass) issues.push(`k = ${k} exceeds the dataset size (2·nPerClass = ${2 * nPerClass}) — every point would be a neighbor`);
    }
    if (metric !== 'euclidean' && metric !== 'manhattan') issues.push('metric must be euclidean or manhattan');
    const qx = p.queryX as number | undefined;
    const qy = p.queryY as number | undefined;
    if (qx !== undefined && !Number.isFinite(qx)) issues.push('queryX must be finite');
    if (qy !== undefined && !Number.isFinite(qy)) issues.push('queryY must be finite');
    return issues;
  },
};

export function register() {
  registerTopic(knnModule);
  // Deterministic per params (seeded generation or points override): DecisionBoundary
  // resolves this classifier via getClassifier('knn') to paint the region grid (2500
  // calls per snapshot — getPoints() serves the cached point set so only the FIRST call
  // per snapshot pays the O(n) generation cost).
  registerClassifier('knn', (x, y, params) => {
    const points = getPoints(params);
    return knnClassify(points, x, y, kOf(params), metricOf(params));
  });
}
