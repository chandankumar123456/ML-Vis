// src/topics/hierarchical-clustering/module.ts
// Task 23 (Wave 5): hierarchical-clustering — AGGLOMERATIVE (bottom-up)
// hierarchical clustering in 2-D with four linkage criteria (single, complete,
// average, Ward), a growing dendrogram, an honest cophenetic correlation, and
// a cut-height slider that partitions the tree.
//
// The algorithm (pure + deterministic):
//   1. Generate n points from 2-3 Gaussian blobs (mulberry32 + Box-Muller —
//      NO Math.random anywhere; the mulberry32 PRNG is a documented copy of the
//      one in src/topics/pca/module.ts, Wave 4, which every Wave-1/2/3/4 topic
//      shares).
//   2. Compute the n×n Euclidean distance matrix.
//   3. Start with n singleton clusters. Repeat n−1 times: evaluate the linkage
//      value of every cluster pair, merge the minimum pair (tie-break: smaller
//      cluster ids), record the merge {height, children}. One SNAPSHOT PER
//      MERGE, each emitting the full merge list so far (chronological) as
//      dendrogram {type:'merge'} commands.
//   4. The run terminates after n−1 merges (step() returns null), at which
//      point the final snapshot also carries the cophenetic correlation and
//      the cut-height partition.
//
// Complexity: O(n³) naive (n ≤ 20 enforced by validateParams). Each merge
// re-evaluates every cluster pair from the raw point sets — honest and simple;
// the plan's O(n³) failure case documents why n is capped (see failures.ts).
//
// DESIGN DECISIONS (documented drift from the plan — see the report):
//  - cluster-animator SUBSTITUTION: the plan's registry table lists
//    cluster-animator as a consumer, but that view is CENTROID-based
//    (centroids + assignment lines + convergence trails) and agglomerative
//    clustering has no centroids — emitting centroid commands would be
//    misleading. Substituted: scatter-plot (points colored by their CURRENT
//    cluster at each merge), dendrogram (the merge tree), matrix-animator
//    (the distance matrix). Revisit cluster-animator if a future topic needs
//    centroids (kmeans is the right consumer; it is a separate topic).
//  - matrix-animator renders a NUMERIC GRID (each cell toFixed(2)), not a
//    colored heatmap — the plan says "distance matrix heatmap"; the registry
//    component renders numbers. The distance matrix is STATIC across the run
//    (it is the input data; the dendrogram carries the evolving CLUSTER
//    distances). The final snapshot adds the cophenetic distance matrix.
//  - loss-curve OMITTED (lossMetricKey not set): the natural candidate
//    copheneticCorr is only defined once the full tree exists (the final
//    snapshot), so a per-merge loss curve would be zeros + one jump — the
//    naive-bayes precedent (no loss-curve layer, no lossMetricKey).
//  - Ward's merge heights are SSE-INCREASE units (squared distance), not raw
//    distances: ΔSSE(A,B) = (|A|·|B|)/(|A|+|B|) · ‖μ_A−μ_B‖². The dendrogram
//    axis is honest about being "merge cost" — the trap question exploits this.
//  - cutHeight slider: cuts the FINAL tree only (partition = merges with
//    height ≤ cutHeight). Default measured so the default run cuts the two
//    blobs apart (see anchors below).
//  - test-only `points` override (JSON '[[x,y],…]'): hand-crafted datasets for
//    the hand-computed test cases and the failure demos (chaining, outliers).
//
// MEASURED ANCHORS (ALL printed by the module before being asserted — the
// honesty rule: every hardcoded number below is a measured value):
//   default run (n 12, linkage single, blobs 2, seed 42, cutHeight 1.2):
//     merge heights  [0.241069, 0.266371, 0.352660, 0.543091, 0.593911,
//                     0.596020, 0.693380, 0.733356, 0.880545, 0.895525,
//                     1.794794]  (11 merges — the last one bridges the two
//                     blobs at 1.795, a clear gap after 0.896)
//     final snapshot: clusterCount 1, cutClusters@1.2 = 2 (the two blobs),
//                     cut@median-height (0.596) = 6, cut@0.75·max = 2,
//                     copheneticCorr = 0.901
//   n 8 (single, blobs 2, seed 42): heights [0.372182, 0.585626, 0.593911,
//     0.596020, 0.702012, 1.176052, 3.170485]; copheneticCorr = 0.960;
//     cutHeight 2.0 → 2 clusters.
//   ward (n 10, blobs 2, seed 42): heights [0.035477, 0.143950, 0.177620,
//     0.240388, 0.389290, 0.473884, 1.341615, 2.225187, 39.940716] in
//     SSE-increase units (non-decreasing); copheneticCorr = 0.900.
//   failure demos: chaining (6 collinear points) → five merges all at height
//     1.000; outliers (9 blob + 1 at (5,5)) → blob merges 0.224–0.361, outlier
//     joins at 6.727 (cut below 6.727 leaves it a singleton → 2 clusters).
//   dataset A copheneticCorr ≈ 0.8985 (hand-computed in testCases.test.ts).
//
// NARRATION honesty: every number in the narration is computed by the module at
// run time from the actual merges — no fabricated values.
import type {
  TopicModule, Params, SimState, VisualCommand, ParamValue, SimulationDef,
} from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { hierarchicalTestCases } from './testCases';
import { hierarchicalFormulas } from './formulas';
import { hierarchicalDerivations } from './derivations';
import { hierarchicalMistakes } from './mistakes';
import { hierarchicalQuestions } from './questions';
import { hierarchicalComparisons } from './comparisons';
import { hierarchicalFailureDemos } from './failures';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export type Linkage = 'single' | 'complete' | 'average' | 'ward';
export const LINKAGES: Linkage[] = ['single', 'complete', 'average', 'ward'];

// Distinct cluster colors indexed by the cluster's MINIMUM leaf index (n ≤ 20,
// so every cluster in a run gets its own color — colors change ONLY when two
// clusters merge, which is exactly the "colored by CURRENT cluster" story).
export const PALETTE = [
  '#2563eb', '#dc2626', '#16a34a', '#9333ea', '#f59e0b', '#0d9488', '#db2777',
  '#64748b', '#65a30d', '#7c3aed', '#c2410c', '#0891b2', '#a21caf', '#4d7c0f',
  '#0f766e', '#b91c1c', '#1d4ed8', '#a16207', '#be123c', '#155e75',
];

// Gaussian-blob synthesis: centers for 2 and 3 blobs, per-axis jitter σ.
const BLOB_CENTERS: Record<number, [number, number][]> = {
  2: [[-2, 0], [2, 0]],
  3: [[-2, -1.2], [2, -1.2], [0, 1.6]],
};
const BLOB_JITTER = 0.55;

// ---------------------------------------------------------------------------
// Deterministic PRNG + data synthesis (mulberry32 copied from pca/module.ts —
// the shared Wave-4 PRNG; documented copy per the cross-topic convention)
// ---------------------------------------------------------------------------

export interface HcPoint { x: number; y: number; }

/** Mulberry32 — deterministic PRNG (documented copy of src/topics/pca/module.ts). */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller standard normal (spare-draw trick) — copied with mulberry32. */
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
 * Deterministic data synthesis. A test-only `points` override (JSON
 * '[[x,y],…]', 4..20 points) bypasses generation entirely — the hand-crafted
 * datasets behind the hand-computed tests and the failure demos. Otherwise:
 * n points drawn from `blobs` (2 or 3) Gaussian blobs with per-axis jitter.
 * Points are assigned to blobs in CONTIGUOUS index ranges (chunked), so a blob
 * is a run of indices — convenient for narration and scatter labels.
 */
export function generateData(p: Params): HcPoint[] {
  if (typeof p.points === 'string') {
    return (JSON.parse(p.points) as [number, number][]).map(([x, y]) => ({ x, y }));
  }
  const n = (p.n as number) ?? 12;
  const blobs = Number(p.blobs) || 2;
  const seed = (p.seed as number) ?? 42;
  const rng = mulberry32(seed);
  const normal = makeNormal(rng);
  const centers = BLOB_CENTERS[blobs] ?? BLOB_CENTERS[2];
  const pts: HcPoint[] = [];
  for (let i = 0; i < n; i++) {
    const blob = Math.min(Math.floor((i * blobs) / n), centers.length - 1);
    const c = centers[blob];
    pts.push({ x: c[0] + BLOB_JITTER * normal(), y: c[1] + BLOB_JITTER * normal() });
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Agglomerative math core (all mutation-free / pure, exported for tests)
// ---------------------------------------------------------------------------

/** Symmetric n×n Euclidean distance matrix (zero diagonal). */
export function distanceMatrix(points: HcPoint[]): number[][] {
  const n = points.length;
  const D = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const v = Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y);
      D[i][j] = v; D[j][i] = v;
    }
  }
  return D;
}

/**
 * Linkage value between two clusters (index sets) under the criterion:
 *   single    min d(a,b) over all pairs
 *   complete  max d(a,b)
 *   average   mean d(a,b)
 *   ward      (|A||B|)/(|A|+|B|) · ‖μ_A − μ_B‖²  — the increase in the
 *             within-cluster sum of squares (variance units, NOT a distance).
 */
export function linkageDistance(
  D: number[][], points: HcPoint[], a: number[], b: number[], linkage: Linkage,
): number {
  switch (linkage) {
    case 'single': {
      let best = Infinity;
      for (const i of a) for (const j of b) best = Math.min(best, D[i][j]);
      return best;
    }
    case 'complete': {
      let best = 0;
      for (const i of a) for (const j of b) best = Math.max(best, D[i][j]);
      return best;
    }
    case 'average': {
      let sum = 0;
      for (const i of a) for (const j of b) sum += D[i][j];
      return sum / (a.length * b.length);
    }
    case 'ward': {
      let ax = 0, ay = 0, bx = 0, by = 0;
      for (const i of a) { ax += points[i].x; ay += points[i].y; }
      for (const j of b) { bx += points[j].x; by += points[j].y; }
      const mux = ax / a.length, muy = ay / a.length;
      const dx = bx / b.length - mux, dy = by / b.length - muy;
      return (a.length * b.length) / (a.length + b.length) * (dx * dx + dy * dy);
    }
  }
}

export interface MergeResult {
  id: string;            // 'm1'…'m(n−1)' — chronological
  height: number;        // linkage value in distance (or Ward: SSE-increase) units
  children: string[];    // leaf ids 'p{i}' or earlier merge ids 'm{k}'
  membersA?: number[];   // leaf indices of the first child cluster (narration)
  membersB?: number[];   // leaf indices of the second child cluster (narration)
}

export interface AggloResult {
  points: HcPoint[];
  linkage: Linkage;
  merges: MergeResult[];        // n−1 merges, chronological
  partitions: number[][][];     // partitions[k] = cluster partition AFTER merge k (0-indexed k)
  distanceMatrix: number[][];   // n×n Euclidean (the matrix-animator grid)
  cophenetic: number[][];       // n×n cophenetic distances (final snapshot only)
  copheneticCorr: number;       // Pearson(d, c) over off-diagonal pairs
  maxHeight: number;
  dataSeed: number;
}

/** Pearson correlation (0 when either series is constant — the degenerate guard). */
export function pearson(xs: number[], ys: number[]): number {
  const m = xs.length;
  if (m === 0 || m !== ys.length) return 0;
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < m; i++) {
    sx += xs[i]; sy += ys[i];
    sxx += xs[i] * xs[i]; syy += ys[i] * ys[i];
    sxy += xs[i] * ys[i];
  }
  const den = Math.sqrt((m * sxx - sx * sx) * (m * syy - sy * sy));
  if (den < 1e-12) return 0; // constant series — correlation undefined; honest 0
  return (m * sxy - sx * sy) / den;
}

/** Cophenetic correlation: Pearson between pairwise distances and merge heights. */
export function copheneticCorrelation(D: number[][], C: number[][]): number {
  const n = D.length;
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) { xs.push(D[i][j]); ys.push(C[i][j]); }
  }
  return pearson(xs, ys);
}

/**
 * The full agglomerative run. Purely functional: returns the merge list, the
 * partition after every merge, the cophenetic matrix + correlation. Pair
 * selection is deterministic — the lexicographically smallest (value, cluster
 * min-id) wins, so ties (duplicate points) resolve by cluster id order.
 */
export function agglomerate(points: HcPoint[], linkage: Linkage): AggloResult {
  const n = points.length;
  const D = distanceMatrix(points);
  let clusters = points.map((_, i) => ({ members: [i], ref: `p${i}`, minId: i }));
  const merges: MergeResult[] = [];
  const partitions: number[][][] = [];
  for (let k = 1; k < n; k++) {
    let bestV = Infinity;
    let bestA = -1;
    let bestB = -1;
    let bestMinA = Infinity;
    let bestMinB = Infinity;
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const v = linkageDistance(D, points, clusters[i].members, clusters[j].members, linkage);
        const minA = clusters[i].minId;
        const minB = clusters[j].minId;
        if (
          v < bestV - 1e-12 ||
          (Math.abs(v - bestV) <= 1e-12 && (minA < bestMinA || (minA === bestMinA && minB < bestMinB)))
        ) {
          bestV = v; bestA = i; bestB = j; bestMinA = minA; bestMinB = minB;
        }
      }
    }
    const rawA = clusters[bestA];
    const rawB = clusters[bestB];
    const [A, B] = rawA.minId <= rawB.minId ? [rawA, rawB] : [rawB, rawA];
    merges.push({ id: `m${k}`, height: bestV, children: [A.ref, B.ref], membersA: A.members, membersB: B.members });
    const merged = { members: [...A.members, ...B.members], ref: `m${k}`, minId: Math.min(A.minId, B.minId) };
    clusters.splice(Math.max(bestA, bestB), 1);
    clusters.splice(Math.min(bestA, bestB), 1);
    clusters.push(merged);
    partitions.push(clusters.map((c) => c.members.slice()));
  }

  // Cophenetic distance c(i,j) = the height of the FIRST merge that puts i and
  // j in the same cluster (their lowest common ancestor in the dendrogram).
  const cophenetic = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = 0; k < partitions.length; k++) {
        const ci = partitions[k].find((c) => c.includes(i));
        if (ci !== undefined && ci.includes(j)) {
          cophenetic[i][j] = cophenetic[j][i] = merges[k].height;
          break;
        }
      }
    }
  }
  return {
    points, linkage, merges, partitions, distanceMatrix: D,
    cophenetic, copheneticCorr: copheneticCorrelation(D, cophenetic),
    maxHeight: merges.length > 0 ? merges[merges.length - 1].height : 0,
    dataSeed: 0,
  };
}

/** Cluster partition after cutting the dendrogram at height h (merges ≤ h). */
export function clustersAtHeight(r: AggloResult, h: number): number[][] {
  let k = 0;
  while (k < r.merges.length && r.merges[k].height <= h) k++;
  if (k === 0) return r.points.map((_, i) => [i]);
  return r.partitions[k - 1].map((c) => c.slice());
}

export function clusterCountAtHeight(r: AggloResult, h: number): number {
  return clustersAtHeight(r, h).length;
}

/** Median merge height (the plan's "cut at the median" demo anchor). */
export function medianHeight(r: AggloResult): number {
  const hs = r.merges.map((m) => m.height);
  if (hs.length === 0) return 0;
  const sorted = hs.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Bounded memoization (the svm/lda precedent): initialState/step stay O(1)
// after the first evaluation of a params key.
const RESULT_CACHE = new Map<string, AggloResult>();
const RESULT_CACHE_MAX = 16;

function resultKey(p: Params): string {
  return JSON.stringify([p.n, p.linkage, p.blobs, p.seed, p.points ?? null]);
}

export function computeResult(p: Params): AggloResult {
  const key = resultKey(p);
  let r = RESULT_CACHE.get(key);
  if (!r) {
    r = agglomerate(generateData(p), ((p.linkage as Linkage) ?? 'single'));
    r.dataSeed = (p.seed as number) ?? 42;
    if (RESULT_CACHE.size >= RESULT_CACHE_MAX) RESULT_CACHE.clear();
    RESULT_CACHE.set(key, r);
  }
  return r;
}

// ---------------------------------------------------------------------------
// Visual command builders
// ---------------------------------------------------------------------------

/** Scatter points colored by their cluster at THIS step (partition). */
function scatterCommands(r: AggloResult, partition: number[][]): VisualCommand[] {
  const colors: string[] = new Array(r.points.length).fill(PALETTE[0]);
  for (const cluster of partition) {
    const color = PALETTE[Math.min(...cluster) % PALETTE.length];
    for (const i of cluster) colors[i] = color;
  }
  return r.points.map((p, i) => ({
    type: 'point', id: `p${i}`, x: p.x, y: p.y, color: colors[i],
  }));
}

/** Dendrogram commands: ALL merges so far (chronological), per the contract. */
function mergeCommands(r: AggloResult, k: number): VisualCommand[] {
  return r.merges.slice(0, k).map((m) => ({
    type: 'merge', id: m.id, height: m.height, children: m.children,
  }));
}

/**
 * Matrix-animator commands. The distance matrix is STATIC across the run (it
 * is the input data — the dendrogram carries the evolving CLUSTER distances);
 * the view renders it as a NUMERIC GRID (not a heatmap). The final snapshot
 * adds the cophenetic matrix so the correlation is visually inspectable.
 */
function matrixCommands(r: AggloResult, includeCophenetic: boolean): VisualCommand[] {
  const n = r.points.length;
  const cmds: VisualCommand[] = [{
    type: 'matrix', id: 'D: pairwise distances', rows: n, cols: n, cells: r.distanceMatrix,
  }];
  if (includeCophenetic) {
    cmds.push({
      type: 'matrix', id: 'C: cophenetic distances', rows: n, cols: n, cells: r.cophenetic,
    });
  }
  return cmds;
}

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

const LINKAGE_NAME: Record<Linkage, string> = {
  single: 'single linkage',
  complete: 'complete linkage',
  average: 'average linkage',
  ward: "Ward's method",
};

const LINKAGE_REASON: Record<Linkage, string> = {
  single: 'the closest pair of members across the two clusters (min pairwise distance)',
  complete: 'the farthest pair of members across the two clusters (max pairwise distance)',
  average: 'the mean of all |A|·|B| pairwise distances between members',
  ward: 'the increase in the within-cluster sum of squares (a variance quantity, in squared units) is minimal',
};

function describeCluster(members: number[]): string {
  return `{${members.map((i) => `p${i}`).join(', ')}}`;
}

function baseAlgorithm(r: AggloResult, k: number): Record<string, ParamValue> {
  return {
    mode: k === r.merges.length ? 'agglomerative-complete' : 'agglomerative-merge',
    step: k,
    linkage: r.linkage,
    n: r.points.length,
    dataSeed: r.dataSeed,
    mergeId: r.merges[k - 1].id,
    mergeHeight: r.merges[k - 1].height,
    clusterCount: r.points.length - k,
    isDone: k === r.merges.length ? 1 : 0,
  };
}

function mergeSnapshot(r: AggloResult, k: number, first: boolean, p: Params): SimState {
  const m = r.merges[k - 1];
  const partition = r.partitions[k - 1];
  const clusterCount = r.points.length - k;
  const reason = LINKAGE_REASON[r.linkage];
  const aDesc = describeCluster(m.membersA ?? []);
  const bDesc = describeCluster(m.membersB ?? []);
  const narration =
    `Merge ${k}/${r.merges.length}: joining ${aDesc} and ${bDesc} at height ${m.height.toFixed(3)} — ` +
    `${LINKAGE_NAME[r.linkage]}: ${reason}. clusterCount → ${clusterCount}. ` +
    (k < r.merges.length
      ? `Keep merging the closest remaining clusters until ONE cluster holds all ${r.points.length} points.`
      : 'This is the LAST merge: the whole dataset is now one cluster — the tree is complete.');
  return {
    algorithm: baseAlgorithm(r, k) as Record<string, ParamValue>,
    visuals: [
      ...scatterCommands(r, partition),
      ...mergeCommands(r, k),
      ...matrixCommands(r, false),
    ],
    math: [
      { latex: 'd(a, b) = \\sqrt{(x_a - x_b)^2 + (y_a - y_b)^2}', id: 'hc-distance' },
      ...(k === 1
        ? [{ latex: 'd(A, B) = \\min_{a \\in A, b \\in B} d(a, b)', id: 'hc-linkage-single' }]
        : []),
    ],
    narration,
    explanation: {
      changed: first
        ? [
            `data: ${r.points.length} points from ${Number(p.blobs ?? 2)} Gaussian blobs (seed ${r.dataSeed})`,
            `distance matrix D: ${r.points.length}×${r.points.length} Euclidean pairwise (matrix-animator grid)`,
            `merge 1: ${aDesc} ∪ ${bDesc} at height ${m.height.toFixed(3)}`,
          ]
        : [
            `merge ${k}: ${aDesc} ∪ ${bDesc} at height ${m.height.toFixed(3)}`,
            `clusterCount → ${clusterCount}`,
          ],
      why: first
        ? `Agglomerative clustering starts with ${r.points.length} singleton clusters (one per point) and merges the CLOSEST pair at every step. ` +
          `"Closest" is defined by the linkage criterion — with ${LINKAGE_NAME[r.linkage]} that is ${reason}. ` +
          `Each merge grows a subtree in the dendrogram; the height of a merge is the linkage value at which the two clusters became one.`
        : `The previous merge shrank the cluster set by one. Re-evaluate every remaining pair under ${LINKAGE_NAME[r.linkage]} and join the minimum — ` +
          `${reason}. The merge heights are non-decreasing, so the dendrogram grows strictly upward.`,
      formulaRef: 'hc-linkage-single',
      dependsOn: ['probability', 'statistics', 'linear-algebra'],
      gateConcepts: ['hierarchical clustering', 'agglomerative', 'linkage', 'dendrogram'],
    },
    highlights: [
      ...(m.membersA ?? []).map((i) => ({ panel: 'canvas', id: `p${i}`, intensity: 0.6 })),
      ...(m.membersB ?? []).map((i) => ({ panel: 'canvas', id: `p${i}`, intensity: 0.6 })),
    ],
    metrics: {
      step: k, n: r.points.length, dataSeed: r.dataSeed,
      clusterCount, mergeHeight: m.height, maxHeight: m.height,
    },
    events: [
      ...(first ? [{ type: 'init' as const, label: 'agglomerative-hierarchical', step: 1 }] : []),
      { type: 'merge' as const, label: `merge-${k}`, step: k },
    ],
    timeline: first ? ['Data', 'Distances', 'Merge 1'] : [`Merge ${k}`],
  };
}

function finalSnapshot(r: AggloResult, p: Params): SimState {
  const k = r.merges.length;
  const cutHeight = (p.cutHeight as number) ?? 1.2;
  const cutClusters = clusterCountAtHeight(r, cutHeight);
  const med = medianHeight(r);
  const medClusters = clusterCountAtHeight(r, med);
  const q75 = 0.75 * r.maxHeight;
  const q75Clusters = clusterCountAtHeight(r, q75);
  const cutPartition = clustersAtHeight(r, cutHeight);
  const last = r.merges[k - 1];
  const narration =
    `Full dendrogram: ${r.points.length} leaves, ${k} merges, max height ${r.maxHeight.toFixed(3)} ` +
    `(${LINKAGE_NAME[r.linkage]}). Cophenetic correlation = ${r.copheneticCorr.toFixed(3)} — Pearson ` +
    `correlation between every pair's distance and its first-join height; a value close to 1 means the ` +
    `dendrogram preserves the distance ordering. Cutting at cutHeight = ${cutHeight.toFixed(2)} ` +
    `(${cutPartition.length} clusters); at the median merge height ${med.toFixed(2)} → ${medClusters} clusters; ` +
    `at 75% of max (${q75.toFixed(2)}) → ${q75Clusters} clusters.`;
  return {
    algorithm: {
      ...baseAlgorithm(r, k),
      copheneticCorr: r.copheneticCorr,
      cutClusters,
    } as Record<string, ParamValue>,
    visuals: [
      ...scatterCommands(r, cutPartition),
      ...mergeCommands(r, k),
      ...matrixCommands(r, true),
    ],
    math: [
      { latex: 'c_{ij} = \\text{height of the first merge joining } i \\text{ and } j', id: 'hc-cophenetic' },
      { latex: '\\text{cophenetic corr} = \\operatorname{corr}\\big(\\{d_{ij}\\}, \\{c_{ij}\\}\\big)', id: 'hc-cophenetic-corr' },
      { latex: '\\text{cut at } h \\Rightarrow \\text{clusters} = \\{\\text{merges with height} \\le h\\}', id: 'hc-cut' },
    ],
    narration,
    explanation: {
      changed: [
        `final merge: ${describeCluster(last.membersA ?? [])} ∪ ${describeCluster(last.membersB ?? [])} at ${last.height.toFixed(3)}`,
        `cophenetic correlation → ${r.copheneticCorr.toFixed(3)}`,
        `cut at ${cutHeight.toFixed(2)} → ${cutClusters} clusters (colored in the scatter)`,
      ],
      why: `The tree is complete. The cophenetic matrix C (shown in the matrix-animator) records, for every pair of ` +
        `points, the height at which they first joined the same cluster; the cophenetic correlation is the Pearson ` +
        `correlation between C and the original distance matrix D — how faithfully the tree reproduces the distances. ` +
        `The cut slider picks a height h: every merge above h is undone, leaving the clusters visible in the scatter ` +
        `(merges with height ≤ h form the clusters). Higher cuts → fewer, bigger clusters.`,
      formulaRef: 'hc-cophenetic-corr',
      dependsOn: ['probability', 'statistics', 'linear-algebra'],
      gateConcepts: ['hierarchical clustering', 'cophenetic correlation', 'dendrogram cut', 'linkage'],
    },
    highlights: [
      { panel: 'canvas', id: 'p0', intensity: 0.5 },
      { panel: 'matrix', id: 'C: cophenetic distances:0,0', intensity: 1 },
      { panel: 'equation', id: 'hc-cophenetic-corr', intensity: 1 },
    ],
    metrics: {
      step: k, n: r.points.length, dataSeed: r.dataSeed,
      clusterCount: 1, mergeHeight: last.height, maxHeight: r.maxHeight,
      copheneticCorr: r.copheneticCorr,
      cutClusters, medianCutClusters: medClusters, q75CutClusters: q75Clusters,
    },
    events: [
      { type: 'merge', label: `merge-${k}`, step: k },
      { type: 'converged', label: 'agglomerative-complete', step: k },
    ],
    timeline: [`Merge ${k}`, 'Cut height', 'Cophenetic'],
  };
}

// ---------------------------------------------------------------------------
// Simulation — one snapshot per merge; terminates after n−1 merges
// ---------------------------------------------------------------------------

export const simulation: SimulationDef = {
  initialState: (p: Params): SimState => {
    const r = computeResult(p);
    return mergeSnapshot(r, 1, true, p);
  },
  step: (p: Params, s: SimState): SimState | null => {
    const r = computeResult(p);
    const current = (s.algorithm.step as number) ?? 1;
    const next = current + 1;
    if (next < r.merges.length) return mergeSnapshot(r, next, false, p);
    if (next === r.merges.length) return finalSnapshot(r, p);
    return null;
  },
};

// ---------------------------------------------------------------------------
// Topic module + registration
// ---------------------------------------------------------------------------

export const hierarchicalClusteringModule: TopicModule = {
  id: 'hierarchical-clustering',
  title: 'Hierarchical Clustering (Agglomerative)',
  version: 1,
  metadata: {
    gateWeightage: 'Medium',
    difficultyHeatmap: { conceptual: 4, mathematical: 3, coding: 3, visualization: 5, gateFrequency: 4 },
    estimatedHours: 5,
    revisionPriority: 'P1',
    examFrequency: 'Frequent',
    prerequisites: ['probability', 'statistics', 'linear-algebra'],
    relatedTopics: ['kmeans', 'naive-bayes', 'svm-soft-margin'],
    revision: { quick: '15m', standard: '45m', deep: '1h30m', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'dendrogram', title: 'Dendrogram — the merge tree grows one merge at a time (click a node to highlight its subtree)' },
      { slot: 'primary', component: 'scatter-plot', title: 'Points colored by their CURRENT cluster at each merge; the final snapshot colors the cut-height partition' },
      { slot: 'sidebar', component: 'matrix-animator', title: 'Distance matrix D (numeric grid; static input) — final snapshot adds the cophenetic matrix C' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivations: Linkage Criteria & Ward SSE Increase' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Data → Distances → Merge 1…n−1 → Cut → Cophenetic' },
      { slot: 'sidebar', component: 'explain-step', title: 'Step Explanation' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'n', label: 'Points n', type: 'number', min: 4, max: 20, step: 1, default: 12 },
    {
      id: 'linkage', label: 'Linkage criterion', type: 'select',
      options: [
        { value: 'single', label: 'Single (min pair)' },
        { value: 'complete', label: 'Complete (max pair)' },
        { value: 'average', label: 'Average (mean pair)' },
        { value: 'ward', label: 'Ward (SSE increase)' },
      ],
      default: 'single',
    },
    {
      id: 'blobs', label: 'Gaussian blobs', type: 'select',
      options: [
        { value: '2', label: '2 blobs' },
        { value: '3', label: '3 blobs' },
      ],
      default: '2',
    },
    { id: 'cutHeight', label: 'Cut height h', type: 'number', min: 0, max: 8, step: 0.1, default: 1.2 },
    { id: 'seed', label: 'Seed', type: 'seed', min: 0, max: 9999, step: 1, default: 42 },
  ],
  simulation,
  formulas: hierarchicalFormulas,
  derivations: hierarchicalDerivations,
  questions: hierarchicalQuestions,
  comparisons: hierarchicalComparisons,
  failureDemos: hierarchicalFailureDemos,
  mistakes: hierarchicalMistakes,
  testCases: hierarchicalTestCases,
  // NOTE: lossMetricKey intentionally omitted — copheneticCorr is only defined
  // once the tree is complete, so no per-snapshot loss curve (naive-bayes
  // precedent; documented in the module header).
  // lossMetricKey: undefined

  validateParams: (p) => {
    const issues: string[] = [];
    const n = p.n as number | undefined;
    if (n !== undefined) {
      if (!Number.isInteger(n) || n < 4 || n > 20) {
        issues.push('n must be an integer in [4, 20] — below 4 the dendrogram is trivial (≤ 3 merges); above 20 the O(n²) distance matrix + naive O(n³) linkage recomputation gets sluggish and the dendrogram becomes unreadable');
      }
    }
    const linkage = p.linkage as string | undefined;
    if (linkage !== undefined && !LINKAGES.includes(linkage as Linkage)) {
      issues.push(`linkage must be one of ${LINKAGES.join(', ')}`);
    }
    const blobs = p.blobs as string | number | undefined;
    if (blobs !== undefined && blobs !== '2' && blobs !== '3' && blobs !== 2 && blobs !== 3) {
      issues.push('blobs must be 2 or 3');
    }
    const cut = p.cutHeight as number | undefined;
    if (cut !== undefined && (!Number.isFinite(cut) || cut < 0)) {
      issues.push('cutHeight must be a finite number ≥ 0 (values above the max merge height simply leave one cluster)');
    }
    const seed = p.seed as number | undefined;
    if (seed !== undefined && (!Number.isInteger(seed) || seed < 0 || seed > 9999)) {
      issues.push('seed must be an integer in [0, 9999]');
    }
    if (typeof p.points === 'string') {
      try {
        const rows = JSON.parse(p.points) as unknown;
        if (
          !Array.isArray(rows) || rows.length < 4 || rows.length > 20 ||
          !rows.every((r) => Array.isArray(r) && r.length === 2 && r.every((v) => typeof v === 'number' && Number.isFinite(v)))
        ) {
          issues.push('points must be a JSON array of 4–20 [x, y] pairs');
        }
      } catch {
        issues.push('points must be a valid JSON array of [x, y] pairs');
      }
    }
    return issues;
  },
};

export function register() {
  registerTopic(hierarchicalClusteringModule);
  // Unsupervised — no classifier to register (contrast with lda/svm).
}
