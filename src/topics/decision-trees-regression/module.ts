// src/topics/decision-trees-regression/module.ts
// Task 21 (Wave 5): decision-trees-regression — CART (Classification And
// Regression Trees) applied to 1-D REGRESSION: greedy binary splits on a noisy
// smooth curve, piecewise-constant (step-function) predictions, and the
// train-vs-test depth overfitting story.
//
// Design decisions (documented drift from the plan):
//  - Truth model: y = A·sin(ω·x) + ε, A = 1.5, ω = 1.6, x ~ U[−3, 3],
//    ε ~ N(0, noise²), ε drawn from a seeded mulberry32 stream via Box–Muller.
//    A SMOOTH truth was chosen over a step truth on purpose: the depth sweep
//    then shows the classic bias → good fit → variance arc (depth 1 = a single
//    horizontal constant line with big train error; mid depth hugs the sine;
//    deep trees memorize noise). A step truth would be "solved" by the first
//    split and the overfit curve would flatten — a worse lesson.
//  - Tree growth is BEST-FIRST, not CART's depth-first recursion: at every
//    step CART splits the current leaf with the LARGEST SSE reduction
//    (ties → shallower depth, then earlier-created id). The split-candidate
//    math is identical (sorted midpoints, SSE reduction); the animation is
//    simply more legible ("the algorithm picks the split that pays off most").
//    Documented drift from the plan's "one snapshot per split" (it does not
//    mandate an order).
//  - Gini impurity is taught as the CLASSIFICATION analog in formulas,
//    questions and mistakes (per the plan); the simulation itself is pure
//    REGRESSION (SSE objective). giniImpurity is exposed as a pure exported
//    helper and pinned by test case 1 — the plan explicitly mixes the two
//    lenses, and this module documents which is which.
//  - Train/test split: the first 70% of the generated samples (rounded down)
//    are train, the rest held-out test — deterministic, documented. The
//    test-only `xys` override ('[[x,y],…]' JSON, failure demos + tests) makes
//    the ENTIRE dataset the training split, so toy tests and failure demos
//    are exact (testError is in-sample there — documented in the narration).
//  - The fitted step function is drawn in scatter-plot as DENSE POINT commands
//    (161 samples of predict over the padded x range — the plan-mandated
//    rendering); the truth curve is a faint reference line. Step predictions
//    are NOT clamped outside the observed range — the flat tails ARE the
//    extrapolation failure (failure demo 1).
//  - mulberry32 / Box–Muller copied verbatim from src/topics/pca/module.ts
//    (Wave-0 pattern: every topic module is self-contained; no shared rng
//    import). giniImpurity, meanOf, sseOf, bestSplitCandidate, growTree,
//    predict and generateData are all pure + exported for hand-verified tests.
import type { TopicModule, Params, SimState, VisualCommand, ParamValue } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { dtrTestCases } from './testCases';
import { dtrFormulas } from './formulas';
import { dtrDerivations } from './derivations';
import { dtrMistakes } from './mistakes';
import { dtrQuestions } from './questions';
import { dtrComparisons } from './comparisons';
import { dtrFailureDemos } from './failures';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const XMIN = -3;
export const XMAX = 3;
export const TRUTH_A = 1.5;   // amplitude of the sine truth
export const TRUTH_W = 1.6;   // angular frequency of the sine truth
export const TRAIN_FRACTION = 0.7; // first 70% of samples = training split

// Visual-semantic colors.
export const TRAIN_COLOR = '#2563eb';  // training points (blue)
export const TEST_COLOR = '#f97316';   // held-out test points (orange)
export const FIT_COLOR = '#dc2626';    // fitted step function (red)
export const TRUTH_COLOR = '#94a3b8';  // underlying truth curve (faint)
export const NODE_INTERNAL_COLOR = '#3b82f6';
export const NODE_LEAF_COLOR = '#16a34a';

// ---------------------------------------------------------------------------
// Deterministic PRNG + data synthesis
// ---------------------------------------------------------------------------

/** Mulberry32 — deterministic PRNG. Copied verbatim from pca/module.ts (Wave-0 self-contained-topic pattern). */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller standard normal (spare-draw trick). Copied from pca/module.ts. */
export function makeNormal(rng: () => number): () => number {
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

/** Truth signal of the synthetic data: y = A·sin(ω·x). */
export function truthY(x: number): number {
  return TRUTH_A * Math.sin(TRUTH_W * x);
}

export interface DtrData {
  xs: number[];
  ys: number[];
  nTrain: number;     // first nTrain samples are train; the rest held-out test
  hasTruth: boolean;  // false when the xys override replaced the sine truth
  xMin: number;       // observed x bounds (for scatter padding + sampling)
  xMax: number;
}

/**
 * Deterministic data synthesis: x ~ U[−3, 3], y = A·sin(ω·x) + noise·N(0,1).
 * Test-only `xys` override (JSON '[[x,y],…]'): uses exactly those points and
 * sets nTrain = ALL of them (the whole dataset is the training split, so toy
 * tests and failure demos are exact).
 */
export function generateData(p: Params): DtrData {
  if (typeof p.xys === 'string') {
    const rows: [number, number][] = JSON.parse(p.xys) as [number, number][];
    const xs = rows.map((r) => r[0]);
    const ys = rows.map((r) => r[1]);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    return { xs, ys, nTrain: rows.length, hasTruth: false, xMin, xMax };
  }
  const n = (p.n as number) ?? 30;
  const noise = (p.noise as number) ?? 0.4;
  const seed = (p.seed as number) ?? 42;
  const rng = mulberry32(seed);
  const normal = makeNormal(rng);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = XMIN + rng() * (XMAX - XMIN);
    xs.push(x);
    ys.push(truthY(x) + (noise > 0 ? noise * normal() : 0));
  }
  const nTrain = Math.max(2, Math.floor(TRAIN_FRACTION * n));
  return { xs, ys, nTrain, hasTruth: true, xMin: XMIN, xMax: XMAX };
}

// ---------------------------------------------------------------------------
// Pure math helpers (exported for hand-verified tests)
// ---------------------------------------------------------------------------

/** Gini impurity of a class-count vector: G = 1 − Σ(p_k²). Returns 0 for empty input. */
export function giniImpurity(counts: number[]): number {
  const total = counts.reduce((a, b) => a + b, 0);
  if (total <= 0) return 0;
  return 1 - counts.reduce((a, c) => a + (c / total) ** 2, 0);
}

/** Arithmetic mean; 0 for an empty list. */
export function meanOf(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** Sum of squared deviations from the mean: SSE = Σ(y − ȳ)². 0 for empty. */
export function sseOf(values: number[]): number {
  const m = meanOf(values);
  return values.reduce((a, y) => a + (y - m) ** 2, 0);
}

// ---------------------------------------------------------------------------
// CART core (all mutation-free; the tree is rebuilt per split-count)
// ---------------------------------------------------------------------------

export interface SplitCandidate {
  threshold: number;   // (x_i + x_{i+1})/2 midpoint of sorted distinct x
  sseAfter: number;    // SSE_left + SSE_right
  reduction: number;   // SSE(node) − sseAfter
  sseLeft: number;
  sseRight: number;
  nLeft: number;
  nRight: number;
}

export interface TreeNode {
  id: string;
  depth: number;           // root = 0
  indices: number[];       // sample indices (into data.xs/data.ys) in this region
  value: number;           // mean of y in indices — the leaf prediction
  sse: number;             // SSE of y in this region
  threshold?: number;      // internal nodes: the chosen split point
  children: string[];      // [leftId, rightId]
  parentId: string | null;
  sseReduction?: number;   // SSE reduction gained by this node's split
  sseLeft?: number;        // SSE of the left child region (post-split)
  sseRight?: number;       // SSE of the right child region (post-split)
  nLeft?: number;
  nRight?: number;
}

/**
 * Exhaustive best split over ALL midpoints between sorted DISTINCT x values
 * in the region: the candidate with the largest SSE reduction wins; ties go to
 * the SMALLEST threshold (deterministic). Returns null when the region has
 * < 2 distinct x values (no valid midpoint exists).
 */
export function bestSplitCandidate(data: DtrData, indices: number[]): SplitCandidate | null {
  const uniq: number[] = [];
  for (const i of indices.slice().sort((a, b) => data.xs[a] - data.xs[b])) {
    const x = data.xs[i];
    if (uniq.length === 0 || Math.abs(x - uniq[uniq.length - 1]) > 1e-12) uniq.push(x);
  }
  if (uniq.length < 2) return null;
  const before = sseOf(indices.map((i) => data.ys[i]));
  let best: SplitCandidate | null = null;
  for (let k = 0; k < uniq.length - 1; k++) {
    const t = (uniq[k] + uniq[k + 1]) / 2;
    const left = indices.filter((i) => data.xs[i] < t);
    const right = indices.filter((i) => data.xs[i] >= t);
    const sseL = sseOf(left.map((i) => data.ys[i]));
    const sseR = sseOf(right.map((i) => data.ys[i]));
    const sseAfter = sseL + sseR;
    const reduction = before - sseAfter;
    if (!best || reduction > best.reduction + 1e-12 ||
      (Math.abs(reduction - best.reduction) <= 1e-12 && t < best.threshold)) {
      best = { threshold: t, sseAfter, reduction, sseLeft: sseL, sseRight: sseR, nLeft: left.length, nRight: right.length };
    }
  }
  return best;
}

/** Best split of a leaf, respecting maxDepth + minLeaf; null when it must not split. */
export function bestSplitForLeaf(data: DtrData, maxDepth: number, minLeaf: number, leaf: TreeNode): SplitCandidate | null {
  if (leaf.depth >= maxDepth) return null;
  if (leaf.indices.length < 2 * minLeaf) return null;
  const cand = bestSplitCandidate(data, leaf.indices);
  if (!cand || cand.reduction <= 1e-12) return null; // no useful split (constant y etc.)
  return cand;
}

export interface GrownTree {
  nodes: TreeNode[];
  lastSplitId: string | null;  // node id that became internal at the LAST growth step (null at k=0)
  lastReduction: number;       // SSE reduction of that split (0 at k=0)
  halted: boolean;             // true when growth could not add another split
}

/**
 * Greedily grow the tree `splits` times (or until no leaf is splittable —
 * then `halted` is true). Best-first: at each step split the leaf with the
 * largest SSE reduction; ties → shallower depth, then earlier-created id.
 */
export function growTree(data: DtrData, maxDepth: number, minLeaf: number, splits: number): GrownTree {
  const byId = new Map<string, TreeNode>();
  const trainIdx = Array.from({ length: data.nTrain }, (_, i) => i);
  const mk = (id: string, depth: number, indices: number[], parentId: string | null): TreeNode => {
    const ys = indices.map((i) => data.ys[i]);
    return { id, depth, indices, parentId, value: meanOf(ys), sse: sseOf(ys), children: [] };
  };
  const root = mk('n0', 0, trainIdx, null);
  byId.set('n0', root);
  let nextId = 1;
  let lastSplitId: string | null = null;
  let lastReduction = 0;
  for (let k = 0; k < splits; k++) {
    let bestNode: TreeNode | null = null;
    let bestCand: SplitCandidate | null = null;
    for (const leaf of byId.values()) {
      if (leaf.children.length > 0) continue;
      const cand = bestSplitForLeaf(data, maxDepth, minLeaf, leaf);
      if (!cand) continue;
      if (!bestNode || cand.reduction > bestCand!.reduction + 1e-12 ||
        (Math.abs(cand.reduction - bestCand!.reduction) <= 1e-12 &&
          (leaf.depth < bestNode.depth || (leaf.depth === bestNode.depth && leaf.id < bestNode.id)))) {
        bestNode = leaf;
        bestCand = cand;
      }
    }
    if (!bestNode) return { nodes: [...byId.values()], lastSplitId, lastReduction, halted: true };
    const lIdx = bestNode.indices.filter((i) => data.xs[i] < bestCand!.threshold);
    const rIdx = bestNode.indices.filter((i) => data.xs[i] >= bestCand!.threshold);
    const left = mk(`n${nextId++}`, bestNode.depth + 1, lIdx, bestNode.id);
    const right = mk(`n${nextId++}`, bestNode.depth + 1, rIdx, bestNode.id);
    bestNode.threshold = bestCand!.threshold;
    bestNode.children = [left.id, right.id];
    bestNode.sseReduction = bestCand!.reduction;
    bestNode.sseLeft = bestCand!.sseLeft;
    bestNode.sseRight = bestCand!.sseRight;
    bestNode.nLeft = bestCand!.nLeft;
    bestNode.nRight = bestCand!.nRight;
    byId.set(left.id, left);
    byId.set(right.id, right);
    lastSplitId = bestNode.id;
    lastReduction = bestCand!.reduction;
  }
  return { nodes: [...byId.values()], lastSplitId, lastReduction, halted: false };
}

/** Piecewise-constant prediction of the tree at x (NO clamping — constant extrapolation is the honest failure). */
export function predict(nodes: TreeNode[], x: number): number {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  let cur = byId.get('n0')!;
  while (cur.children.length > 0) {
    cur = byId.get(x < cur.threshold! ? cur.children[0] : cur.children[1])!;
  }
  return cur.value;
}

/** True when at least one leaf can still gain a useful split (the run's final check). */
export function hasSplittableLeaf(data: DtrData, nodes: TreeNode[], maxDepth: number, minLeaf: number): boolean {
  return nodes.some((n) => n.children.length === 0 && bestSplitForLeaf(data, maxDepth, minLeaf, n) !== null);
}

// ---------------------------------------------------------------------------
// Bounded memoization (the pca precedent): initialState/step stay O(1) after
// the first evaluation of a (params, splitCount) key.
// ---------------------------------------------------------------------------

const DATA_CACHE = new Map<string, DtrData>();
const DATA_CACHE_MAX = 16;
const TREE_CACHE = new Map<string, GrownTree>();
const TREE_CACHE_MAX = 64;

function dataKey(p: Params): string {
  return JSON.stringify([p.n ?? null, p.noise ?? null, p.seed ?? null, p.xys ?? null]);
}

export function cachedData(p: Params): DtrData {
  const key = dataKey(p);
  let d = DATA_CACHE.get(key);
  if (!d) {
    d = generateData(p);
    if (DATA_CACHE.size >= DATA_CACHE_MAX) DATA_CACHE.clear();
    DATA_CACHE.set(key, d);
  }
  return d;
}

function treeKey(p: Params, splits: number): string {
  return `${dataKey(p)}|${p.maxDepth ?? 4}|${p.minLeaf ?? 2}|${splits}`;
}

export function cachedTree(p: Params, data: DtrData, splits: number): GrownTree {
  const key = treeKey(p, splits);
  let t = TREE_CACHE.get(key);
  if (!t) {
    t = growTree(data, (p.maxDepth as number) ?? 4, (p.minLeaf as number) ?? 2, splits);
    if (TREE_CACHE.size >= TREE_CACHE_MAX) TREE_CACHE.clear();
    TREE_CACHE.set(key, t);
  }
  return t;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

function metricsOf(data: DtrData, grown: GrownTree, splitCount: number): Record<string, number> {
  const { nodes } = grown;
  const leaves = nodes.filter((n) => n.children.length === 0);
  const sse = leaves.reduce((a, n) => a + n.sse, 0);
  const ssTot = sseOf(data.ys.slice(0, data.nTrain));
  const r2 = ssTot < 1e-12 ? 1 : 1 - sse / ssTot;
  const trainError = sse / data.nTrain;
  let testError = trainError; // xys override (whole dataset is train) → in-sample
  if (data.ys.length > data.nTrain) {
    let err = 0;
    for (let i = data.nTrain; i < data.ys.length; i++) {
      err += (predict(nodes, data.xs[i]) - data.ys[i]) ** 2;
    }
    testError = err / (data.ys.length - data.nTrain);
  }
  const root = nodes.find((n) => n.parentId === null)!;
  return {
    step: splitCount + 1,
    sse,
    trainError,
    testError,
    r2,
    nLeaves: leaves.length,
    nSplits: splitCount,
    depth: Math.max(...nodes.map((n) => n.depth)),
    nTrain: data.nTrain,
    nTest: data.ys.length - data.nTrain,
    lastReduction: grown.lastReduction,
    rootThreshold: root.threshold ?? -1,   // -1 = the root never split
    rootSse: root.children.length > 0 ? (root.sseLeft ?? 0) + (root.sseRight ?? 0) : -1,
    rootReduction: root.sseReduction ?? 0,
    rootNLeft: root.nLeft ?? 0,
    rootNRight: root.nRight ?? 0,
  };
}

// ---------------------------------------------------------------------------
// Visuals
// ---------------------------------------------------------------------------

/** Scatter: train pts, test pts, faint truth line, and the fitted step function as DENSE POINT commands. */
function buildScatter(data: DtrData, nodes: TreeNode[]): VisualCommand[] {
  const cmds: VisualCommand[] = [];
  for (let i = 0; i < data.nTrain; i++) {
    cmds.push({ type: 'point', id: `tr${i}`, x: data.xs[i], y: data.ys[i], color: TRAIN_COLOR });
  }
  for (let i = data.nTrain; i < data.ys.length; i++) {
    cmds.push({ type: 'point', id: `te${i}`, x: data.xs[i], y: data.ys[i], color: TEST_COLOR });
  }
  if (data.hasTruth) {
    const pts: [number, number][] = [];
    for (let i = 0; i <= 60; i++) {
      const x = data.xMin + ((data.xMax - data.xMin) * i) / 60;
      pts.push([x, truthY(x)]);
    }
    cmds.push({ type: 'line', id: 'truth-curve', points: pts, color: TRUTH_COLOR });
  }
  // Fitted step function: plan-mandated dense POINT commands over the padded
  // x range (flat constant tails beyond the data are intentional — the
  // extrapolation failure is visible in the same view).
  const SAMPLES = 161;
  const pad = (data.xMax - data.xMin) * 0.1;
  const x0 = data.xMin - pad;
  const x1 = data.xMax + pad;
  for (let i = 0; i < SAMPLES; i++) {
    const x = x0 + ((x1 - x0) * i) / (SAMPLES - 1);
    cmds.push({ type: 'point', id: `fit${i}`, x, y: predict(nodes, x), color: FIT_COLOR });
  }
  return cmds;
}

function yOf(depth: number, maxDepth: number): number {
  return (depth + 1) / (maxDepth + 1);
}

/**
 * tree-builder node commands: leaves evenly spaced left-to-right by in-order
 * traversal, internal nodes centered over their children; y by depth. purity =
 * 1 − SSE(node)/SSE(root) (fraction of total variance explained by the
 * partition reaching this node) — matches the bar's "0 = impure, 1 = pure"
 * semantics for classification trees.
 */
function buildTreeView(nodes: TreeNode[], maxDepth: number): VisualCommand[] {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const root = nodes.find((n) => n.parentId === null)!;
  const rootSse = Math.max(root.sse, 1e-12);
  const leaves: string[] = [];
  const inorder = (id: string): void => {
    const n = byId.get(id)!;
    if (n.children.length === 0) { leaves.push(id); return; }
    inorder(n.children[0]);
    inorder(n.children[1]);
  };
  inorder(root.id);
  const pos = new Map<string, { x: number; y: number }>();
  leaves.forEach((id, i) => {
    pos.set(id, { x: (i + 0.5) / leaves.length, y: yOf(byId.get(id)!.depth, maxDepth) });
  });
  for (const n of nodes.filter((nd) => nd.children.length > 0).sort((a, b) => b.depth - a.depth)) {
    const cx = n.children.map((c) => pos.get(c)!.x);
    pos.set(n.id, { x: (cx[0] + cx[1]) / 2, y: yOf(n.depth, maxDepth) });
  }
  return nodes.map((n) => {
    const p = pos.get(n.id)!;
    const internal = n.children.length > 0;
    return {
      type: 'node',
      id: n.id,
      x: p.x,
      y: p.y,
      label: internal ? `x < ${n.threshold!.toFixed(2)}` : `\u0177 = ${n.value.toFixed(2)}`,
      splitInfo: internal
        ? `SSE \u2193 ${n.sseReduction!.toFixed(2)} (${n.nLeft}/${n.nRight})`
        : `n=${n.indices.length} \u00b7 \u0177=${n.value.toFixed(2)}`,
      children: n.children,
      purity: 1 - n.sse / rootSse,
      color: internal ? NODE_INTERNAL_COLOR : NODE_LEAF_COLOR,
    } as VisualCommand;
  });
}

/** matrix-animator story: the CART split-decision ledger (node, threshold, SSE reduction, partition sizes). */
function buildMatrices(grown: GrownTree): VisualCommand[] {
  const internals = grown.nodes.filter((n) => n.children.length > 0);
  if (internals.length === 0) return []; // no splits yet — nothing to show
  const cells: (number | string)[][] = [
    ['node', 'depth', 't', '\u0394SSE', 'nL', 'nR'],
    ...internals.map((n) => [
      Number(n.id.slice(1)), n.depth, n.threshold ?? 0, n.sseReduction ?? 0, n.nLeft ?? 0, n.nRight ?? 0,
    ]),
  ];
  return [{
    type: 'matrix', id: 'CART splits', rows: cells.length, cols: 6, cells,
  } as VisualCommand];
}

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

function snapshot(p: Params, data: DtrData, grown: GrownTree, splitCount: number): SimState {
  const m = metricsOf(data, grown, splitCount);
  const maxDepth = (p.maxDepth as number) ?? 4;
  const final = !hasSplittableLeaf(data, grown.nodes, maxDepth, (p.minLeaf as number) ?? 2);
  const { nodes } = grown;
  const lastNode = grown.lastSplitId ? nodes.find((n) => n.id === grown.lastSplitId) : null;
  const bestRemaining = (() => {
    let best = 0;
    for (const n of nodes) {
      if (n.children.length > 0) continue;
      const c = bestSplitForLeaf(data, maxDepth, (p.minLeaf as number) ?? 2, n);
      if (c && c.reduction > best) best = c.reduction;
    }
    return best;
  })();

  let narration: string;
  let why: string;
  const changed: string[] = [];
  const events: SimState['events'] = [{ type: 'init', label: 'cart-root-leaf', step: 1 }];
  const timeline: string[] = ['Data', 'Root'];

  if (splitCount === 0) {
    const root = nodes.find((n) => n.parentId === null)!;
    narration =
      `Root leaf only (depth 0): the whole training set (n = ${data.nTrain}) is a single region predicting the ` +
      `constant \u0177 = ${root.value.toFixed(3)} — train SSE = ${m.sse.toFixed(3)} (train error ${m.trainError.toFixed(3)}, ` +
      `test error ${m.testError.toFixed(3)}). CART now scans every midpoint between the sorted x values and picks ` +
      `the split with the largest SSE reduction.`;
    why =
      `No split yet. A depth-0 regression tree is just the global mean — the simplest possible piecewise-constant ` +
      `model. CART's next move is greedy: evaluate EVERY midpoint (x_i + x_{i+1})/2 between consecutive distinct ` +
      `x values, partition the samples, and keep the split that most reduces SSE = \u03a3(y \u2212 leaf-mean)\u00b2.`;
    changed.push(`root leaf: \u0177 = ${root.value.toFixed(3)}`, `train SSE = ${m.sse.toFixed(3)}`);
  } else {
    events.push({ type: 'split', label: 'cart-split', step: splitCount + 1 });
    timeline.push('Split', 'Evaluate');
    if (final) events.push({ type: 'converged', label: 'cart-tree-complete', step: splitCount + 1 });
    if (lastNode) {
      const sseAfter = lastNode.sse - (lastNode.sseReduction ?? 0);
      changed.push(
        `split #${splitCount}: ${lastNode.id} at x < ${lastNode.threshold!.toFixed(2)}`,
        `SSE \u2193 ${lastNode.sseReduction!.toFixed(3)} (${lastNode.sse.toFixed(3)} \u2192 ${sseAfter.toFixed(3)})`,
        `leaves \u2192 ${m.nLeaves} (depth ${m.depth})`,
        `train error \u2192 ${m.trainError.toFixed(3)}`,
        `test error \u2192 ${m.testError.toFixed(3)}`,
      );
    }
    narration =
      `Split #${splitCount}: the highest-paying leaf ${lastNode!.id} splits at x < ${lastNode!.threshold!.toFixed(2)} — ` +
      `SSE reduction ${grown.lastReduction.toFixed(3)} (node SSE ${lastNode!.sse.toFixed(3)} \u2192 ${(lastNode!.sse - grown.lastReduction).toFixed(3)}); ` +
      `the tree now has ${m.nLeaves} leaves at depth ${m.depth}. ` +
      `train error ${m.trainError.toFixed(3)}, test error ${m.testError.toFixed(3)} (n = ${data.nTrain}/${data.ys.length - data.nTrain} train/test). ` +
      (bestRemaining > 1e-9
        ? `Greedy CART repeats: the next best split would gain \u0394SSE = ${bestRemaining.toFixed(3)}.`
        : `No leaf can reduce SSE further — the tree is complete.`);
    why =
      `CART picks the leaf whose split gives the LARGEST SSE reduction \u0394 = SSE(node) \u2212 SSE(left) \u2212 SSE(right) ` +
      `(${grown.lastReduction.toFixed(3)} for this split). Each split replaces one constant region with two — the fit ` +
      `becomes a finer step function. Watch the loss curve: train error falls every split; test error bottoms out then ` +
      `rises once the steps start memorizing noise (variance).`;
  }
  if (final) {
    timeline.push('Complete');
    changed.push('tree complete — no splittable leaf remains');
  }

  const algorithm: Record<string, ParamValue> = {
    mode: 'cart-regression',
    splitCount,
    nLeaves: m.nLeaves,
    depth: m.depth,
    dataSeed: (p.seed as number) ?? 42,
    lastSplitId: grown.lastSplitId ?? 'none',
  };

  const math = splitCount === 0
    ? [
        { latex: '\\hat{y}(S) = \\frac{1}{n_S} \\sum_{i \\in S} y_i', id: 'leaf-mean' },
        { latex: 'SSE(S) = \\sum_{i \\in S} (y_i - \\hat{y}(S))^2', id: 'sse-node' },
        { latex: '\\Delta SSE = SSE(S) - SSE(S_L) - SSE(S_R)', id: 'sse-reduction' },
      ]
    : [
        { latex: `x < ${lastNode!.threshold!.toFixed(2)} \\Rightarrow \\Delta SSE = ${grown.lastReduction.toFixed(3)}`, id: 'sse-reduction' },
        { latex: '\\hat{y}(S) = \\frac{1}{n_S} \\sum_{i \\in S} y_i', id: 'leaf-mean' },
      ];

  return {
    algorithm,
    visuals: [
      ...buildScatter(data, nodes),
      ...buildTreeView(nodes, maxDepth),
      ...buildMatrices(grown),
    ],
    math,
    narration,
    explanation: {
      changed,
      why,
      formulaRef: splitCount === 0 ? 'leaf-mean' : 'sse-reduction',
      dependsOn: ['statistics', 'variance', 'greedy-algorithms'],
      gateConcepts: ['CART', 'regression tree', 'SSE', 'greedy splitting', 'overfitting'],
    },
    highlights: grown.lastSplitId
      ? [{ panel: 'canvas', id: grown.lastSplitId, intensity: 1 }]
      : [],
    metrics: m,
    events,
    timeline,
  };
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

export const simulation = {
  /** Snapshot 1 = the depth-0 tree (root leaf only). */
  initialState: (p: Params): SimState => {
    const data = cachedData(p);
    const grown = cachedTree(p, data, 0);
    return snapshot(p, data, grown, 0);
  },

  /** Grow the tree one split; null once no leaf can split further. */
  step: (p: Params, s: SimState): SimState | null => {
    const data = cachedData(p);
    const current = (s.algorithm.splitCount as number) ?? 0;
    const next = current + 1;
    const grown = cachedTree(p, data, next);
    if (grown.halted) return null; // no new split was applied at this step
    return snapshot(p, data, grown, next);
  },
};

// ---------------------------------------------------------------------------
// Topic module + registration
// ---------------------------------------------------------------------------

export const dtrModule: TopicModule = {
  id: 'decision-trees-regression',
  title: 'Regression Trees (CART)',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 3, coding: 3, visualization: 3, gateFrequency: 4 },
    estimatedHours: 5,
    revisionPriority: 'P1',
    examFrequency: 'Frequent',
    prerequisites: ['statistics', 'variance', 'decision-trees'],
    relatedTopics: ['decision-trees', 'simple-linear-regression', 'knn', 'bias-variance'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'tree-builder', title: 'Tree Growth: Greedy SSE-Reduction Splits (bar = 1 − SSE/SSE_root)' },
      { slot: 'primary', component: 'scatter-plot', title: 'Data, Truth Curve & Fitted Step Function (red = tree prediction)' },
      { slot: 'primary', component: 'loss-curve', title: 'Train vs Test Error vs Splits — lower = better; divergence = overfitting' },
    ],
    core: [
      { slot: 'sidebar', component: 'matrix-animator', title: 'CART Split Ledger: threshold, SSE reduction, partition sizes' },
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
      { slot: 'primary', component: 'derivation-player', title: 'Derivations: SSE → variance, midpoints, leaf = mean' },
      { slot: 'primary', component: 'explain-step', title: 'Step Explanation' },
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: Data → Root → Split → Evaluate → Complete' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'n', label: 'Samples (n)', type: 'number', min: 10, max: 100, step: 1, default: 30 },
    { id: 'noise', label: 'Noise σ', type: 'number', min: 0, max: 2, step: 0.05, default: 0.4 },
    { id: 'maxDepth', label: 'Max depth', type: 'number', min: 1, max: 6, step: 1, default: 4 },
    { id: 'minLeaf', label: 'Min samples/leaf', type: 'number', min: 1, max: 8, step: 1, default: 2 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
  ],
  simulation,
  formulas: dtrFormulas,
  derivations: dtrDerivations,
  questions: dtrQuestions,
  comparisons: dtrComparisons,
  failureDemos: dtrFailureDemos,
  mistakes: dtrMistakes,
  testCases: dtrTestCases,
  lossMetricKey: 'trainError',
  lossMetricKey2: 'testError',

  validateParams: (p) => {
    const issues: string[] = [];
    const n = p.n as number | undefined;
    if (n !== undefined) {
      if (!Number.isInteger(n) || n < 5) {
        issues.push('n must be an integer ≥ 5 — with fewer than 5 samples the 70% train split leaves no room for a meaningful test split');
      }
      if (n > 100) issues.push('n > 100 exceeds the lightweight demo size (keep n ≤ 100 for smooth scrubbing)');
    }
    const noise = p.noise as number | undefined;
    if (noise !== undefined && !Number.isFinite(noise)) issues.push('noise must be a finite number');
    if (noise !== undefined && noise < 0) issues.push('noise must be ≥ 0');
    if (noise !== undefined && noise > 2) issues.push('noise > 2 swamps the sine truth — the tree only fits noise (the demo remains honest, but the signal is invisible)');
    const maxDepth = p.maxDepth as number | undefined;
    if (maxDepth !== undefined && (!Number.isInteger(maxDepth) || maxDepth < 1 || maxDepth > 6)) {
      issues.push('maxDepth must be an integer in [1, 6] — deeper trees overfit the noise (the demo topic)');
    }
    const minLeaf = p.minLeaf as number | undefined;
    if (minLeaf !== undefined && (!Number.isInteger(minLeaf) || minLeaf < 1)) {
      issues.push('minLeaf must be an integer ≥ 1');
    }
    const seed = p.seed as number | undefined;
    if (seed !== undefined && (!Number.isInteger(seed) || seed < 0 || seed > 9999)) issues.push('seed must be an integer in [0, 9999]');
    // Train split size drives the minLeaf guard (xys override → whole dataset is train).
    const nTrain = typeof p.xys === 'string'
      ? 0
      : Math.max(2, Math.floor(TRAIN_FRACTION * ((n as number) ?? 30)));
    if (typeof p.xys === 'string') {
      try {
        const rows = JSON.parse(p.xys) as unknown;
        if (!Array.isArray(rows) || rows.length < 1 || !rows.every((r) => Array.isArray(r) && r.length === 2 && r.every(Number.isFinite))) {
          issues.push('xys must be a JSON array of [x, y] pairs');
        } else {
          const xs = (rows as [number, number][]).map((r) => r[0]);
          const ys = (rows as [number, number][]).map((r) => r[1]);
          if (rows.length < 2) issues.push('WARNING: only one point — the tree can never split (degenerate)');
          else if (new Set(xs.map((x) => Math.round(x * 1e9))).size < 2) issues.push('WARNING: all x identical — no midpoint exists, the tree can never split');
          else if (ys.every((y) => Math.abs(y - ys[0]) < 1e-12)) issues.push('WARNING: all y equal — SSE reduction is 0 for every split, the tree will not split');
          if (minLeaf !== undefined && minLeaf * 2 > rows.length) {
            issues.push(`minLeaf = ${minLeaf} but 2·minLeaf = ${minLeaf * 2} > dataset size ${rows.length} — the root can never split`);
          }
        }
      } catch {
        issues.push('xys must be a valid JSON array of [x, y] pairs');
      }
    } else if (minLeaf !== undefined && minLeaf * 2 > nTrain) {
      issues.push(`minLeaf = ${minLeaf} but 2·minLeaf = ${minLeaf * 2} > nTrain = ${nTrain} — the root can never split (lower minLeaf or raise n)`);
    }
    return issues;
  },
};

export function register() {
  registerTopic(dtrModule);
  // Regression tree — no classifier to register (1-D regression, not a classifier).
}
