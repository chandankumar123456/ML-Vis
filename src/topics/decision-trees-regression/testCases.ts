// src/topics/decision-trees-regression/testCases.ts
// TestCase[] consumed by the centralized runner (src/test/runTestCases.test.ts)
// and by testCases.test.ts (computeRun + exact assertions). The required plan
// cases that are pure-helper math (gini impurity, leaf = mean, midpoint
// membership) live as direct assertions in testCases.test.ts — the computeRun
// cases below cover the run-level behaviors with an INDEPENDENT exhaustive
// midpoints check (inline SSE arithmetic here, no imports from module.ts).
import type { TestCase } from '../../engine/types';

/** Toy dataset A — two well-separated clusters (left high, right low). */
export const TOY_A: [number, number][] = [[-2, 1], [-1.2, 1.3], [0.1, -0.9], [1.5, -1.1], [2.2, -0.7]];

/** Toy dataset B — continuous x with a unique best midpoint. */
export const TOY_B: [number, number][] = [[0.5, 0], [2.3, 2], [4.7, 2], [9.1, 3]];

/**
 * INDEPENDENT exhaustive search: enumerate every midpoint between sorted
 * distinct x values, compute the split SSE with raw arithmetic (no module
 * imports — a genuine cross-check of bestSplitCandidate), return the argmin.
 */
export function exhaustiveBestSplit(pts: [number, number][]): { threshold: number; sse: number } {
  const xs = [...new Set(pts.map((p) => p[0]))].sort((a, b) => a - b);
  const sse = (rows: [number, number][]) => {
    const n = rows.length;
    if (n === 0) return 0;
    const m = rows.reduce((a, r) => a + r[1], 0) / n;
    return rows.reduce((a, r) => a + (r[1] - m) ** 2, 0);
  };
  let best: { threshold: number; sse: number } | null = null;
  for (let k = 0; k < xs.length - 1; k++) {
    const t = (xs[k] + xs[k + 1]) / 2;
    const left = pts.filter((p) => p[0] < t);
    const right = pts.filter((p) => p[0] >= t);
    const total = sse(left) + sse(right);
    if (!best || total < best.sse - 1e-12 || (Math.abs(total - best.sse) <= 1e-12 && t < best.threshold)) {
      best = { threshold: t, sse: total };
    }
  }
  return best!;
}

export const dtrTestCases: TestCase[] = [
  {
    name: 'regression tree splits at the SSE-minimizing midpoint (exhaustive check)',
    params: { xys: JSON.stringify(TOY_A), maxDepth: 1, minLeaf: 1, seed: 42 },
    maxSteps: 20,
    expect: {
      converged: true,
      eventLabels: ['cart-split', 'cart-tree-complete'],
      finalMetrics: {
        nLeaves: 2,
        nSplits: 1,
        // the module's chosen root split must be the exhaustive argmin…
        rootThreshold: (v) => v === exhaustiveBestSplit(TOY_A).threshold,
        rootSse: (v) => Math.abs(v - exhaustiveBestSplit(TOY_A).sse) < 1e-9,
        // …and the reduction must be strictly positive (a real gain).
        rootReduction: (v) => v > 0,
      },
    },
  },
  {
    name: 'CART chooses thresholds among sorted midpoints only (continuous features)',
    params: { xys: JSON.stringify(TOY_B), maxDepth: 1, minLeaf: 1, seed: 42 },
    maxSteps: 20,
    expect: {
      converged: true,
      finalMetrics: {
        nLeaves: 2,
        // (0.5+2.3)/2 = 1.4, (2.3+4.7)/2 = 3.5, (4.7+9.1)/2 = 6.9 — the ONLY
        // admissible thresholds; the argmin (verified independently) is 1.4.
        rootThreshold: (v) => v === 1.4 && v === exhaustiveBestSplit(TOY_B).threshold,
        rootSse: (v) => Math.abs(v - exhaustiveBestSplit(TOY_B).sse) < 1e-9,
      },
    },
  },
  {
    name: 'all-equal y → no useful split (degenerate: root stays a leaf)',
    params: { xys: '[[-2,3],[0,3],[2,3]]', maxDepth: 3, minLeaf: 1, seed: 42 },
    maxSteps: 20,
    expect: {
      converged: true,
      finalMetrics: {
        nLeaves: 1,
        nSplits: 0,
        sse: 0,                 // every y equals the leaf mean 3
        lastReduction: 0,       // nothing was ever split
        rootThreshold: -1,      // sentinel: the root never split
        rootSse: -1,
      },
    },
  },
  {
    name: 'single point → degenerate: a lone sample can never split',
    params: { xys: '[[1,2]]', maxDepth: 6, minLeaf: 1, seed: 42 },
    maxSteps: 20,
    expect: {
      converged: true,
      finalMetrics: {
        nLeaves: 1,
        nSplits: 0,
        sse: 0,
        lastReduction: 0,
      },
    },
  },
  {
    name: 'default run grows a tree then halts (train/test divergence story)',
    params: { n: 30, noise: 0.4, maxDepth: 4, minLeaf: 2, seed: 42 },
    maxSteps: 60,
    expect: {
      converged: true,
      eventLabels: ['cart-split', 'cart-tree-complete'],
      finalMetrics: {
        nLeaves: (v) => v >= 2,
        nSplits: (v) => v >= 1,
        trainError: (v) => v >= 0 && v <= 1,
        testError: (v) => v >= 0,
        r2: (v) => v > 0 && v <= 1,
      },
    },
  },
];
