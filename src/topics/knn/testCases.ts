// src/topics/knn/testCases.ts
import type { TestCase } from '../../engine/types';

// Empirical anchors (seed 42, nPerClass 12, margin 1.0, spread 1.5, grid 31):
//   regions: k=1 → 51, k=3 → 44, k=15 → 35   (monotone decrease — boundary smoothing)
//   trainError: k=1 → 0 (memorization), k=15 → 0.208
//   LOO error: k=1 → 0.417, k=15 → 0.208
// Tie-break craft (balanced 2+2): query (0,0), k=2, points [[1,0,1],[-1,0,0],[2,0,1],[0,2,0]] —
//   nearest two are (1,0)@d=1 cls1 and (-1,0)@d=1 cls0 → votes 1-1 → nearest-of-tied both d=1 →
//   lower class index wins → class 0.
// Metric craft (balanced 2+2): query (0,0), k=1, points [[3,4,0],[-3,4,0],[5,0,1],[0,5,1]] —
//   euclidean: all four at d=5 → tie → class 0; manhattan: (5,0),(0,5) at 5 < (3,4),(-3,4) at 7 → class 1.
export const knnTestCases: TestCase[] = [
  {
    // Plan case 1: k=1 memorizes the training set — every point is its own nearest neighbor.
    name: 'k=1 reproduces training labels (zero train error)',
    params: { k: 1, nPerClass: 12, margin: 1.0, metric: 'euclidean', seed: 42 },
    maxSteps: 20,
    expect: {
      finalMetrics: { trainError: (v: number) => v === 0 },
    },
  },
  {
    // Plan case 2: larger k smooths the boundary — region count drops (51 → 35 on this seed).
    name: 'k=15 smooths the decision boundary (region count < k=1)',
    params: { k: 15, nPerClass: 12, margin: 1.0, metric: 'euclidean', seed: 42 },
    maxSteps: 20,
    expect: {
      finalMetrics: { regions: (v: number) => v < 50 }, // k=1 gives 51, k=15 gives 35
    },
  },
  {
    // Plan case 3: exact-distance tie (both at d=1) at k=2 → votes 1-1 → nearest-of-tied
    // is equal → lower class index wins → class 0. Deterministic, verified by hand + scratch.
    name: 'majority vote tie-break: nearest of tied wins (exact-distance tie → class 0)',
    params: { k: 2, points: '[[1,0,1],[-1,0,0],[2,0,1],[0,2,0]]', queryX: 0, queryY: 0, metric: 'euclidean', seed: 42 },
    maxSteps: 20,
    expect: {
      finalMetrics: { queryClass: (v: number) => v === 0 },
    },
  },
  {
    // Plan case 4a: L1 vs L2 change the classification on the SAME points. Euclid: all four
    // neighbors at d=5 → tie → class 0 (lower index among tied).
    name: 'distance metric changes classification: euclidean → class 0',
    params: { k: 1, points: '[[3,4,0],[-3,4,0],[5,0,1],[0,5,1]]', queryX: 0, queryY: 0, metric: 'euclidean', seed: 42 },
    maxSteps: 20,
    expect: {
      finalMetrics: { queryClass: (v: number) => v === 0 },
    },
  },
  {
    // Plan case 4b: manhattan on the SAME points → (5,0) and (0,5) at 5 beat (3,4),(-3,4) at 7 → class 1.
    // The strict "classifications differ" claim is asserted in testCases.test.ts (cross-run).
    name: 'distance metric changes classification: manhattan → class 1',
    params: { k: 1, points: '[[3,4,0],[-3,4,0],[5,0,1],[0,5,1]]', queryX: 0, queryY: 0, metric: 'manhattan', seed: 42 },
    maxSteps: 20,
    expect: {
      finalMetrics: { queryClass: (v: number) => v === 1 },
    },
  },
  {
    // Extra (cheap): k=1 memorizes (train 0) but the smoothing from larger k costs train
    // accuracy — the "k=1 overfits" signature, honest and deterministic.
    name: 'k=1 memorizes (train 0) while k=15 loses train accuracy (train error > 0)',
    params: { k: 15, nPerClass: 12, margin: 1.0, metric: 'euclidean', seed: 42 },
    maxSteps: 20,
    expect: {
      finalMetrics: { trainError: (v: number) => v > 0 },
    },
  },
];
