// src/topics/hierarchical-clustering/testCases.ts
//
// Hand-verified datasets (the math is done by hand in testCases.test.ts):
//   DATASET A (single-linkage chain): p0=(0,0) p1=(1,0) p2=(3,0) p3=(3.5,0)
//     → m1=(p2,p3)@0.5, m2=(p0,p1)@1, m3@2; copheneticCorr ≈ 0.8985.
//   DATASET B: p0=(0,0) p1=(1,0.5) p2=(2,0.2) p3=(2.2,0.3)
//     → single: [0.22361, 1.04403, 1.11803]; complete: [0.22361, 1.11803, 2.22036]
//     (different dendrogram structure — the linkage-affects-structure case).
//   DATASET C (degenerate duplicates): [[0,0],[0,0],[1,0],[1,0]] → m1@0, m2@0, m3@1.
//
// Measured anchors (Gaussian-blob data — ALL measured by running the module):
//   n 8, linkage single, blobs 2, seed 42 → 7 merges; max merge height measured
//   in module.ts header; cutHeight 1.2 on the n=12 default run → cutClusters = 2
//   (the two blobs; cut@2.0 = 1 because max merge is 1.795 < 2.0).
import type { TestCase } from '../../engine/types';

export const hierarchicalTestCases: TestCase[] = [
  {
    // Plan case 1: single linkage merges the nearest pair. Hand-computed:
    // heights [0.5, 1, 2] on the 4-point chain (distances in the explanation).
    name: 'single linkage merges nearest pair: dataset A → heights 0.5, 1, 2',
    params: { n: 4, linkage: 'single', seed: 42, cutHeight: 2.5, points: '[[0,0],[1,0],[3,0],[3.5,0]]' },
    maxSteps: 50,
    expect: {
      converged: true,
      eventLabels: ['merge-1', 'merge-2', 'merge-3', 'agglomerative-complete'],
      finalMetrics: {
        mergeHeight: (v: number) => Math.abs(v - 2) < 1e-9,
        clusterCount: (v: number) => v === 1,
      },
      finalAlgorithm: {
        mode: 'agglomerative-complete',
        linkage: 'single',
        isDone: 1,
      },
    },
  },
  {
    // Plan case 2a: single linkage on dataset B — chaining through p1.
    name: 'linkage affects structure (single): dataset B final height 1.11803',
    params: { n: 4, linkage: 'single', seed: 42, cutHeight: 2.5, points: '[[0,0],[1,0.5],[2,0.2],[2.2,0.3]]' },
    maxSteps: 50,
    expect: {
      converged: true,
      finalMetrics: {
        mergeHeight: (v: number) => Math.abs(v - 1.118033988749895) < 1e-9,
        clusterCount: (v: number) => v === 1,
      },
    },
  },
  {
    // Plan case 2b: complete linkage on the SAME data — different final height
    // (2.22036) and a different second merge (the contrast is asserted in .test.ts).
    name: 'linkage affects structure (complete): dataset B final height 2.22036',
    params: { n: 4, linkage: 'complete', seed: 42, cutHeight: 2.5, points: '[[0,0],[1,0.5],[2,0.2],[2.2,0.3]]' },
    maxSteps: 50,
    expect: {
      converged: true,
      finalMetrics: {
        mergeHeight: (v: number) => Math.abs(v - 2.220360331117452) < 1e-9,
        clusterCount: (v: number) => v === 1,
      },
    },
  },
  {
    // Plan case 3: the dendrogram is a tree — 8 points → 7 merges (one snapshot
    // per merge); the .test.ts asserts every child reference is valid + acyclic.
    name: 'dendrogram is a tree: n=8 → 7 merges, one snapshot per merge',
    params: { n: 8, linkage: 'single', blobs: '2', seed: 42 },
    maxSteps: 500,
    expect: {
      converged: true,
      eventLabels: ['merge-1', 'merge-2', 'merge-3', 'merge-4', 'merge-5', 'merge-6', 'merge-7', 'agglomerative-complete'],
      finalMetrics: {
        clusterCount: (v: number) => v === 1,
        n: (v: number) => v === 8,
      },
    },
  },
  {
    // Plan case 4: cutting the dendrogram at height h. cutHeight 1.2 on the
    // default n=12 run → exactly 2 clusters (the two blobs; measured in
    // module.ts header — cut@2.0 gives 1 because max merge height is 1.795).
    name: 'cut at height: n=12 single linkage, cutHeight 1.2 → 2 clusters (two blobs)',
    params: { n: 12, linkage: 'single', blobs: '2', seed: 42, cutHeight: 1.2 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        cutClusters: (v: number) => Math.abs(v - 2) < 1e-9,
      },
    },
  },
  {
    // Degenerate: duplicate points → zero-distance merges (honest, no NaN).
    name: 'degenerate duplicates: zero-distance merges still terminate',
    params: { n: 4, linkage: 'single', seed: 42, points: '[[0,0],[0,0],[1,0],[1,0]]' },
    maxSteps: 50,
    expect: {
      converged: true,
      finalMetrics: {
        mergeHeight: (v: number) => Math.abs(v - 1) < 1e-9,
        copheneticCorr: (v: number) => v > 0 && v <= 1,
      },
    },
  },
  {
    // Ward linkage: variance-based cost still yields a valid tree (n−1 merges).
    name: 'ward linkage: variance-based merge cost → n−1 non-decreasing merges',
    params: { n: 10, linkage: 'ward', blobs: '2', seed: 42 },
    maxSteps: 500,
    expect: {
      converged: true,
      finalMetrics: {
        n: (v: number) => v === 10,
        clusterCount: (v: number) => v === 1,
      },
    },
  },
];
