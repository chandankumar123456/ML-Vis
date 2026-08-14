// src/topics/hierarchical-clustering/testCases.test.ts
//
// Hand-verified datasets (the math is done by hand below — no module values):
//
// DATASET A (single-linkage chain, 4 collinear points):
//   p0=(0,0) p1=(1,0) p2=(3,0) p3=(3.5,0)
//   d(p0,p1)=1, d(p0,p2)=3, d(p0,p3)=3.5, d(p1,p2)=2, d(p1,p3)=2.5, d(p2,p3)=0.5
//   Single linkage:  m1=(p2,p3)@0.5  m2=(p0,p1)@1  m3=({p0,p1},{p2,p3})@2
//     (min across the two clusters = min(3,3.5,2,2.5) = 2)
//   Cophenetic distances: c(p2,p3)=0.5, c(p0,p1)=1, all cross pairs=2.
//   Cophenetic correlation: Pearson over (d,c) pairs
//     x=[1,3,3.5,2,2.5,0.5] y=[1,2,2,2,2,0.5]
//     r = (6·23.25 − 12.5·9.5)/√((6·32.75−12.5²)(6·17.25−9.5²)) = 20.75/√533.3125 ≈ 0.8985
//
// DATASET B (linkage changes the STRUCTURE, 4 points):
//   p0=(0,0) p1=(1,0.5) p2=(2,0.2) p3=(2.2,0.3)
//   d01=√1.25≈1.11803, d02=√4.04≈2.00998, d03=√4.93≈2.22027,
//   d12=√1.09≈1.04403, d13=√1.48≈1.21655, d23=√0.05≈0.22361
//   SINGLE:  m1=(p2,p3)@0.22361  m2=({p2,p3},p1)@1.04403  m3=({p1,p2,p3},p0)@1.11803
//   COMPLETE: m1=(p2,p3)@0.22361  m2=(p0,p1)@1.11803     m3=({p0,p1},{p2,p3})@2.22027
//   → different second merge AND different final height (1.118 vs 2.220).
//
// DATASET C (degenerate — duplicate points → zero-distance merges):
//   p0=(0,0) p1=(0,0) p2=(1,0) p3=(1,0)  → m1=(p0,p1)@0, m2=(p2,p3)@0, m3@1.
//
// Measured anchors (Gaussian-blob data, ALL verified by running the module):
//   see the anchors block in testCases.ts / module.ts header.
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import type { SnapshotRun } from '../../engine/types';
import { listTopics } from '../../registry/topicRegistry';
import {
  simulation,
  hierarchicalClusteringModule,
  register,
  agglomerate,
  clustersAtHeight,
  clusterCountAtHeight,
  computeResult,
  generateData,
  distanceMatrix,
  type MergeResult,
} from './module';
import { hierarchicalTestCases } from './testCases';

const DATASET_A: [number, number][] = [[0, 0], [1, 0], [3, 0], [3.5, 0]];
const DATASET_B: [number, number][] = [[0, 0], [1, 0.5], [2, 0.2], [2.2, 0.3]];
const DATASET_C: [number, number][] = [[0, 0], [0, 0], [1, 0], [1, 0]];
const paramsA = (linkage: string) => ({
  n: 4, linkage, seed: 42, cutHeight: 2.5,
  points: JSON.stringify(DATASET_A),
});
const paramsB = (linkage: string) => ({
  n: 4, linkage, seed: 42, cutHeight: 2.5,
  points: JSON.stringify(DATASET_B),
});
const paramsC = { n: 4, linkage: 'single', seed: 42, points: JSON.stringify(DATASET_C) };
const GAUSS = { n: 8, linkage: 'single', blobs: '2', seed: 42 };

/** The FULL chronological merge list = the final snapshot's dendrogram commands. */
function fullMerges(run: SnapshotRun): MergeResult[] {
  const last = run.snapshots[run.snapshots.length - 1];
  return last.visuals
    .filter((v) => v.type === 'merge')
    .map((v) => ({ id: v.id as string, height: v.height as number, children: v.children as string[] }));
}

// ---------------------------------------------------------------------------
// data-driven testCases (the centralized runner runs the same list)
// ---------------------------------------------------------------------------
describe('hierarchical-clustering testCases (data-driven)', () => {
  for (const tc of hierarchicalTestCases) {
    it(tc.name, () => {
      const run = computeRun(simulation, tc.params, tc.maxSteps ?? 500);
      if (tc.expect.converged !== undefined) {
        if (tc.expect.converged) expect(run.telemetry.failedAtStep).toBeUndefined();
        else expect(run.telemetry.failedAtStep).toBeDefined();
      }
      if (tc.expect.eventLabels) {
        const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
        for (const lbl of tc.expect.eventLabels) expect(labels).toContain(lbl);
      }
      if (tc.expect.finalMetrics) {
        const m = run.snapshots[run.snapshots.length - 1].metrics;
        for (const [k, pred] of Object.entries(tc.expect.finalMetrics)) {
          if (typeof pred === 'function') expect(pred(m[k]), `metric ${k} failed for ${tc.name}`).toBe(true);
          else expect(m[k]).toBeCloseTo(pred, 6);
        }
      }
      if (tc.expect.finalAlgorithm) {
        const a = run.snapshots[run.snapshots.length - 1].algorithm;
        for (const [k, pred] of Object.entries(tc.expect.finalAlgorithm)) {
          if (typeof pred === 'function') expect(pred(a[k]), `algorithm ${k} failed for ${tc.name}`).toBe(true);
          else expect(a[k]).toBe(pred);
        }
      }
    });
  }
});

// ---------------------------------------------------------------------------
// Plan case 1: single linkage merges the nearest pair (hand-computed)
// ---------------------------------------------------------------------------
describe('hierarchical: plan case 1 — single linkage merges nearest pair', () => {
  it('dataset A: merge order (p2,p3)@0.5 → (p0,p1)@1 → final@2, exactly', () => {
    const run = computeRun(simulation, paramsA('single'), 50);
    expect(run.telemetry.failedAtStep).toBeUndefined();
    const merges = fullMerges(run);
    expect(merges.length).toBe(3);
    expect(merges[0].height).toBeCloseTo(0.5, 12);
    expect(merges[0].children).toEqual(['p2', 'p3']);
    expect(merges[1].height).toBeCloseTo(1, 12);
    expect(merges[1].children).toEqual(['p0', 'p1']);
    expect(merges[2].height).toBeCloseTo(2, 12);
    expect(merges[2].children).toEqual(['m2', 'm1']);
  });

  it('cophenetic correlation of dataset A = 0.8985 (hand-computed Pearson)', () => {
    const r = agglomerate(DATASET_A.map(([x, y]) => ({ x, y })), 'single');
    expect(r.copheneticCorr).toBeCloseTo(0.89852, 2);
  });

  it('cophenetic matrix of dataset A: (p2,p3)→0.5, (p0,p1)→1, cross pairs→2', () => {
    const r = agglomerate(DATASET_A.map(([x, y]) => ({ x, y })), 'single');
    expect(r.cophenetic[2][3]).toBeCloseTo(0.5, 12);
    expect(r.cophenetic[0][1]).toBeCloseTo(1, 12);
    for (const [i, j] of [[0, 2], [0, 3], [1, 2], [1, 3]] as const) {
      expect(r.cophenetic[i][j]).toBeCloseTo(2, 12);
    }
  });
});

// ---------------------------------------------------------------------------
// Plan case 2: linkage affects the structure (single vs complete)
// ---------------------------------------------------------------------------
describe('hierarchical: plan case 2 — linkage affects structure', () => {
  it('dataset B: single and complete produce DIFFERENT second merges', () => {
    const single = fullMerges(computeRun(simulation, paramsB('single'), 50));
    const complete = fullMerges(computeRun(simulation, paramsB('complete'), 50));
    // same first merge (the tight pair), different second merge:
    expect(single[0].children).toEqual(['p2', 'p3']);
    expect(complete[0].children).toEqual(['p2', 'p3']);
    expect(single[1].children).toEqual(['p1', 'm1']);     // single chains p1 onto the pair
    expect(complete[1].children).toEqual(['p0', 'p1']);   // complete merges the far pair first
    // and the final merge heights differ: 1.118 vs 2.220
    expect(single[2].height).toBeCloseTo(1.118033988749895, 9);
    expect(complete[2].height).toBeCloseTo(2.220360331117452, 9);
    expect(single[2].height).not.toBeCloseTo(complete[2].height, 6);
  });

  it('dataset B, complete linkage: full hand-computed merge list', () => {
    const merges = fullMerges(computeRun(simulation, paramsB('complete'), 50));
    expect(merges[0].height).toBeCloseTo(0.22360679774997896, 9);
    expect(merges[1].height).toBeCloseTo(1.118033988749895, 9);
    expect(merges[2].height).toBeCloseTo(2.220360331117452, 9);
    expect(merges[2].children).toEqual(['m2', 'm1']);
  });

  it('dataset B, single linkage: full hand-computed merge list', () => {
    const merges = fullMerges(computeRun(simulation, paramsB('single'), 50));
    expect(merges[0].height).toBeCloseTo(0.22360679774997896, 9);
    expect(merges[1].height).toBeCloseTo(1.0440306508910547, 9);
    expect(merges[2].height).toBeCloseTo(1.118033988749895, 9);
    expect(merges[2].children).toEqual(['p0', 'm2']);
  });

  it('the same config with a deterministic seed reproduces both dendrograms', () => {
    const a = fullMerges(computeRun(simulation, paramsB('complete'), 50));
    const b = fullMerges(computeRun(simulation, paramsB('complete'), 50));
    expect(a).toEqual(b);
  });
});

// ---------------------------------------------------------------------------
// Plan case 3: the dendrogram is a tree — n points → n−1 merges
// ---------------------------------------------------------------------------
describe('hierarchical: plan case 3 — dendrogram is a tree', () => {
  it('n=8 Gaussian run has exactly 7 merges, in chronological order', () => {
    const run = computeRun(simulation, GAUSS, 500);
    expect(run.telemetry.failedAtStep).toBeUndefined();
    expect(run.snapshots.length).toBe(7); // one snapshot per merge
    const merges = fullMerges(run);
    expect(merges.length).toBe(7);
  });

  it('every merge references only leaves or EARLIER merges (no forward/unknown refs)', () => {
    const run = computeRun(simulation, GAUSS, 500);
    const merges = fullMerges(run);
    const defined = new Set<string>(merges.map((m) => m.id));
    const seen = new Set<string>();
    for (const m of merges) {
      for (const c of m.children) {
        // valid iff it is a leaf id (p0..) or an already-emitted merge id
        expect(defined.has(c) || /^p\d+$/.test(c), `merge ${m.id} references unknown id ${c}`).toBe(true);
        if (defined.has(c)) expect(seen.has(c), `merge ${m.id} forward-references ${c}`).toBe(true);
      }
      seen.add(m.id);
    }
  });

  it('merge heights are non-decreasing across the run (monotone dendrogram)', () => {
    const run = computeRun(simulation, GAUSS, 500);
    const merges = fullMerges(run);
    for (let i = 1; i < merges.length; i++) {
      expect(merges[i].height).toBeGreaterThanOrEqual(merges[i - 1].height - 1e-9);
    }
  });

  it('every leaf appears in the dendrogram exactly once as a child', () => {
    const run = computeRun(simulation, GAUSS, 500);
    const merges = fullMerges(run);
    const leafCounts = new Map<string, number>();
    for (const m of merges) for (const c of m.children) {
      if (/^p\d+$/.test(c)) leafCounts.set(c, (leafCounts.get(c) ?? 0) + 1);
    }
    expect(leafCounts.size).toBe(8);
    for (const count of leafCounts.values()) expect(count).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Plan case 4: cutting the dendrogram at height h
// ---------------------------------------------------------------------------
describe('hierarchical: plan case 4 — cut at height h', () => {
  it('dataset A (heights 0.5, 1, 2): exact cluster counts at hand-computed cuts', () => {
    const r = agglomerate(DATASET_A.map(([x, y]) => ({ x, y })), 'single');
    expect(clusterCountAtHeight(r, 0.4)).toBe(4);  // no merge cut
    expect(clusterCountAtHeight(r, 0.75)).toBe(3); // only m1 (0.5) cut
    expect(clusterCountAtHeight(r, 1.5)).toBe(2);  // m1 + m2 cut
    expect(clusterCountAtHeight(r, 2.5)).toBe(1);  // everything cut
  });

  it('cut counts are monotone: higher cut → fewer (never more) clusters', () => {
    const r = computeResult(GAUSS);
    const heights = [0, 0.2, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, r.maxHeight + 1];
    let prev = clusterCountAtHeight(r, heights[0]);
    for (let i = 1; i < heights.length; i++) {
      const cur = clusterCountAtHeight(r, heights[i]);
      expect(cur).toBeLessThanOrEqual(prev);
      prev = cur;
    }
  });

  it('cut at 0 → n singletons; cut at ≥ maxHeight → 1 cluster', () => {
    const r = computeResult(GAUSS);
    expect(clusterCountAtHeight(r, 0)).toBe(8);
    expect(clusterCountAtHeight(r, r.maxHeight)).toBe(1);
  });

  it('clustersAtHeight returns the exact member sets of each cut cluster', () => {
    const r = agglomerate(DATASET_A.map(([x, y]) => ({ x, y })), 'single');
    const at075 = clustersAtHeight(r, 0.75);
    expect(at075.length).toBe(3);
    expect(at075).toContainEqual([2, 3]); // the m1 cluster (members p2,p3)
  });
});

// ---------------------------------------------------------------------------
// Determinism + validateParams + degenerate data + registration
// ---------------------------------------------------------------------------
describe('hierarchical: determinism', () => {
  it('same params → identical snapshot arrays (two full runs)', () => {
    const a = computeRun(simulation, GAUSS, 500);
    const b = computeRun(simulation, GAUSS, 500);
    expect(a.snapshots).toEqual(b.snapshots);
    expect(a.telemetry.snapshotCount).toBe(b.telemetry.snapshotCount);
  });

  it('different seeds → different data but the SAME structural identities (n−1 merges)', () => {
    const run = computeRun(simulation, { ...GAUSS, seed: 7 }, 500);
    expect(run.telemetry.failedAtStep).toBeUndefined();
    expect(fullMerges(run).length).toBe(7);
  });
});

describe('hierarchical: validateParams guards', () => {
  it('rejects n outside [4, 20]', () => {
    expect(hierarchicalClusteringModule.validateParams?.({ ...GAUSS, n: 3 })).not.toEqual([]);
    expect(hierarchicalClusteringModule.validateParams?.({ ...GAUSS, n: 21 })).not.toEqual([]);
    expect(hierarchicalClusteringModule.validateParams?.({ ...GAUSS, n: 4.5 })).not.toEqual([]);
  });
  it('rejects unknown linkage and bad seed', () => {
    expect(hierarchicalClusteringModule.validateParams?.({ ...GAUSS, linkage: 'centroid' })).not.toEqual([]);
    expect(hierarchicalClusteringModule.validateParams?.({ ...GAUSS, seed: -1 })).not.toEqual([]);
    expect(hierarchicalClusteringModule.validateParams?.({ ...GAUSS, seed: 10000 })).not.toEqual([]);
  });
  it('rejects malformed points override', () => {
    expect(hierarchicalClusteringModule.validateParams?.({ ...GAUSS, points: '[[0,0]]' })).not.toEqual([]);
    expect(hierarchicalClusteringModule.validateParams?.({ ...GAUSS, points: 'not json' })).not.toEqual([]);
  });
  it('accepts the valid defaults', () => {
    expect(hierarchicalClusteringModule.validateParams?.(GAUSS)).toEqual([]);
    expect(hierarchicalClusteringModule.validateParams?.({ n: 20, linkage: 'ward', blobs: '3', seed: 9999, cutHeight: 8 })).toEqual([]);
  });
});

describe('hierarchical: degenerate data', () => {
  it('duplicate points → zero-distance merges, still n−1 merges', () => {
    const run = computeRun(simulation, paramsC, 50);
    expect(run.telemetry.failedAtStep).toBeUndefined();
    const merges = fullMerges(run);
    expect(merges.length).toBe(3);
    expect(merges[0].height).toBeCloseTo(0, 12);
    expect(merges[1].height).toBeCloseTo(0, 12);
    expect(merges[2].height).toBeCloseTo(1, 12);
    expect(merges[0].children).toEqual(['p0', 'p1']);
    expect(merges[1].children).toEqual(['p2', 'p3']);
  });
});

describe('hierarchical: registration idempotence', () => {
  it('register() twice registers exactly one hierarchical-clustering topic', () => {
    const before = listTopics().filter((t) => t.id === 'hierarchical-clustering').length;
    register();
    register();
    const after = listTopics().filter((t) => t.id === 'hierarchical-clustering').length;
    expect(after).toBe(1);
    expect(before).toBeLessThanOrEqual(1); // standalone: 0; via test-runner: 1
  });
});

// ---------------------------------------------------------------------------
// pure-function sanity (the math core is mutation-free)
// ---------------------------------------------------------------------------
describe('hierarchical: pure math core', () => {
  it('distanceMatrix is symmetric with zero diagonal', () => {
    const pts = generateData({ ...GAUSS, n: 6 });
    const D = distanceMatrix(pts);
    for (let i = 0; i < 6; i++) {
      expect(D[i][i]).toBe(0);
      for (let j = i + 1; j < 6; j++) {
        expect(D[i][j]).toBe(D[j][i]);
        expect(D[i][j]).toBeGreaterThan(0);
      }
    }
  });

  it('ward linkage also terminates with n−1 non-decreasing merges (variance units)', () => {
    const r = computeResult({ ...GAUSS, n: 10, linkage: 'ward' });
    expect(r.merges.length).toBe(9);
    for (let i = 1; i < r.merges.length; i++) {
      expect(r.merges[i].height).toBeGreaterThanOrEqual(r.merges[i - 1].height - 1e-9);
    }
    expect(r.copheneticCorr).toBeGreaterThan(0);
  });

  it('average linkage terminates with n−1 merges', () => {
    const r = computeResult({ ...GAUSS, n: 9, linkage: 'average' });
    expect(r.merges.length).toBe(8);
    expect(r.copheneticCorr).toBeGreaterThan(0);
  });
});