import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import { simulation, dtrModule, register, giniImpurity, generateData, growTree, bestSplitCandidate } from './module';
import { dtrTestCases, TOY_A, TOY_B, exhaustiveBestSplit } from './testCases';
import { registerTopic, getTopic } from '../../registry/topicRegistry';

describe('decision-trees-regression testCases', () => {
  for (const tc of dtrTestCases) {
    it(tc.name, () => {
      const run = computeRun(simulation, tc.params, tc.maxSteps ?? 60);
      if (tc.expect.converged !== undefined) {
        if (tc.expect.converged) {
          expect(run.telemetry.failedAtStep).toBeUndefined();
        } else {
          expect(run.telemetry.failedAtStep).toBeDefined();
        }
      }
      if (tc.expect.finalMetrics) {
        const m = run.snapshots[run.snapshots.length - 1].metrics;
        for (const [k, pred] of Object.entries(tc.expect.finalMetrics)) {
          if (typeof pred === 'function') expect(pred(m[k]), `metric ${k} failed for ${tc.name}`).toBe(true);
          else expect(m[k]).toBeCloseTo(pred, 6);
        }
      }
      if (tc.expect.eventLabels) {
        const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
        for (const lbl of tc.expect.eventLabels) expect(labels).toContain(lbl);
      }
    });
  }

  // ---- Plan case 1: gini impurity (pure helper, exact) ----
  it('gini impurity: pure node = 0, uniform 2-class = 0.5 (exact)', () => {
    expect(giniImpurity([10])).toBe(0);                 // one pure class
    expect(giniImpurity([5, 5])).toBe(0.5);             // uniform 2-class
    expect(giniImpurity([3, 1])).toBe(0.375);           // 1 − (9/16 + 1/16)
    expect(giniImpurity([])).toBe(0);                   // empty guard
    expect(giniImpurity([0, 4])).toBe(0);               // zeros ignored
  });

  // ---- Plan case 2: midpoint optimum verified by independent exhaustive search ----
  it('best split on TOY_A minimizes SSE over ALL midpoints (independent check)', () => {
    const data = generateData({ xys: JSON.stringify(TOY_A) });
    const indices = TOY_A.map((_, i) => i);
    const cand = bestSplitCandidate(data, indices)!;
    const ref = exhaustiveBestSplit(TOY_A);
    expect(cand.threshold).toBeCloseTo(ref.threshold, 9);
    expect(cand.sseAfter).toBeCloseTo(ref.sse, 9);
    expect(cand.sseAfter).toBeCloseTo(0.125, 9);   // hand-computed anchor
    expect(cand.threshold).toBeCloseTo(-0.55, 9);  // (−1.2 + 0.1)/2
  });

  it('best split on TOY_B picks the unique argmin midpoint 1.4', () => {
    const data = generateData({ xys: JSON.stringify(TOY_B) });
    const indices = TOY_B.map((_, i) => i);
    const cand = bestSplitCandidate(data, indices)!;
    expect(cand.threshold).toBe(1.4);
    expect(cand.sseAfter).toBeCloseTo(2 / 3, 9);   // hand-computed anchor
    expect(cand.threshold).toBe(exhaustiveBestSplit(TOY_B).threshold);
  });

  // ---- Plan case 3: leaf prediction = mean of samples in the node ----
  it('regression leaf value = mean(y in node) for every leaf', () => {
    const data = generateData({ xys: JSON.stringify(TOY_A) });
    const grown = growTree(data, 3, 1, 3); // 3 splits → 4 leaves
    expect(grown.halted).toBe(false);
    const leaves = grown.nodes.filter((n) => n.children.length === 0);
    expect(leaves.length).toBe(4); // root + 3 splits = 4 leaves
    for (const leaf of leaves) {
      const ys = leaf.indices.map((i) => data.ys[i]);
      const inlineMean = ys.reduce((a, b) => a + b, 0) / ys.length;
      expect(leaf.value).toBeCloseTo(inlineMean, 12);
    }
    // hand-checked partition of TOY_A: {(−2,1)}, {(−1.2,1.3)},
    // {(0.1,−0.9),(1.5,−1.1)}, {(2.2,−0.7)} → values 1, 1.3, −1.0, −0.7
    const values = leaves.map((l) => l.value).sort((a, b) => a - b);
    expect(values).toEqual([-1, -0.7, 1, 1.3]);
  });

  // ---- Plan case 4: continuous features — thresholds among sorted midpoints ----
  it('CART thresholds are exactly the sorted-data midpoints (never arbitrary reals)', () => {
    const data = generateData({ xys: JSON.stringify(TOY_B) });
    const cand = bestSplitCandidate(data, TOY_B.map((_, i) => i))!;
    const mids = [1.4, 3.5, 6.9]; // ((0.5+2.3)/2, (2.3+4.7)/2, (4.7+9.1)/2)
    expect(mids).toContain(cand.threshold);
    // any admissible threshold is a midpoint of consecutive sorted distinct x
    const xs = [...new Set(TOY_B.map((p) => p[0]))].sort((a, b) => a - b);
    const allMids = xs.slice(0, -1).map((x, i) => (x + xs[i + 1]) / 2);
    expect(allMids).toContain(cand.threshold);
  });

  // ---- Determinism ----
  it('deterministic: same params → identical snapshot sequence', () => {
    const params = { n: 30, noise: 0.4, maxDepth: 4, minLeaf: 2, seed: 42 };
    const a = computeRun(simulation, params, 60);
    const b = computeRun(simulation, params, 60);
    expect(a.snapshots.length).toBe(b.snapshots.length);
    expect(JSON.stringify(a.snapshots)).toBe(JSON.stringify(b.snapshots));
    expect(a.telemetry.failedAtStep).toBeUndefined();
  });

  // ---- Degenerate input via computeRun (beyond the testCases list) ----
  it('all-equal-y dataset: run completes with a single unsplit leaf (0 SSE)', () => {
    const run = computeRun(simulation, { xys: '[[-2,3],[0,3],[2,3]]', maxDepth: 3, minLeaf: 1, seed: 42 }, 20);
    expect(run.snapshots.length).toBe(1); // root leaf only, no splits
    expect(run.snapshots[0].metrics.nLeaves).toBe(1);
    expect(run.snapshots[0].metrics.sse).toBe(0);
  });

  // ---- validateParams guards ----
  it('validateParams flags degenerate parameter combinations', () => {
    const vp = dtrModule.validateParams!;
    expect(vp({ n: 4, noise: 0.4, maxDepth: 4, minLeaf: 2, seed: 42 }).some((s) => /n must be/i.test(s))).toBe(true);
    expect(vp({ n: 30, noise: 0.4, maxDepth: 4, minLeaf: 12, seed: 42 }).some((s) => /minLeaf/i.test(s))).toBe(true);
    expect(vp({ n: 30, noise: 0.4, maxDepth: 7, minLeaf: 2, seed: 42 }).some((s) => /maxDepth/i.test(s))).toBe(true);
    expect(vp({ n: 30, noise: -1, maxDepth: 4, minLeaf: 2, seed: 42 }).some((s) => /noise/i.test(s))).toBe(true);
    expect(vp({ n: 30, noise: 0.4, maxDepth: 4, minLeaf: 2, seed: -5 }).some((s) => /seed/i.test(s))).toBe(true);
    expect(vp({ n: 30, noise: 0.4, maxDepth: 4, minLeaf: 2, seed: 42 })).toEqual([]);
    expect(vp({ xys: '[[1,2],[2,3]]', minLeaf: 2, seed: 42 }).some((s) => /minLeaf/i.test(s))).toBe(true);
  });

  // ---- Registration idempotence ----
  it('register() is idempotent: twice → one module, no throw', () => {
    expect(() => { register(); register(); }).not.toThrow();
    const t = getTopic('decision-trees-regression');
    expect(t).toBeDefined();
    expect(t!.version).toBe(1);
  });

  // registration must survive a fresh registerTopic call of the same version
  it('re-register keeps the existing module (version guard)', () => {
    registerTopic(dtrModule);
    registerTopic(dtrModule);
    expect(getTopic('decision-trees-regression')?.version).toBe(1);
  });
});
