// src/topics/knn/testCases.test.ts
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import { simulation, knnModule } from './module';
import { knnTestCases } from './testCases';

describe('knn testCases', () => {
  for (const tc of knnTestCases) {
    it(tc.name, () => {
      const run = computeRun(simulation, tc.params, tc.maxSteps ?? 500);
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
          // finalMetrics may hold predicates OR plain numbers (type union) — dispatch
          if (typeof pred === 'function') expect(pred(m[k]), `metric ${k} failed for ${tc.name}`).toBe(true);
          else expect(m[k]).toBeCloseTo(pred, 6);
        }
      }
    });
  }

  it('the same dataset classifies differently under L1 vs L2 (euclidean 0, manhattan 1)', () => {
    const params = { k: 1, points: '[[3,4,0],[-3,4,0],[5,0,1],[0,5,1]]', queryX: 0, queryY: 0, seed: 42 };
    const eu = computeRun(simulation, { ...params, metric: 'euclidean' }, 20);
    const ma = computeRun(simulation, { ...params, metric: 'manhattan' }, 20);
    const mEu = eu.snapshots[eu.snapshots.length - 1].metrics;
    const mMa = ma.snapshots[ma.snapshots.length - 1].metrics;
    expect(mEu.queryClass).toBe(0);
    expect(mMa.queryClass).toBe(1);
    expect(mEu.queryClass).not.toBe(mMa.queryClass);
  });

  it('the k-sweep last snapshot matches the slider k (loss curves align to the UI dial)', () => {
    const run = computeRun(simulation, { k: 9, nPerClass: 12, margin: 1.0, metric: 'euclidean', seed: 42 }, 20);
    const last = run.snapshots[run.snapshots.length - 1].metrics;
    expect(last.k).toBe(9);
    // one snapshot per k on [1..9] → exactly 9 snapshots, no duplicates
    expect(run.snapshots).toHaveLength(9);
    expect(run.snapshots.map((s) => s.metrics.k)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('LOO error (error) is nonzero at k=1 — the honest, non-memorizing metric', () => {
    const run = computeRun(simulation, { k: 1, nPerClass: 12, margin: 1.0, metric: 'euclidean', seed: 42 }, 20);
    const m = run.snapshots[run.snapshots.length - 1].metrics;
    expect(m.trainError).toBe(0); // memorization
    expect(m.error).toBeGreaterThan(0); // LOO — honest
    expect(m.error).toBeGreaterThan(m.trainError as number);
  });

  // The engine sandbox does NOT call validateParams (the UI does before a run) — so the
  // plan spec demands explicit tests of the validation path.
  it('validateParams rejects k = 0', () => {
    const issues = knnModule.validateParams?.({ k: 0, nPerClass: 12, metric: 'euclidean', seed: 42 }) ?? [];
    expect(issues.some((s) => /k must be ≥ 1/i.test(s))).toBe(true);
  });

  it('validateParams rejects k > dataset size (2·nPerClass)', () => {
    const issues = knnModule.validateParams?.({ k: 25, nPerClass: 12, metric: 'euclidean', seed: 42 }) ?? [];
    expect(issues.some((s) => /exceeds the dataset size/i.test(s))).toBe(true);
  });

  it('validateParams rejects unbalanced custom points (odd count)', () => {
    const issues = knnModule.validateParams?.({ k: 2, points: '[[1,0,0],[0,0,1],[2,0,0]]', seed: 42 }) ?? [];
    expect(issues.some((s) => /even count/i.test(s))).toBe(true);
  });

  it('validateParams rejects custom points with unequal class counts', () => {
    const issues = knnModule.validateParams?.({ k: 2, points: '[[1,0,0],[0,0,1],[2,0,0],[3,0,0]]', seed: 42 }) ?? [];
    expect(issues.some((s) => /equal class counts/i.test(s))).toBe(true);
  });

  it('validateParams rejects custom points outside the [−5,5]² domain', () => {
    const issues = knnModule.validateParams?.({ k: 2, points: '[[1,0,0],[-1,0,1],[9,0,0],[-9,0,1]]', seed: 42 }) ?? [];
    expect(issues.some((s) => /within \[−5,5\]²/i.test(s))).toBe(true);
  });

  it('validateParams accepts a valid balanced custom dataset and rejects k > its size', () => {
    const valid = knnModule.validateParams?.({ k: 2, points: '[[1,0,0],[-1,0,1],[2,0,0],[-2,0,1]]', seed: 42 }) ?? [];
    expect(valid.length).toBe(0);
    const oversized = knnModule.validateParams?.({ k: 6, points: '[[1,0,0],[-1,0,1],[2,0,0],[-2,0,1]]', seed: 42 }) ?? [];
    expect(oversized.some((s) => /exceeds the custom dataset size/i.test(s))).toBe(true);
  });
});
