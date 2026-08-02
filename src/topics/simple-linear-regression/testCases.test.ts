import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import { simulation } from './module';
import { slrTestCases } from './testCases';

describe('simple-linear-regression testCases', () => {
  for (const tc of slrTestCases) {
    it(tc.name, () => {
      const run = computeRun(simulation, tc.params, tc.maxSteps ?? 500);
      if (tc.expect.finalMetrics) {
        const m = run.snapshots[run.snapshots.length - 1].metrics;
        for (const [k, pred] of Object.entries(tc.expect.finalMetrics)) {
          // finalMetrics may hold predicates OR plain numbers (type union)
          if (typeof pred === 'function') expect(pred(m[k])).toBe(true);
          else expect(m[k]).toBeCloseTo(pred, 6);
        }
      }
    });
  }
});
