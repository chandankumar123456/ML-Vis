// src/topics/gradient-descent/testCases.test.ts
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import { simulation } from './module';
import { gdTestCases } from './testCases';

describe('gradient-descent testCases', () => {
  for (const tc of gdTestCases) {
    it(tc.name, () => {
      const run = computeRun(simulation, tc.params, tc.maxSteps ?? 500);
      // Tri-state per TestCase contract (engine/types.ts): only assert when the
      // case pins converged; an unset field means the run may end either way.
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
          if (typeof pred === 'function') expect(pred(m[k])).toBe(true);
        }
      }
      if (tc.expect.eventLabels) {
        const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
        for (const lbl of tc.expect.eventLabels) expect(labels).toContain(lbl);
      }
    });
  }
});
