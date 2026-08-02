import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import { simulation, mlrModule } from './module';
import { mlrTestCases } from './testCases';

describe('multiple-linear-regression testCases', () => {
  for (const tc of mlrTestCases) {
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

  // Plan spec case 3 requires testing the validateParams path explicitly (the engine
  // sandbox does not call validateParams — the UI does before a run).
  it('validateParams flags collinear features (XᵀX singular)', () => {
    const issues = mlrModule.validateParams?.({ n: 20, nFeatures: 2, noise: 0, collinear: true, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /collinear/i.test(s))).toBe(true);
  });

  it('validateParams flags n < d+1 (underdetermined system)', () => {
    const issues = mlrModule.validateParams?.({ n: 2, nFeatures: 3, noise: 0, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /underdetermined|rank-deficient/i.test(s))).toBe(true);
  });
});
