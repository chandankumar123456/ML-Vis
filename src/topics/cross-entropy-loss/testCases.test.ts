import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import { simulation, ceModule, crossEntropy } from './module';
import { ceTestCases } from './testCases';

describe('cross-entropy-loss testCases', () => {
  for (const tc of ceTestCases) {
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
      if (tc.expect.eventLabels) {
        const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
        for (const lbl of tc.expect.eventLabels) expect(labels).toContain(lbl);
      }
    });
  }

  // ---- numeric identities on the LIVE sweep (not just the final snapshot) ----

  it('CE = H + KL identity holds on every cross-entropy sweep snapshot', () => {
    const run = computeRun(simulation, { facet: 'cross-entropy', p0: 0.7, q0: 0.95 }, 500);
    expect(run.snapshots.length).toBe(19); // 0.05 → 0.95 at 0.05 steps
    for (const s of run.snapshots) {
      expect(Math.abs(s.metrics.cePQ - (s.metrics.hP + s.metrics.klPQ))).toBeLessThan(1e-12);
      expect(s.metrics.cePQ).toBeGreaterThanOrEqual(0);
    }
  });

  it('cross-entropy is asymmetric: CE(p,q) ≠ CE(q,p) with a clear margin', () => {
    const run = computeRun(simulation, { facet: 'cross-entropy', p0: 0.8, q0: 0.3 }, 500);
    const m = run.snapshots[run.snapshots.length - 1].metrics;
    // Independent re-computation in the test (definition only — mirrors the run's
    // own crossEntropy for the same params; the swapped direction must differ).
    const cpq = crossEntropy([0.8, 0.2], [0.3, 0.7]);
    const cqp = crossEntropy([0.3, 0.7], [0.8, 0.2]);
    expect(Math.abs(m.cePQ - cpq)).toBeLessThan(1e-12); // run matches the definition
    expect(Math.abs(cpq - cqp)).toBeGreaterThan(0.1); // asymmetry margin (0.159 actual)
    expect(Math.abs(m.cePQ - cqp)).toBeGreaterThan(0.1);
  });

  it('MLE maximizes likelihood ⟺ minimizes per-sample cross-entropy (NLL = CE)', () => {
    const run = computeRun(simulation, { facet: 'mle', nFlips: 20, heads: 12 }, 500);
    const snapshots = run.snapshots;
    const argmaxLike = snapshots.reduce(
      (best, s, i) => (s.metrics.likelihood > snapshots[best].metrics.likelihood ? i : best),
      0
    );
    const argminCE = snapshots.reduce(
      (best, s, i) => (s.metrics.cePQ < snapshots[best].metrics.cePQ ? i : best),
      0
    );
    expect(argmaxLike).toBe(argminCE); // the equivalence
    // θ̂ = 0.6 lies ON the θ grid, so the argmax snapshot is exactly θ = 0.6.
    expect(Math.abs(snapshots[argmaxLike].metrics.theta - 0.6)).toBeLessThan(1e-12);
    expect(snapshots[argmaxLike].events.some((e) => e.label === 'mle-at-argmax')).toBe(true);
    expect(run.snapshots.length).toBe(21); // θ grid 0.02 → 0.98
  });

  it('cross-entropy over q0 is minimized exactly at q0 = p0 (strict convexity)', () => {
    const run = computeRun(simulation, { facet: 'cross-entropy', p0: 0.7, q0: 0.95 }, 500);
    const snapshots = run.snapshots;
    const argmin = snapshots.reduce(
      (best, s, i) => (s.metrics.cePQ < snapshots[best].metrics.cePQ ? i : best),
      0
    );
    expect(snapshots[argmin].algorithm.q0).toBeCloseTo(0.7, 12);
    // the sweep is monotone: q0 increases by 0.05 every snapshot
    for (let i = 1; i < snapshots.length; i++) {
      expect((snapshots[i].algorithm.q0 as number) - (snapshots[i - 1].algorithm.q0 as number)).toBeCloseTo(0.05, 12);
    }
  });

  // ---- explicit validateParams path tests (the engine sandbox does not call it) ----

  it('validateParams rejects an unknown facet', () => {
    const issues = ceModule.validateParams?.({ facet: 'bogus', p0: 0.7, q0: 0.3 }) ?? [];
    expect(issues.some((s) => /facet/i.test(s))).toBe(true);
  });

  it('validateParams rejects out-of-range / degenerate probabilities', () => {
    const p = ceModule.validateParams ?? (() => []);
    expect(p({ facet: 'cross-entropy', p0: 0, q0: 0.3 }).some((s) => /out of range/i.test(s))).toBe(true);
    expect(p({ facet: 'cross-entropy', p0: 1.5, q0: 0.3 }).some((s) => /out of range/i.test(s))).toBe(true);
    expect(p({ facet: 'cross-entropy', p0: 0.7, q0: 0 }).some((s) => /degenerate|log\(0\)/i.test(s))).toBe(true);
    expect(p({ facet: 'cross-entropy', p0: 0.7, q0: 1 }).some((s) => /degenerate|log\(0\)/i.test(s))).toBe(true);
    expect(p({ facet: 'cross-entropy', p0: 0.7, q0: 0.97 }).some((s) => /sweep grid/i.test(s))).toBe(true);
  });

  it('validateParams rejects invalid MLE counts', () => {
    const p = ceModule.validateParams ?? (() => []);
    expect(p({ facet: 'mle', nFlips: 0, heads: 0 }).some((s) => /nFlips/i.test(s))).toBe(true);
    expect(p({ facet: 'mle', nFlips: 20, heads: 21 }).some((s) => /cannot exceed/i.test(s))).toBe(true);
    expect(p({ facet: 'mle', nFlips: 20, heads: -1 }).some((s) => /non-negative/i.test(s))).toBe(true);
  });

  it('validateParams passes a valid full parameter set', () => {
    const issues = ceModule.validateParams?.({ facet: 'cross-entropy', p0: 0.7, q0: 0.3, nFlips: 20, heads: 12 }) ?? [];
    expect(issues).toHaveLength(0);
    const mleIssues = ceModule.validateParams?.({ facet: 'mle', p0: 0.7, q0: 0.3, nFlips: 20, heads: 12 }) ?? [];
    expect(mleIssues).toHaveLength(0);
  });
});
