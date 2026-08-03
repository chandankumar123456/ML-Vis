import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import { simulation, polyModule } from './module';
import { polyTestCases } from './testCases';

describe('polynomial-regression testCases', () => {
  for (const tc of polyTestCases) {
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

  // ---------------------------------------------------------------------------
  // Plan spec case 4 — bias-variance demo, averaged over 5 fixed seeds.
  // The TestCase contract is single-run, so the multi-seed error curve is
  // asserted here directly on the simulation (the honest teaching story):
  //   • avg train error falls monotonically as degree rises (bias shrinks)
  //   • avg test error is U-shaped: d=1 underfits, d=2 (the truth) is best,
  //     high degrees overfit catastrophically
  // ---------------------------------------------------------------------------
  it('bias-variance: train error falls with degree, test error is U-shaped (5-seed averages)', () => {
    const seeds = [1, 2, 3, 4, 5];
    const degrees = [1, 2, 3, 5, 8, 12, 15];
    const base = { nTrain: 30, nTest: 20, noise: 0.5, fitOn: 'train' };
    const curve = degrees.map((d) => {
      let tr = 0, te = 0;
      for (const s of seeds) {
        const snap = simulation.initialState({ ...base, degree: d, seed: s });
        tr += snap.metrics.trainMse;
        te += snap.metrics.testMse;
      }
      return { d, tr: tr / seeds.length, te: te / seeds.length };
    });
    // train error monotonically decreases (more flexible model → better train fit)
    for (let i = 1; i < curve.length; i++) {
      expect(curve[i].tr, `avgTrain(d=${curve[i].d}) < avgTrain(d=${curve[i - 1].d})`).toBeLessThan(curve[i - 1].tr);
    }
    const d1 = curve.find((c) => c.d === 1)!;
    const d2 = curve.find((c) => c.d === 2)!;
    const d5 = curve.find((c) => c.d === 5)!;
    const d8 = curve.find((c) => c.d === 8)!;
    const d15 = curve.find((c) => c.d === 15)!;
    // U-shape: d=1 underfits (bias), d=2 is the truth, high degrees overfit (variance)
    expect(d1.te).toBeGreaterThan(d2.te);
    expect(d5.te).toBeGreaterThan(d2.te);
    expect(d8.te).toBeGreaterThan(d2.te);
    expect(d15.te).toBeGreaterThan(d2.te);
    // the overfit tail must be catastrophic, not marginal
    expect(d15.te).toBeGreaterThan(d8.te);
  });

  // Plan spec: degree ≥ n → underdetermined → validateParams flags it explicitly
  // (the engine sandbox does NOT call validateParams — the UI does before a run).
  it('validateParams flags degree ≥ nTrain (underdetermined ΦᵀΦ)', () => {
    const issues = polyModule.validateParams?.({ degree: 20, nTrain: 15, noise: 0.5, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /underdetermined|rank-deficient/i.test(s))).toBe(true);
  });

  it('validateParams warns when the high-degree basis is ill-conditioned', () => {
    const issues = polyModule.validateParams?.({ degree: 15, nTrain: 30, noise: 0.5, seed: 42 }) ?? [];
    expect(issues.some((s) => /ill-conditioned|condition number/i.test(s))).toBe(true);
  });

  it('validateParams stays quiet at a well-conditioned moderate degree', () => {
    const issues = polyModule.validateParams?.({ degree: 5, nTrain: 30, noise: 0.5, seed: 42 }) ?? [];
    expect(issues.length).toBe(0);
  });

  // Plan failure demo: degree 30 → the power-basis Gram is numerically singular
  // → the sandboxed run fails cleanly (non-finite θ → failedAtStep defined).
  it('degree 30 fails cleanly (normal equation numerically singular)', () => {
    const run = computeRun(simulation, { degree: 30, nTrain: 40, nTest: 20, noise: 0.5, seed: 42, fitOn: 'train' }, 10);
    expect(run.telemetry.failedAtStep).toBeDefined();
  });
});
