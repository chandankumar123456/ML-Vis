import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import { simulation, nbModule, classifyNB, posteriorOf, fitGaussianNB, generateData } from './module';
import { nbTestCases } from './testCases';

describe('naive-bayes testCases', () => {
  for (const tc of nbTestCases) {
    it(tc.name, () => {
      const run = computeRun(simulation, tc.params, tc.maxSteps ?? 500);
      if (tc.expect.converged !== undefined) {
        if (tc.expect.converged) {
          expect(run.telemetry.failedAtStep).toBeUndefined();
        } else {
          expect(run.telemetry.failedAtStep).toBeDefined();
        }
      }
      const last = run.snapshots[run.snapshots.length - 1];
      if (tc.expect.finalMetrics) {
        const m = last.metrics;
        for (const [k, pred] of Object.entries(tc.expect.finalMetrics)) {
          // finalMetrics may hold predicates OR plain numbers (type union) — dispatch
          if (typeof pred === 'function') expect(pred(m[k]), `metric ${k} failed for ${tc.name}`).toBe(true);
          else expect(m[k]).toBeCloseTo(pred, 6);
        }
      }
      if (tc.expect.finalAlgorithm) {
        const a = last.algorithm;
        for (const [k, pred] of Object.entries(tc.expect.finalAlgorithm)) {
          // finalAlgorithm may hold predicates OR plain values (type union) — dispatch
          if (typeof pred === 'function') expect(pred(a[k]), `algorithm ${k} failed for ${tc.name}`).toBe(true);
          else expect(a[k]).toEqual(pred);
        }
      }
    });
  }

  // ---- validateParams path tests (the engine sandbox does NOT call
  // validateParams — the UI does before a run, so the path is tested explicitly).

  it('validateParams accepts a valid parameter set', () => {
    const issues = nbModule.validateParams?.({ nClasses: 2, nPerClass: 25, correlation: 0.9, smoothing: 0.1, seed: 42, queryX1: 2.4, queryX2: 2 }) ?? [];
    expect(issues).toEqual([]);
  });

  it('validateParams rejects nClasses outside {2,3}', () => {
    const issues = nbModule.validateParams?.({ nClasses: 4, nPerClass: 25, correlation: 0.5, smoothing: 0.1, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /nClasses/i.test(s))).toBe(true);
  });

  it('validateParams rejects a negative smoothing α', () => {
    const issues = nbModule.validateParams?.({ nClasses: 2, nPerClass: 25, correlation: 0.5, smoothing: -0.5, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /smoothing/i.test(s))).toBe(true);
  });

  it('validateParams rejects correlation outside [0, 1)', () => {
    const tooHigh = nbModule.validateParams?.({ nClasses: 2, nPerClass: 25, correlation: 1.2, smoothing: 0.1, seed: 42 }) ?? [];
    expect(tooHigh.some((s) => /correlation/i.test(s))).toBe(true);
    const negative = nbModule.validateParams?.({ nClasses: 2, nPerClass: 25, correlation: -0.1, smoothing: 0.1, seed: 42 }) ?? [];
    expect(negative.some((s) => /correlation/i.test(s))).toBe(true);
  });

  it('validateParams rejects nPerClass < 1', () => {
    const issues = nbModule.validateParams?.({ nClasses: 2, nPerClass: 0, correlation: 0.5, smoothing: 0.1, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /nPerClass/i.test(s))).toBe(true);
  });

  // ---- direct-contract checks (independent of the run harness)

  it('classifyNB is deterministic and matches the Gaussian posterior argmax', () => {
    const params = { nClasses: 2, nPerClass: 25, correlation: 0.9, smoothing: 0.1, seed: 42, queryX1: 2.4, queryX2: 2 };
    const data = generateData(params);
    const fits = fitGaussianNB(data, 2, 0.1);
    const post = posteriorOf(fits, 2.4, 2);
    const argmax = post.indexOf(Math.max(...post));
    expect(classifyNB(2.4, 2, params)).toBe(argmax);
    expect(classifyNB(2.4, 2, params)).toBe(classifyNB(2.4, 2, params)); // deterministic
  });
});
