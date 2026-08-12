// src/topics/svm-soft-margin/testCases.test.ts
// Verified anchors (measured on the deterministic default data, seed 42):
//   • Hard-margin reference (independent exhaustive search, 14400 orientations):
//     margin = 2.211834, supports = [4, 18].
//   • Soft C = 1000 (separable): margin = 2.211849 — rel. diff ≈ 6.6e-6 (< 1%),
//     slack = 0 exactly, supports ⊇ {4, 18}.
//   • Outlier on: margin(C = 0.1) = 2.64 vs margin(C = 1000) = 2.21;
//     slackSum 4.055 → 3.685; outlier slack ξ₉ = 3.27 (≥ 1 ⇒ misclassified).
//   • Slack at C = 1 (outlier on): min ξ = 0, max ξ = 3.565, all non-violated
//     points have y·f ≥ 1 (ξ = 0 exactly).
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import {
  simulation, getPoints, solveSoftMargin, evaluateFit, solveHardMarginReference,
  hinge, svmSoftMarginModule, register, C_GRID,
} from './module';
import { svmSoftTestCases } from './testCases';
import { getClassifier } from '../../registry/viewRegistry';
import type { Params, SimState } from '../../engine/types';

function runAt(C: number, extra: Record<string, unknown> = {}): { run: ReturnType<typeof computeRun>; last: SimState } {
  const params: Params = { C, nPerClass: 10, margin: 1.5, spread: 0.5, seed: 42, ...extra };
  const run = computeRun(simulation, params, 20);
  return { run, last: run.snapshots[run.snapshots.length - 1] };
}

describe('svm-soft-margin testCases', () => {
  for (const tc of svmSoftTestCases) {
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

  // ===== Plan case 1: C → ∞ approximates the hard margin (measured anchors) =====

  it('C = 1000 on separable data matches the independent hard-margin reference (margin within 1%, same supports)', () => {
    const points = getPoints({ nPerClass: 10, margin: 1.5, spread: 0.5, outlier: false, seed: 42 });
    const ref = solveHardMarginReference(points);
    expect(ref).not.toBeNull();
    const refFit = ref as NonNullable<typeof ref>;
    const fit = solveSoftMargin(points, 1000);
    const evals = evaluateFit(points, fit.w1, fit.w2, fit.b);

    // margin within 1% (measured rel. diff ≈ 6.6e-6)
    expect(Math.abs(fit.margin - refFit.margin) / refFit.margin).toBeLessThan(0.01);
    // weight direction matches within 2° (measured ≈ 0.017°)
    const dot = (fit.w1 * refFit.w1 + fit.w2 * refFit.w2) / (fit.s * refFit.s);
    const angle = Math.acos(Math.min(1, Math.max(-1, dot))) * 180 / Math.PI;
    expect(angle).toBeLessThan(2);
    // zero slack on separable data at C = 1000 → no violated points
    expect(fit.slackSum).toBe(0);
    // the hard-margin support vectors are among the soft-margin supports
    const supIdx = evals.map((e, i) => (e.support ? i : -1)).filter((i) => i >= 0);
    for (const s of refFit.supports) expect(supIdx).toContain(s);
    expect(supIdx.length).toBeGreaterThanOrEqual(refFit.supports.length);
  });

  // ===== Plan case 2: small C allows misclassification (measured anchors) =====

  it('with an outlier, small C ignores it (wider margin, more slack); large C tightens', () => {
    const out = { outlier: true, outlierStrength: 3 };
    const small = runAt(0.1, out).last.metrics;
    const large = runAt(1000, out).last.metrics;

    // measured: margin 2.64 (C=0.1) vs 2.21 (C=1000) — gap ≈ 0.43
    expect(small.margin).toBeGreaterThan(large.margin + 0.15);
    // measured: Σξ 4.055 → 3.685 (large C reduces total slack)
    expect(large.slackSum).toBeLessThan(small.slackSum as number);
    // the outlier is ignored at small C: violated, and misclassified (ξ ≥ 1)
    const fit = solveSoftMargin(getPoints({ ...out, seed: 42 }), 0.1);
    const evals = evaluateFit(getPoints({ ...out, seed: 42 }), fit.w1, fit.w2, fit.b);
    expect(evals[9].xi).toBeGreaterThanOrEqual(1);   // outlier point (last class-0)
    expect(evals.filter((e) => e.violated).length).toBeGreaterThanOrEqual(1);
  });

  // the qualitative claim holds across seeds (not a seed-42 fluke)
  it('small-C-wider-margin holds across several seeds', () => {
    for (const seed of [1, 2, 3, 42, 99]) {
      const p = { nPerClass: 10, margin: 1.5, spread: 0.5, seed, outlier: true, outlierStrength: 3 };
      const small = solveSoftMargin(getPoints({ ...p }), 0.1);
      const large = solveSoftMargin(getPoints({ ...p }), 1000);
      expect(small.margin, `seed ${seed}`).toBeGreaterThan(large.margin + 0.1);
      expect(large.slackSum, `seed ${seed}`).toBeLessThan(small.slackSum);
    }
  });

  // ===== Plan case 3: hinge loss is zero for confident-correct points =====

  it('hinge loss is exactly 0 when y·f ≥ 1 (hand-computed values)', () => {
    expect(hinge(1)).toBe(0);
    expect(hinge(1.5)).toBe(0);
    expect(hinge(2)).toBe(0);
    expect(hinge(100)).toBe(0);
    // crafted non-confident values: hinge = 1 − y·f exactly
    expect(hinge(0.999)).toBeCloseTo(0.001, 12);
    expect(hinge(0.5)).toBe(0.5);
    expect(hinge(0)).toBe(1);
    expect(hinge(-1)).toBe(2);
    expect(hinge(-3)).toBe(4);
  });

  it('confident-correct points in a real fit have ξ = 0 (measured)', () => {
    const m = runAt(1, { outlier: true }).last.metrics;
    const points = getPoints({ outlier: true, seed: 42 });
    const evals = evaluateFit(points, m.w1 as number, m.w2 as number, m.b as number);
    for (let i = 0; i < evals.length; i++) {
      if (evals[i].yf >= 1) expect(evals[i].xi, `point ${i}`).toBe(0);
    }
  });

  // ===== Plan case 4: slack variables =====

  it('slack variables: ξᵢ ≥ 0 everywhere, and ξᵢ = 0 for every non-violated point', () => {
    const m = runAt(1, { outlier: true }).last.metrics;
    const points = getPoints({ outlier: true, seed: 42 });
    const evals = evaluateFit(points, m.w1 as number, m.w2 as number, m.b as number);
    for (let i = 0; i < evals.length; i++) {
      expect(evals[i].xi, `point ${i} ξ ≥ 0`).toBeGreaterThanOrEqual(-1e-12);
    }
    const nonViolated = evals.filter((e) => !e.violated);
    expect(nonViolated.length).toBeGreaterThan(0);
    for (const e of nonViolated) {
      expect(e.xi).toBeLessThanOrEqual(1e-6);   // the "ξ = 0" claim, up to XI_TOL
      expect(e.yf).toBeGreaterThanOrEqual(1 - 1e-6); // ξ = max(0, 1 − y·f) ⇒ y·f ≥ 1
    }
    // and the violated points are exactly the ones with ξ > 0
    expect(evals.filter((e) => e.violated).length).toBe(m.violatedCount);
  });

  // ===== Sweep semantics =====

  it('C-sweep: one snapshot per log-grid value below the slider C, then exactly C last', () => {
    const { run, last } = runAt(50);
    expect(run.snapshots.map((s) => s.metrics.C)).toEqual([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 50]);
    expect(last.metrics.C).toBe(50);
    expect(run.telemetry.failedAtStep).toBeUndefined();

    const max = runAt(1000);
    expect(max.run.snapshots).toHaveLength(C_GRID.length);
    expect(max.last.metrics.C).toBe(1000);
  });

  // ===== Determinism =====

  it('deterministic: identical params produce identical runs', () => {
    const p = { C: 1, nPerClass: 10, margin: 1.5, spread: 0.5, outlier: true, seed: 42 };
    const r1 = computeRun(simulation, p, 20);
    const r2 = computeRun(simulation, p, 20);
    expect(r1.snapshots.map((s) => s.metrics)).toEqual(r2.snapshots.map((s) => s.metrics));
    expect(r1.snapshots.map((s) => s.algorithm)).toEqual(r2.snapshots.map((s) => s.algorithm));
  });

  // ===== Classifier wiring =====

  it('classifier correctness at each C: reads w1/w2/b from snapshot.algorithm', () => {
    register(); // idempotent — registers the classifier into the registry
    const classifier = getClassifier('svm-soft-margin');
    expect(classifier).toBeDefined();
    const c = classifier as (x: number, y: number, params: Params) => number;
    const p: Params = { C: 1, nPerClass: 10, margin: 1.5, spread: 0.5, outlier: true, seed: 42 };
    const run = computeRun(simulation, p, 20);
    const points = getPoints(p);
    for (const snap of run.snapshots) {
      const merged = { ...p, ...snap.algorithm };
      const w1 = snap.algorithm.w1 as number;
      const w2 = snap.algorithm.w2 as number;
      const b = snap.algorithm.b as number;
      for (const pt of points) {
        expect(c(pt.x, pt.y, merged)).toBe(w1 * pt.x + w2 * pt.y + b >= 0 ? 1 : 0);
      }
      // far-left → class 0, far-right → class 1 (boundary separates the clusters)
      expect(c(-10, 0, merged)).toBe(0);
      expect(c(10, 0, merged)).toBe(1);
      // every snapshot carries the current hyperplane
      expect(snap.algorithm.mode).toBe('svm-soft-margin');
      expect(Number.isFinite(w1)).toBe(true);
      expect(Number.isFinite(b)).toBe(true);
    }
  });

  // ===== validateParams paths (the engine sandbox never calls it) =====

  it('validateParams rejects C = 0 / negative / out of log-slider range', () => {
    const base = { nPerClass: 10, margin: 1.5, spread: 0.5, seed: 42 };
    const zero = svmSoftMarginModule.validateParams?.({ ...base, C: 0 }) ?? [];
    expect(zero.some((s) => /C must be > 0/i.test(s))).toBe(true);
    const neg = svmSoftMarginModule.validateParams?.({ ...base, C: -1 }) ?? [];
    expect(neg.length).toBeGreaterThan(0);
    const tiny = svmSoftMarginModule.validateParams?.({ ...base, C: 0.005 }) ?? [];
    expect(tiny.some((s) => /log-slider range/i.test(s))).toBe(true);
    const huge = svmSoftMarginModule.validateParams?.({ ...base, C: 5000 }) ?? [];
    expect(huge.some((s) => /log-slider range/i.test(s))).toBe(true);
  });

  it('validateParams rejects degenerate data (nPerClass < 2, margin ≤ 0, spread ≤ 0, negative outlierStrength)', () => {
    const base = { C: 1, margin: 1.5, spread: 0.5, seed: 42 };
    expect((svmSoftMarginModule.validateParams?.({ ...base, nPerClass: 1 }) ?? []).length).toBeGreaterThan(0);
    expect((svmSoftMarginModule.validateParams?.({ ...base, margin: 0 }) ?? []).length).toBeGreaterThan(0);
    expect((svmSoftMarginModule.validateParams?.({ ...base, spread: 0 }) ?? []).length).toBeGreaterThan(0);
    expect((svmSoftMarginModule.validateParams?.({ ...base, outlierStrength: -1 }) ?? []).length).toBeGreaterThan(0);
  });

  it('validateParams accepts the default parameters', () => {
    const issues = svmSoftMarginModule.validateParams?.({ C: 1, nPerClass: 10, margin: 1.5, spread: 0.5, outlier: false, outlierStrength: 3, seed: 42 }) ?? [];
    expect(issues).toEqual([]);
  });

  // ===== Solver internals (determinism + termination, documented in module.ts) =====

  it('solver is deterministic and C > 0 is enforced', () => {
    const points = getPoints({ seed: 42 });
    const a = solveSoftMargin(points, 7);
    const b = solveSoftMargin(points, 7);
    expect(a).toEqual(b);
    expect(() => solveSoftMargin(points, 0)).toThrow(/C must be > 0/);
  });
});
