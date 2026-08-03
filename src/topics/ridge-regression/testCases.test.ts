import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import { simulation, fitRidge, generateData, ridgeModule } from './module';
import { ridgeTestCases } from './testCases';
import type { Params } from '../../engine/types';

// Run one simulation at `lambda`, return the LAST snapshot's metrics (the sweep
// ends at the slider's λ). maxSteps exceeds the snapshot count so the engine
// never flags a spurious step-budget failure.
function runAt(lambda: number, p: Omit<Params, 'lambda'>): Record<string, number> {
  const run = computeRun(simulation, { ...p, lambda }, Math.round(lambda / 0.5) + 4);
  return run.snapshots[run.snapshots.length - 1].metrics;
}

describe('ridge-regression testCases', () => {
  for (const tc of ridgeTestCases) {
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

  // ===== Multi-run comparisons (each needs 2+ computeRun calls) =====

  // Plan spec case 2: increasing λ shrinks coefficients — ‖θ(λ₂)‖ < ‖θ(λ₁)‖ for λ₂ > λ₁.
  // Deterministic (shared seed, same data); the norm is provably strictly decreasing in λ.
  it('increasing λ shrinks the coefficient norm ‖θ‖', () => {
    const base = { n: 25, nFeatures: 3, noise: 0.5, seed: 42 };
    const mSmall = runAt(0.5, base);
    const mLarge = runAt(5, base);
    expect(mLarge.normTheta).toBeLessThan(mSmall.normTheta);
    // and the shrinkage is real, not a numerical no-op: λ=5 cuts ‖θ‖ by > 1%
    expect(mSmall.normTheta - mLarge.normTheta).toBeGreaterThan(0.01);
  });

  // Plan spec case 3: OLS explodes on near-collinear features while ridge stays stable.
  it('OLS (λ = 0) explodes on near-collinear features; ridge λ = 1 is stable', () => {
    const base = { n: 20, nFeatures: 2, noise: 0.05, collinear: true, collinearJitter: 1e-5, seed: 42 };
    const mOls = runAt(0, base);
    const mRidge = runAt(1, base);
    // deterministic thresholds: near-singular data (ε ~ 1e-5) → bounded-but-huge OLS
    expect(Number.isFinite(mOls.normTheta)).toBe(true);
    expect(mOls.normTheta).toBeGreaterThan(1e3);
    expect(mRidge.normTheta).toBeLessThan(10);
    expect(mOls.normTheta / mRidge.normTheta).toBeGreaterThan(100);
  });

  // Plan spec case 4: bias–variance tradeoff averaged over fixed seeds.
  // DEVIATION (flagged in the Task 3 implementation report): with the plan's
  // constraints (d ≤ 3, λ ≤ 10, i.i.d. well-conditioned data) the prediction
  // variance is σ²·d/n — far too small for a measurable test-error dip before
  // bias dominates, so the test error curve is honest-and-monotone here. We
  // assert the robust facts: (a) train error rises with λ, (b) aggressive λ
  // worsens test error (underfit), (c) moderate λ costs almost nothing on test
  // error while visibly shrinking ‖θ‖.
  it('bias–variance: train error rises with λ; large λ underfits (test error ↑)', () => {
    const seeds = [1, 2, 3, 4, 5];
    const lambdas = [0, 0.5, 1, 2, 3, 5, 10];
    const avgTrain = lambdas.map(() => 0);
    const avgTest = lambdas.map(() => 0);
    const avgNorm = lambdas.map(() => 0);
    for (const s of seeds) {
      for (let li = 0; li < lambdas.length; li++) {
        const m = runAt(lambdas[li], { n: 15, nFeatures: 3, noise: 1.5, seed: s });
        avgTrain[li] += m.trainMse / seeds.length;
        avgTest[li] += m.testMse / seeds.length;
        avgNorm[li] += m.normTheta / seeds.length;
      }
    }
    // (a) train error grows as λ grows (ridge fits the train set worse)
    expect(avgTrain[5]).toBeGreaterThan(avgTrain[0] + 0.05);
    // (b) the "too much shrinkage" tail: assert the STRONG signal at λ=10
    //     (net ≈ +0.14 analytically, ~30× the λ=5 net of ≈ +0.005 that sits at
    //     the per-seed MC noise floor — the earlier λ=5-only assertion was
    //     deterministic-but-noise-lucky, fixed per review NIT-1)
    expect(avgTest[6]).toBeGreaterThan(avgTest[0] + 0.05);
    expect(avgTest[6]).toBeGreaterThan(avgTest[5]);
    // (c) moderate λ ≈ free shrinkage on test error while ‖θ‖ drops — the honest
    //     sweet spot for well-conditioned data (the dip regime is collinear/high-d,
    //     covered by the near-collinear case above)
    expect(avgTest[1]).toBeLessThan(avgTest[0] + 0.05);
    expect(avgNorm[5]).toBeLessThan(avgNorm[0]);
  });

  // Plan spec case 3 (ridge side, exact dependence): ridge λ>0 solves EXACTLY
  // collinear data where OLS is singular. Tested at the fit level — computeRun's
  // λ-sweep cannot pass through the singular λ=0 snapshot, so the sweep aborts
  // before reaching λ=1; the closed-form fit itself is well-defined.
  it('ridge λ = 1 produces finite coefficients on exactly collinear data (OLS null)', () => {
    const data = generateData({ n: 20, nFeatures: 2, noise: 0.0, collinear: true, lambda: 1, seed: 42 });
    const theta = fitRidge({ lambda: 1 }, data);
    expect(theta).not.toBeNull();
    const norm = Math.sqrt((theta ?? []).reduce((a, v) => a + v * v, 0));
    expect(norm).toBeGreaterThan(0);
    expect(norm).toBeLessThan(10);
  });

  // ===== validateParams paths (explicit — the engine sandbox never calls it) =====

  it('validateParams flags exactly collinear features at λ = 0 (OLS singular)', () => {
    const issues = ridgeModule.validateParams?.({ n: 20, nFeatures: 2, noise: 0, collinear: true, lambda: 0, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /collinear/i.test(s))).toBe(true);
  });

  it('validateParams does NOT flag collinear features when λ > 0 (ridge restores invertibility)', () => {
    const issues = ridgeModule.validateParams?.({ n: 20, nFeatures: 2, noise: 0, collinear: true, lambda: 1, seed: 42 }) ?? [];
    expect(issues.some((s) => /collinear/i.test(s))).toBe(false);
  });

  it('validateParams flags n < d+1 at λ = 0 (underdetermined, XᵀX singular)', () => {
    const issues = ridgeModule.validateParams?.({ n: 2, nFeatures: 3, noise: 0, lambda: 0, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /underdetermined|rank-deficient/i.test(s))).toBe(true);
  });

  it('validateParams does NOT flag n < d+1 when λ > 0 (ridge handles underdetermined)', () => {
    const issues = ridgeModule.validateParams?.({ n: 2, nFeatures: 3, noise: 0, lambda: 1, seed: 42 }) ?? [];
    expect(issues.some((s) => /underdetermined|rank-deficient/i.test(s))).toBe(false);
  });

  it('validateParams rejects a negative λ', () => {
    const issues = ridgeModule.validateParams?.({ n: 20, nFeatures: 2, noise: 0.5, lambda: -1, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /non-negative|λ|lambda/i.test(s))).toBe(true);
  });
});
