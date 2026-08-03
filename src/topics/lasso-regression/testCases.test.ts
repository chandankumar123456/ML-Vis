// src/topics/lasso-regression/testCases.test.ts
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import { simulation, lassoModule, generateData, standardize } from './module';
import { lassoTestCases } from './testCases';
import { mean, transpose, matMul } from '../../lib/math/linAlg';
import type { LassoData } from './module';

/** Gauss–Jordan inversion with partial pivoting (local copy — self-contained ridge solver). */
function matInverse(A: number[][]): number[][] {
  const n = A.length;
  const aug = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(aug[r][col]) > Math.abs(aug[piv][col])) piv = r;
    [aug[col], aug[piv]] = [aug[piv], aug[col]];
    const dv = aug[col][col];
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= dv;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = aug[r][col];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) aug[r][j] -= f * aug[col][j];
    }
  }
  return aug.map((row) => row.slice(n));
}

/**
 * Ridge closed form on STANDARDIZED features with an unpenalized intercept (mirrors the
 * lasso convention: features z-scored, bias free). θ̃ = (ZᵀZ + λI)⁻¹ Zᵀ y_c, b = ȳ.
 * Self-contained on purpose: the ridge topic is written in parallel and must not be imported.
 */
function fitRidgeStandardized(data: LassoData, lambda: number): { thetaTilde: number[]; b: number } {
  const sc = standardize(data);
  const d = data.d;
  const Z = data.xs.map((row) => row.map((x, j) => (x - sc.mu[j]) / sc.sigma[j]));
  const ybar = mean(data.ys);
  const yc = data.ys.map((y) => y - ybar);
  const Zt = transpose(Z);
  const ZtZ = matMul(Zt, Z);
  for (let j = 0; j < d; j++) ZtZ[j][j] += lambda;
  const Zty = matMul(Zt, yc.map((v) => [v])).map((r) => r[0]);
  const thetaTilde = matMul(matInverse(ZtZ), Zty.map((v) => [v])).map((r) => r[0]);
  return { thetaTilde, b: ybar };
}

describe('lasso-regression testCases', () => {
  for (const tc of lassoTestCases) {
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

  // Plan spec: validateParams is exercised EXPLICITLY (the engine sandbox never calls it).
  it('validateParams flags n < d + 1 (underdetermined system)', () => {
    const issues = lassoModule.validateParams?.({ n: 4, nFeatures: 6, noise: 0.5, lambda: 0.5, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /underdetermined/i.test(s))).toBe(true);
  });

  it('validateParams flags a negative λ', () => {
    const issues = lassoModule.validateParams?.({ n: 25, nFeatures: 4, lambda: -1, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /non-negative/i.test(s))).toBe(true);
  });

  it('validateParams accepts a healthy configuration', () => {
    const issues = lassoModule.validateParams?.({ n: 25, nFeatures: 4, noise: 0.5, lambda: 0.5, seed: 42 }) ?? [];
    expect(issues).toEqual([]);
  });

  // Plan case 3 (ridge half): same data, same λ — ridge's closed form keeps |w₄| > 0
  // while the lasso run (asserted above in the tc loop) zeroes w₄ exactly.
  it('ridge keeps small coefficients nonzero while lasso zeroes them (same data, same λ)', () => {
    const params = { n: 40, nFeatures: 4, noise: 0.5, lambda: 1.5, seed: 42 };
    const run = computeRun(simulation, params, 2000);
    const last = run.snapshots[run.snapshots.length - 1].metrics;
    expect(Math.abs(last.w4)).toBeLessThan(1e-12); // lasso: exactly 0 (soft-threshold)
    expect(last.nNonzero).toBe(3);                 // features 1–3 survive
    const { thetaTilde } = fitRidgeStandardized(generateData(params), 1.5);
    expect(Math.abs(thetaTilde[3])).toBeGreaterThan(0.01); // ridge: nonzero standardized w4
  });

  // Plan case 4: monotone objective decrease — the signature of coordinate descent.
  // The standard runner only inspects the final snapshot, so assert on the trajectory.
  it('coordinate descent decreases the objective monotonically per sweep', () => {
    const params = { n: 40, nFeatures: 4, noise: 0.5, lambda: 0.5, seed: 42 };
    const run = computeRun(simulation, params, 2000);
    expect(run.telemetry.failedAtStep).toBeUndefined();
    const objs = run.snapshots.map((s) => s.metrics.objective);
    // small epsilon absorbs float wobble; exact coordinate minimization is monotone
    for (let i = 1; i < objs.length; i++) {
      expect(objs[i], `objective non-monotone at snapshot ${i} (${objs[i - 1]} → ${objs[i]})`).toBeLessThanOrEqual(objs[i - 1] + 1e-9);
    }
    expect(objs[objs.length - 1]).toBeLessThan(objs[0]);
  });
});
