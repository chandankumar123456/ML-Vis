import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import {
  simulation, sigmoid, softplus, generateData, ceLoss, ceGradient,
  predictProb, predictClass, train, classifyByParams, logisticModule,
} from './module';
import { logisticTestCases } from './testCases';

describe('logistic-regression testCases', () => {
  for (const tc of logisticTestCases) {
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

  // ===== Plan case 2 — unit level: the sigmoid maps R → (0,1) =====

  it('sigmoid maps to (0,1): σ(0) = 0.5, σ(10) ≈ 1, σ(−10) ≈ 0', () => {
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(10)).toBeGreaterThan(0.9999);
    expect(sigmoid(10)).toBeLessThan(1);
    expect(sigmoid(-10)).toBeLessThan(1e-4);
    expect(sigmoid(-10)).toBeGreaterThan(0);
    // asymptotic, never attained — the saturation trap the failure demo leans on
    expect(sigmoid(30)).toBeLessThan(1);
    expect(sigmoid(30)).toBeGreaterThan(1 - 1e-12);
    expect(sigmoid(-30)).toBeGreaterThan(0);
    expect(sigmoid(-30)).toBeLessThan(1e-12);
    // stable piecewise form agrees with the naive formula where both are safe
    const z = 2.5;
    expect(sigmoid(z)).toBeCloseTo(1 / (1 + Math.exp(-z)), 12);
  });

  it('softplus is exact and stable: CE per point never produces NaN/Infinity', () => {
    expect(softplus(0)).toBeCloseTo(Math.log(2), 12);
    expect(softplus(1000)).toBeCloseTo(1000, 6); // no overflow → ≈ z
    expect(softplus(-1000)).toBeCloseTo(0, 6);
    // cePoint at extreme log-odds stays finite and sensible
    expect(1 * softplus(-30) + 0 * softplus(30)).toBeCloseTo(Math.log(1 + Math.exp(-30)), 6);
  });

  // ===== Plan case 3 — trajectory: CE monotone non-increasing =====

  it('cross-entropy loss is monotone non-increasing over epochs (+1e-9 wobble)', () => {
    const run = computeRun(
      simulation,
      { nPerClass: 30, margin: 2, noise: 1, lr: 0.3, epochs: 300, init: 'zero', seed: 42 },
      320,
    );
    const ces = run.snapshots.map((s) => s.metrics.ce);
    expect(ces.length).toBe(301); // epoch 0..300
    for (let i = 1; i < ces.length; i++) {
      expect(ces[i], `CE rose at epoch ${i}: ${ces[i - 1]} → ${ces[i]}`)
        .toBeLessThanOrEqual(ces[i - 1] + 1e-9);
    }
    // and the decrease is real, not a no-op: the run ends far below the ln 2 start
    expect(ces[0]).toBeCloseTo(Math.log(2), 3);
    expect(ces[ces.length - 1]).toBeLessThan(0.55);
  });

  // ===== Plan case 4 — calibration: threshold agreement + the (ŷ−y) identity =====

  it('probability calibration: p > 0.5 ⟺ prediction, on every training point', () => {
    const p = { nPerClass: 30, margin: 2, noise: 1, lr: 0.3, epochs: 300, init: 'random', seed: 42 };
    const { theta, data } = train(p);
    for (let i = 0; i < data.xs.length; i++) {
      const prob = predictProb(theta, data.xs[i]);
      expect(predictClass(theta, data.xs[i]), `point ${i}`).toBe(prob > 0.5 ? 1 : 0);
      // weak calibration: for confident predictions, threshold and label agree
      if (prob > 0.6 || prob < 0.4) {
        expect(data.ys[i], `misclassified confident point ${i} (p = ${prob})`).toBe(prob > 0.5 ? 1 : 0);
      }
    }
  });

  it('classifyByParams returns the CURRENT-STEP class index (snapshot weights win); falls back to the trained model', () => {
    const p = { nPerClass: 20, margin: 2, noise: 1, lr: 0.3, epochs: 200, init: 'random', seed: 42 };
    // The decision-boundary view merges snapshot.algorithm into params before
    // every grid call, so params.w1/w2/b are the CURRENT epoch's weights.
    const atEpoch = { ...p, epoch: 137, w1: 1.5, w2: -0.8, b: 0.3 };
    for (const [x1, x2] of [[-3, 0], [0, 0], [3, 0], [1.7, -1.2], [0, 2.2]] as const) {
      const z = 1.5 * x1 + (-0.8) * x2 + 0.3;
      expect(classifyByParams(x1, x2, atEpoch), `point (${x1}, ${x2})`).toBe(z > 0 ? 1 : 0);
    }
    // No snapshot yet → deterministic final trained model (memoized per params).
    const { theta } = train(p);
    expect(classifyByParams(-3, 0, p)).toBe(theta[0] * -3 + theta[2] > 0 ? 1 : 0);
    expect(classifyByParams(3, 0, p)).toBe(theta[0] * 3 + theta[2] > 0 ? 1 : 0);
    // memoization must not leak across param changes
    const p2 = { ...p, seed: 43, epochs: 20 };
    const th2 = train(p2).theta;
    expect(classifyByParams(0, 0, p2)).toBe(th2[1] * 0 + th2[2] > 0 ? 1 : 0);
  });

  // ===== Extra: the famous (ŷ − y)x gradient vs finite differences =====

  it('∂L/∂w = (1/n)Σ(ŷ−y)x matches central finite differences', () => {
    const p = { nPerClass: 20, margin: 2, noise: 1, lr: 0.3, epochs: 300, init: 'random', seed: 42 };
    const data = generateData(p);
    // a NON-converged operating point in ORIGINAL coordinates → non-trivial gradient
    const theta = [0.5, -0.4, 0.2];
    const eps = 1e-5;
    const g = ceGradient(theta, data);
    for (let j = 0; j < theta.length; j++) {
      const t1 = theta.slice(); t1[j] += eps;
      const t2 = theta.slice(); t2[j] -= eps;
      const num = (ceLoss(t1, data) - ceLoss(t2, data)) / (2 * eps);
      expect(Math.abs(num - g[j]), `gradient component ${j}: analytic ${g[j]} vs finite-diff ${num}`)
        .toBeLessThan(1e-4);
    }
  });

  // ===== validateParams paths (explicit — the engine sandbox never calls it) =====

  it('validateParams flags nPerClass < 2', () => {
    const issues = logisticModule.validateParams?.({ nPerClass: 1, margin: 2, noise: 1, lr: 0.3, epochs: 100, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /nPerClass/i.test(s))).toBe(true);
  });

  it('validateParams flags lr ≤ 0', () => {
    const issues = logisticModule.validateParams?.({ nPerClass: 20, margin: 2, noise: 1, lr: 0, epochs: 100, seed: 42 }) ?? [];
    expect(issues.some((s) => /learning rate|positive/i.test(s))).toBe(true);
  });

  it('validateParams flags epochs < 1', () => {
    const issues = logisticModule.validateParams?.({ nPerClass: 20, margin: 2, noise: 1, lr: 0.3, epochs: 0, seed: 42 }) ?? [];
    expect(issues.length).toBeGreaterThan(0);
    expect(issues.some((s) => /epochs/i.test(s))).toBe(true);
  });

  it('validateParams flags non-positive noise (cluster spread)', () => {
    const issues = logisticModule.validateParams?.({ nPerClass: 20, margin: 2, noise: 0, lr: 0.3, epochs: 100, seed: 42 }) ?? [];
    expect(issues.some((s) => /noise/i.test(s))).toBe(true);
  });

  it('validateParams accepts a sensible default configuration', () => {
    const issues = logisticModule.validateParams?.({ nPerClass: 20, margin: 2, noise: 1, lr: 0.3, epochs: 200, seed: 42 }) ?? [];
    expect(issues.length).toBe(0);
  });
});
