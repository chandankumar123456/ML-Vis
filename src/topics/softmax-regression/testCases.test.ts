import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import type { Params } from '../../engine/types';
import { getClassifier } from '../../registry/viewRegistry';
import { simulation, softmaxModule, register, generateData, standardize, toStandard, fromStandard,
  softmax, categoricalCE, categoricalCEGrad, predictClass, classifyPoint, clusterCenters } from './module';
import { softmaxTestCases } from './testCases';

describe('softmax-regression testCases', () => {
  // Wave-2 registry contract: register() wires classifyPoint into the view
  // registry so the decision-boundary view can paint 3-class regions.
  it('register() exposes the topic classifier to the view registry', () => {
    register();
    const clf = getClassifier('softmax-regression');
    expect(clf).toBeTypeOf('function');
    const cls = clf!(0, 0, { nPerClass: 20, margin: 3, learningRate: 0.1, epochs: 300, seed: 42 });
    expect([0, 1, 2]).toContain(cls);
  });

  for (const tc of softmaxTestCases) {
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

  // ===== Plan case 1: Σ softmax(z) = 1 for random z (deterministic seeded z's) =====
  // toBeCloseTo 1e-12 per plan. The stable softmax (max-shift) computes the same
  // normalized vector as the textbook form, so the sum is 1 up to IEEE rounding.
  it('softmax outputs sum to 1 for random z (1e-12)', () => {
    let a = 1234567 >>> 0; // tiny inline LCG so the "random" z's are deterministic
    const rnd = () => { a = (a * 1664525 + 1013904223) >>> 0; return a / 4294967296; };
    for (let trial = 0; trial < 50; trial++) {
      const z = Array.from({ length: 3 + Math.floor(rnd() * 5) }, () => (rnd() - 0.5) * 40); // incl. negative logits
      const p = softmax(z);
      const sum = p.reduce((acc, v) => acc + v, 0);
      expect(Math.abs(sum - 1)).toBeLessThan(1e-12);
    }
  });

  // ===== Plan case 2: softmax(z) = softmax(z + c) (shift invariance) =====
  // Algebraic identity: e^{z_k+c}/Σe^{z_j+c} = e^{z_k}/Σe^{z_j}. The max-shift
  // implementation is exactly the c = −max z case, so this also guards the
  // stability trick itself. Tolerance 1e-10: for c = 1e6 the shifted logits are
  // 1e6-scale floats, so v + c − m carries ~1e-12 float rounding — the residual
  // (≤ 4.4e-12 here) is pure IEEE rounding, not a math error.
  it('softmax is invariant to constant shifts (stable form)', () => {
    const z = [2.5, -1.2, 0.4, 3.1];
    for (const c of [5, 100, -7, 1e6]) {
      const p0 = softmax(z);
      const p1 = softmax(z.map((v) => v + c));
      for (let k = 0; k < z.length; k++) {
        expect(Math.abs(p0[k] - p1[k])).toBeLessThan(1e-10);
      }
    }
  });

  // ===== Plan case 4: numeric check of ∂L/∂w_k = Σ(ŷ_k − 1{y=k})x =====
  // Central finite differences with h = 1e-5 on FIXED data and FIXED weights —
  // no RNG in the check (deterministic). Tolerance: relative 1e-4-ish; the mean
  // CE and its analytic gradient are O(1), so absolute 1e-6 ≈ relative 1e-5.
  it('categorical CE gradient matches central finite differences', () => {
    // Fixed 3-class / 2-feature data (deterministic — the check never samples).
    const xs = [
      [0.3, -0.2], [-0.7, 0.4], [0.9, 0.1], [-0.4, -0.8], [0.5, 0.6], [-0.1, 0.3],
    ];
    const ys = [0, 1, 2, 0, 1, 2];
    const data = { xs, ys, n: xs.length, d: 2, K: 3 };
    const W = [
      [0.4, -0.3],
      [-0.2, 0.5],
      [0.1, 0.2],
    ];
    const b = [0.05, -0.1, 0.15];
    const { dW, db } = categoricalCEGrad(data, W, b);
    const h = 1e-5;
    const perturb = (k: number, j: number, s: number, onW: boolean): number => {
      const Wp = W.map((r) => [...r]);
      const bp = [...b];
      if (onW) Wp[k][j] += s * h; else bp[k] += s * h;
      return categoricalCE(data, Wp, bp);
    };
    for (let k = 0; k < 3; k++) {
      for (let j = 0; j < 2; j++) {
        const num = (perturb(k, j, 1, true) - perturb(k, j, -1, true)) / (2 * h);
        expect(Math.abs(num - dW[k][j])).toBeLessThan(1e-6);
      }
      const numB = (perturb(k, 0, 1, false) - perturb(k, 0, -1, false)) / (2 * h);
      expect(Math.abs(numB - db[k])).toBeLessThan(1e-6);
    }
  });

  // ===== Classifier consistency (plan: boundary must reflect the trained model) =====
  // classifyPoint re-derives the FINAL weights deterministically from params
  // (seeded data + deterministic GD), so it is bit-consistent with the run's last
  // snapshot. Sanity: every cluster MEAN is classified to its own class.
  it('classifier is consistent with the trained model (cluster means → own class)', () => {
    const p = { nPerClass: 20, margin: 3, learningRate: 0.1, epochs: 300, seed: 42 };
    const centers = clusterCenters(3);
    centers.forEach((c, k) => {
      expect(classifyPoint(c[0], c[1], p)).toBe(k);
    });
  });

  // Snapshot-aware path: the DecisionBoundary merges snapshot.algorithm into
  // params before calling the classifier; with w11..b3 present classifyPoint must
  // reproduce the run's own per-point argmax exactly (the boundary then animates
  // with the epoch and is bit-consistent with the snapshot's accuracy metric).
  it('classifier honors merged snapshot weights (boundary == run argmax at that step)', () => {
    const run = computeRun(simulation, { nPerClass: 20, margin: 3, learningRate: 0.1, epochs: 50, seed: 42 }, 80);
    const data = generateData({ nPerClass: 20, margin: 3, seed: 42 });
    // scrub to a mid-training snapshot: merged params = topic params + algorithm
    const snap = run.snapshots[25];
    const merged: Params = { nPerClass: 20, margin: 3, learningRate: 0.1, epochs: 50, seed: 42, ...snap.algorithm };
    for (const row of data.xs) {
      expect(classifyPoint(row[0], row[1], merged)).toBe(predictClass(
        [[merged.w11 as number, merged.w12 as number], [merged.w21 as number, merged.w22 as number], [merged.w31 as number, merged.w32 as number]],
        [merged.b1 as number, merged.b2 as number, merged.b3 as number],
        row,
      ));
    }
  });

  // ===== Standardization round trip (toStandard/fromStandard inverse) =====
  it('toStandard/fromStandard round-trips exactly (Wave-1 corrected bias pair)', () => {
    const data = generateData({ nPerClass: 10, margin: 3, seed: 42 });
    const sc = standardize(data);
    const W = [[1.5, -0.7], [0.2, 2.1], [-1.1, 0.4]];
    const b = [0.3, -0.2, 0.9];
    const { Wt, bt } = toStandard(W, b, sc);
    const back = fromStandard(Wt, bt, sc);
    for (let k = 0; k < 3; k++) {
      for (let j = 0; j < 2; j++) expect(Math.abs(back.W[k][j] - W[k][j])).toBeLessThan(1e-12);
      expect(Math.abs(back.b[k] - b[k])).toBeLessThan(1e-12);
    }
  });

  // ===== Trajectory: CE monotone-ish decrease + accuracy floor on separable data =====
  // Full-batch GD with exact softmax gradient on well-separated standardized data
  // decreases CE at every epoch for this deterministic run (asserted weakly: final
  // ≪ initial, and CE at the last epoch is below every early-epoch value checked).
  it('CE decreases over training on separable data', () => {
    const run = computeRun(simulation, { nPerClass: 20, margin: 3, learningRate: 0.1, epochs: 100, seed: 42 }, 150);
    const ces = run.snapshots.map((s) => s.metrics.ce);
    expect(ces[ces.length - 1]).toBeLessThan(ces[0]);          // final < epoch-1
    expect(ces[ces.length - 1]).toBeLessThan(ces[10]);         // final < epoch-11
    // First snapshot is AFTER one GD epoch (initialState runs it), so CE is
    // already below the uniform-guess baseline ln 3 ≈ 1.099 — assert the bound,
    // not the pre-training value (which never appears in the run).
    expect(ces[0]).toBeLessThan(Math.log(3));
    expect(ces[0]).toBeGreaterThan(Math.log(3) * 0.9);         // still near-uniform early
    const accs = run.snapshots.map((s) => s.metrics.accuracy);
    expect(accs[accs.length - 1]).toBe(1);                     // converged to perfect
  });

  // ===== validateParams paths (explicit — the engine sandbox never calls it) =====

  it('validateParams flags nPerClass < 2', () => {
    const issues = softmaxModule.validateParams?.({ nPerClass: 1, margin: 3, learningRate: 0.1, epochs: 300, seed: 42 }) ?? [];
    expect(issues.some((s) => /at least 2/i.test(s))).toBe(true);
  });

  it('validateParams flags lr ≤ 0', () => {
    const issues = softmaxModule.validateParams?.({ nPerClass: 20, margin: 3, learningRate: 0, epochs: 300, seed: 42 }) ?? [];
    expect(issues.some((s) => /learning rate must be positive/i.test(s))).toBe(true);
  });

  it('validateParams flags epochs < 1', () => {
    const issues = softmaxModule.validateParams?.({ nPerClass: 20, margin: 3, learningRate: 0.1, epochs: 0, seed: 42 }) ?? [];
    expect(issues.some((s) => /epochs must be at least 1/i.test(s))).toBe(true);
  });

  it('validateParams flags non-positive margin', () => {
    const issues = softmaxModule.validateParams?.({ nPerClass: 20, margin: 0, learningRate: 0.1, epochs: 300, seed: 42 }) ?? [];
    expect(issues.some((s) => /separation must be positive/i.test(s))).toBe(true);
  });

  it('validateParams accepts a valid configuration', () => {
    const issues = softmaxModule.validateParams?.({ nPerClass: 20, margin: 3, learningRate: 0.1, epochs: 300, seed: 42 }) ?? [];
    expect(issues.length).toBe(0);
  });

  // ===== Misc deterministic contract sanity =====
  it('predictClass equals argmax of the probability vector', () => {
    const p = { nPerClass: 5, margin: 3, learningRate: 0.1, epochs: 50, seed: 1 };
    const data = generateData(p);
    const sc = standardize(data);
    const { Wt, bt } = toStandard([[0.2, -0.1], [-0.3, 0.4], [0.1, 0.05]], [0, 0, 0], sc);
    const { W, b } = fromStandard(Wt, bt, sc);
    for (const row of data.xs) {
      const probs = softmax(W.map((wr, k) => wr.reduce((a, w, j) => a + w * row[j], 0) + b[k]));
      let best = 0;
      for (let k = 1; k < probs.length; k++) if (probs[k] > probs[best]) best = k;
      expect(predictClass(W, b, row)).toBe(best);
    }
  });
});
