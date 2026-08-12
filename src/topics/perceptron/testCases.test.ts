// src/topics/perceptron/testCases.test.ts
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import {
  simulation, perceptronModule, register, buildPlan, getData,
  perceptronUpdate, classifyByParams, OSCILLATION_CAP, MAX_UPDATES,
} from './module';
import { perceptronTestCases } from './testCases';
import { getClassifier } from '../../registry/viewRegistry';

// Measured anchors (verified by running the simulator — every number below was
// printed by the module before being asserted):
//   default separable (nPerClass 20, margin 1.2, noise 0.5, η 1, zero init, seed 42):
//     R = 2.244588, converges in 4 UPDATEs (accuracy 1, γ = 0.047207),
//     final w = (2.713533, 0.385045), b = 0, ‖w‖ = 2.740716 (< 2R ≈ 4.489),
//     theorem bound (R·‖w*‖/γ)² = 16981.816 — 4 updates, 4245× under the bound
//     run = 6 snapshots (init + 4 updates + converged), epochMarks [{3:2},{4:0}]
//     max per-update ‖Δw‖ (3D, incl. bias) = 1.958212 ≤ η·√(R²+1) = 2.457270
//   non-separable (separable: false, seed 42): NO exact cycle within MAX_UPDATES
//     on overlapping clouds (measured: none for ANY seed 0..60 — the float state
//     drifts quasi-periodically) → the OSCILLATION_CAP (180) fires:
//     run = 181 snapshots, failedAtStep = 181, failureReason matches /oscillat/i,
//     final mistakesPerEpoch = 4 > 0, accuracy = 0.725 < 1, mistakes = 180,
//     exactly ONE snapshot (update 180) carries the 'Oscillate' timeline stage.
//   case 3 (margin 2.5, noise 0.4): converges in 1 update on point d0
//     d0 = (−2.882465, −0.109210, label −1) → w = (2.882465, 0.109210), b = −1
//   η invariance: η = 1 vs 0.5 → identical update count (4) and 6 snapshots;
//     final weights scale EXACTLY ×2 (η=1 is 2× η=0.5 componentwise).
//   random init (scale 0.1): converges in 2 updates, accuracy 1.
//   seed 7 separable: converges in 23 updates.

const DEFAULT = { nPerClass: 20, margin: 1.2, noise: 0.5, eta: 1, init: 'zero', seed: 42, separable: true };
const NONSEP = { nPerClass: 20, margin: 1.2, noise: 0.5, eta: 1, init: 'zero', seed: 42, separable: false };
const CASE3 = { nPerClass: 20, margin: 2.5, noise: 0.4, eta: 1, init: 'zero', seed: 42, separable: true };

describe('perceptron testCases (data-driven)', () => {
  for (const tc of perceptronTestCases) {
    it(tc.name, () => {
      const run = computeRun(simulation, tc.params, tc.maxSteps ?? 500);
      if (tc.expect.converged !== undefined) {
        if (tc.expect.converged) {
          expect(run.telemetry.failedAtStep).toBeUndefined();
        } else {
          expect(run.telemetry.failedAtStep).toBeDefined();
        }
      }
      if (tc.expect.eventLabels) {
        const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
        for (const lbl of tc.expect.eventLabels) expect(labels).toContain(lbl);
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
});

describe('perceptron: plan case 1 — converges on separable data (measured)', () => {
  it('zero train error within the (R·‖w*‖/γ)² theorem bound — 4 updates ≪ 16982', () => {
    const plan = buildPlan(DEFAULT);
    expect(plan.converged).toBe(true);
    expect(plan.finalAccuracy).toBe(1);
    expect(plan.updates).toBe(4);
    // the theorem bound built from the MEASURED final separator must hold and
    // be loose: updates ≤ (R·‖w*‖/γ)² = 16981.816
    expect(plan.bound).toBeCloseTo(16981.816, 1);
    expect(plan.updates).toBeLessThanOrEqual(plan.bound);
    // the converged separator really separates: geometric margin > 0
    expect(plan.finalGamma).toBeGreaterThan(0);
    expect(plan.finalGamma).toBeCloseTo(0.047207, 4);
    expect(plan.data.R).toBeCloseTo(2.244588, 4);
  });

  it('the run converges cleanly: 6 snapshots, no failedAtStep, final metrics honest', () => {
    const run = computeRun(simulation, DEFAULT, 300);
    expect(run.telemetry.failedAtStep).toBeUndefined();
    expect(run.snapshots).toHaveLength(6); // init + 4 updates + converged snapshot
    const last = run.snapshots[run.snapshots.length - 1];
    expect(last.metrics.accuracy).toBe(1);
    expect(last.metrics.mistakesPerEpoch).toBe(0);
    expect(last.metrics.mistakes).toBe(4);
    expect(last.metrics.normW).toBeCloseTo(2.740716, 4);
    expect(last.metrics.gamma).toBeCloseTo(0.047207, 4);
    // the final clean sweep is a REAL epoch event: the converged timeline stage
    const lastStage = last.timeline[last.timeline.length - 1];
    expect(lastStage).toBe('Converge');
    // event labels tell the story Initialize → Converge
    const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
    expect(labels.some((l) => /^init/i.test(l))).toBe(true);
    expect(labels.some((l) => /converged in 4 updates/.test(l))).toBe(true);
  });

  it('the mistake counter advances by exactly 1 per update (one online update per step)', () => {
    const run = computeRun(simulation, DEFAULT, 300);
    run.snapshots.forEach((s) => {
      expect(s.metrics.mistakes).toBe(s.metrics.updates); // mistakes ≡ update count
    });
    // each step fires exactly one update; the finished run re-emits the final
    // (converged) state without a new update, so the last snapshot repeats the count
    for (let i = 1; i < run.snapshots.length; i++) {
      const d = run.snapshots[i].metrics.updates - run.snapshots[i - 1].metrics.updates;
      expect(d).toBeGreaterThanOrEqual(0);
      expect(d).toBeLessThanOrEqual(1);
    }
    const last = run.snapshots[run.snapshots.length - 1];
    const prev = run.snapshots[run.snapshots.length - 2];
    expect(last.metrics.updates).toBe(prev.metrics.updates); // converged re-emission
  });
});

describe('perceptron: plan case 2 — oscillation on non-separable data (measured)', () => {
  it('never converges: honest telemetry with an oscillation reason', () => {
    const plan = buildPlan(NONSEP);
    expect(plan.converged).toBe(false);
    expect(plan.updates).toBe(OSCILLATION_CAP); // the snapshot cap = the run length
    // exact float cycle: NOT the mechanism here (none within MAX_UPDATES on
    // seed 42 — confirmed for every seed 0..60; see module header)
    expect(plan.cycleLength).toBe(0);
    const run = computeRun(simulation, NONSEP, 600);
    expect(run.telemetry.failedAtStep).toBeDefined();
    expect(run.telemetry.failedAtStep).toBe(181); // cap + init snapshot (measured)
    expect(run.telemetry.failureReason).toMatch(/oscillat/i);
    expect(run.telemetry.failureReason).toMatch(/not linearly separable/i);
    expect(run.snapshots).toHaveLength(181); // well under the ~200 snapshot cap
    const last = run.snapshots[run.snapshots.length - 1];
    expect(last.metrics.accuracy).toBeLessThan(1);
    expect(last.metrics.mistakesPerEpoch).toBeGreaterThan(0);
    expect(last.metrics.mistakes).toBe(OSCILLATION_CAP);
  });

  it('the Oscillate verdict lands exactly once — on the final snapshot, with one event', () => {
    const run = computeRun(simulation, NONSEP, 600);
    const oscStages = run.snapshots.filter((s) => s.timeline.includes('Oscillate'));
    expect(oscStages).toHaveLength(1);
    expect(oscStages[0]!.metrics.updates).toBe(OSCILLATION_CAP);
    const oscEvents = run.snapshots.flatMap((s) => s.events.map((e) => e.label)).filter((l) => /oscillat/i.test(l));
    expect(oscEvents).toHaveLength(1);
    // the pre-final snapshots still show the mistake-driven cycle honestly
    const mid = run.snapshots[Math.floor(run.snapshots.length / 2)];
    expect(mid.timeline.join('→')).toBe('Mistake→Update→Repeat');
  });

  it('mistakes keep firing in every epoch on non-separable data', () => {
    const plan = buildPlan(NONSEP);
    const mpe = plan.epochMarks.map((em) => em.mistakes);
    // completed epochs all fired ≥ 1 update — and many fired more (measured max 4)
    expect(mpe.every((m) => m > 0)).toBe(true);
    expect(Math.max(...mpe)).toBeGreaterThanOrEqual(4);
  });
});

describe('perceptron: plan case 3 — the single update rule, hand-verified', () => {
  it('w ← w + η·y·x and b ← b + η·y applied on a hand example (exact)', () => {
    // w = (1, −1), b = 0.5, x = (2, 3), y = +1, η = 0.5:
    expect(perceptronUpdate(1, -1, 0.5, 2, 3, 1, 0.5)).toEqual({ w1: 2, w2: 0.5, b: 1 });
    // same point, y = −1:
    expect(perceptronUpdate(1, -1, 0.5, 2, 3, -1, 0.5)).toEqual({ w1: 0, w2: -2.5, b: 0 });
    // η = 0 is the no-op edge (validateParams rejects η ≤ 0; the rule itself is inert):
    expect(perceptronUpdate(1, -1, 0.5, 2, 3, 1, 0)).toEqual({ w1: 1, w2: -1, b: 0.5 });
  });

  it('the module applies THE rule on the first misclassified point d0 (zero init)', () => {
    const plan = buildPlan(CASE3);
    expect(plan.converged).toBe(true);
    expect(plan.updates).toBe(1); // one clean sweep after a single fix
    const d0 = getData(CASE3).points[0];
    expect(plan.trace[0]!.hitIdx).toBe(0);
    // trace[0] must equal perceptronUpdate(0,0,0, d0.x, d0.y, −1, 1) componentwise
    const hand = perceptronUpdate(0, 0, 0, d0.x, d0.y, d0.label, 1);
    expect(plan.trace[0]!.w1).toBeCloseTo(hand.w1, 9);
    expect(plan.trace[0]!.w2).toBeCloseTo(hand.w2, 9);
    expect(plan.trace[0]!.b).toBeCloseTo(hand.b, 9);
    // measured exact numbers: y·x = (2.882465, 0.109210) with y = −1, so
    // w₁ = −x = 2.882465, w₂ = −y = 0.109210, b = −1
    expect(plan.trace[0]!.w1).toBeCloseTo(2.882465, 4);
    expect(plan.trace[0]!.w2).toBeCloseTo(0.109210, 4);
    expect(plan.trace[0]!.b).toBe(-1);
    // every snapshot's weight delta equals η·yᵢ·xᵢ of its highlighted point
    const run = computeRun(simulation, CASE3, 200);
    expect(run.snapshots).toHaveLength(3); // init + 1 update + converged
    const s1 = run.snapshots[1]!;
    const a0 = run.snapshots[0]!.algorithm;
    expect(s1.algorithm.w1 as number - (a0.w1 as number)).toBeCloseTo(hand.w1, 9);
    expect(s1.algorithm.w2 as number - (a0.w2 as number)).toBeCloseTo(hand.w2, 9);
    expect(s1.algorithm.b as number - (a0.b as number)).toBeCloseTo(hand.b, 9);
  });
});

describe('perceptron: plan case 4 — weight norm growth is bounded (measured)', () => {
  it('every update obeys ‖Δw‖ ≤ η·√(R²+1) — the data-radius bound incl. the bias term', () => {
    const plan = buildPlan(DEFAULT);
    const R = plan.data.R;
    const cap = Math.sqrt(R * R + 1); // η·√(R²+1) with η = 1
    let maxDelta = 0;
    for (let i = 0; i < plan.trace.length; i++) {
      const t = plan.trace[i]!;
      const w = i === 0 ? plan.init : plan.trace[i - 1]!;
      const delta = Math.hypot(t.w1 - w.w1, t.w2 - w.w2, t.b - w.b);
      expect(delta).toBeLessThanOrEqual(cap + 1e-12);
      maxDelta = Math.max(maxDelta, delta);
    }
    expect(maxDelta).toBeCloseTo(1.958212, 4); // measured max, comfortably under 2.457270
  });

  it('‖w‖ stays within O(R): final ‖w‖ = 2.741 < 2R ≈ 4.489, and grows one update at a time', () => {
    const run = computeRun(simulation, DEFAULT, 300);
    const last = run.snapshots[run.snapshots.length - 1];
    expect(last.metrics.normW).toBeGreaterThan(0);
    expect(last.metrics.normW).toBeCloseTo(2.740716, 4);
    expect(last.metrics.normW).toBeLessThan(2 * last.metrics.R);
    // ‖w‖ does not jump: consecutive snapshots differ by at most η·√(R²+1) too
    for (let i = 1; i < run.snapshots.length; i++) {
      const a = run.snapshots[i]!.algorithm;
      const p = run.snapshots[i - 1]!.algorithm;
      const d = Math.hypot(
        (a.w1 as number) - (p.w1 as number),
        (a.w2 as number) - (p.w2 as number),
        (a.b as number) - (p.b as number),
      );
      expect(d).toBeLessThanOrEqual(Math.sqrt(last.metrics.R * last.metrics.R + 1) + 1e-9);
    }
  });
});

describe('perceptron: classifier contract (decision-boundary view)', () => {
  it('register() exposes the classifier; both snapshot and fallback paths classify 40/40', () => {
    register(); // idempotent — registers topic + classifier
    const cls = getClassifier('perceptron');
    expect(cls).toBeDefined();
    const data = getData(DEFAULT);
    // snapshot path: w1/w2/b merged from the CURRENT snapshot's algorithm
    const run = computeRun(simulation, DEFAULT, 300);
    const last = run.snapshots[run.snapshots.length - 1];
    const snapshotParams = { ...DEFAULT, ...last.algorithm };
    for (const pt of data.points) {
      expect(cls!(pt.x, pt.y, snapshotParams)).toBe(pt.label === 1 ? 1 : 0);
    }
    // fallback path (no snapshot yet): the memoized plan's final weights
    for (const pt of data.points) {
      expect(classifyByParams(pt.x, pt.y, DEFAULT)).toBe(pt.label === 1 ? 1 : 0);
    }
  });
});

describe('perceptron: determinism + η invariance (measured)', () => {
  it('same seed → identical snapshot arrays; different seed → different data', () => {
    const r1 = computeRun(simulation, DEFAULT, 300);
    const r2 = computeRun(simulation, DEFAULT, 300);
    expect(JSON.stringify(r1.snapshots)).toBe(JSON.stringify(r2.snapshots));
    expect(r1.telemetry.snapshotCount).toBe(r2.telemetry.snapshotCount);
    const r7 = computeRun(simulation, { ...DEFAULT, seed: 7 }, 300);
    expect(JSON.stringify(r1.snapshots)).not.toBe(JSON.stringify(r7.snapshots));
    expect(buildPlan({ ...DEFAULT, seed: 7 }).updates).toBe(23); // measured
  });

  it('η is scale-invariant: identical update count/sequence; final weights scale exactly by η', () => {
    const run1 = computeRun(simulation, DEFAULT, 300);
    const run05 = computeRun(simulation, { ...DEFAULT, eta: 0.5 }, 300);
    expect(run1.snapshots).toHaveLength(6);
    expect(run05.snapshots).toHaveLength(6); // same update count (4)
    const a1 = run1.snapshots[run1.snapshots.length - 1].algorithm;
    const a05 = run05.snapshots[run05.snapshots.length - 1].algorithm;
    expect(a1.w1 as number).toBeCloseTo(2 * (a05.w1 as number), 9);
    expect(a1.w2 as number).toBeCloseTo(2 * (a05.w2 as number), 9);
    expect(a1.b as number).toBeCloseTo(2 * (a05.b as number), 9);
    // the mistakesPerEpoch series is identical
    const m1 = run1.snapshots.map((s) => s.metrics.mistakesPerEpoch);
    const m05 = run05.snapshots.map((s) => s.metrics.mistakesPerEpoch);
    expect(JSON.stringify(m1)).toBe(JSON.stringify(m05));
    // the geometric margin is scale-independent: same γ
    expect(run1.snapshots[run1.snapshots.length - 1].metrics.gamma)
      .toBeCloseTo(run05.snapshots[run05.snapshots.length - 1].metrics.gamma, 9);
  });
});

describe('perceptron: validateParams', () => {
  it('accepts the default parameter set', () => {
    expect(perceptronModule.validateParams?.(DEFAULT) ?? []).toHaveLength(0);
  });

  it('rejects η ≤ 0 (the update is undefined for η = 0 in classic form) and η > 1000 (overflow risk)', () => {
    const e0 = perceptronModule.validateParams?.({ ...DEFAULT, eta: 0 }) ?? [];
    expect(e0.some((s) => /η must be positive/.test(s))).toBe(true);
    const en = perceptronModule.validateParams?.({ ...DEFAULT, eta: -1 }) ?? [];
    expect(en.some((s) => /η must be positive/.test(s))).toBe(true);
    const huge = perceptronModule.validateParams?.({ ...DEFAULT, eta: 1e6 }) ?? [];
    expect(huge.some((s) => /risk/.test(s))).toBe(true);
  });

  it('rejects nPerClass < 2 and non-positive noise/margin', () => {
    const n = perceptronModule.validateParams?.({ ...DEFAULT, nPerClass: 1 }) ?? [];
    expect(n.some((s) => /≥ 2/.test(s))).toBe(true);
    expect((perceptronModule.validateParams?.({ ...DEFAULT, noise: 0 }) ?? []).some((s) => /positive/.test(s))).toBe(true);
    expect((perceptronModule.validateParams?.({ ...DEFAULT, margin: -1 }) ?? []).some((s) => /non-negative/.test(s))).toBe(true);
  });

  it('warns about non-separability risk when noise is large relative to separation', () => {
    const issues = perceptronModule.validateParams?.({ ...DEFAULT, margin: 0.9, noise: 0.6 }) ?? [];
    expect(issues.some((s) => /separable/i.test(s) && /risk/i.test(s))).toBe(true);
  });

  it('keeps the loss-curve metric honest: mistakes per epoch, not a loss', () => {
    expect(perceptronModule.lossMetricKey).toBe('mistakesPerEpoch');
    expect(perceptronModule.layers.foundation.some((v) => /NO loss/i.test(v.title))).toBe(true);
  });
});

describe('perceptron: honest telemetry on non-separable data', () => {
  it('the oscillation cap bounds every run — nothing ever hangs (snapshots < 200)', () => {
    // the worst case is the cap itself; every non-convergent config terminates there
    const plan = buildPlan(NONSEP);
    expect(plan.updates).toBeLessThanOrEqual(MAX_UPDATES);
    expect(plan.updates).toBeLessThan(200);
    // decided by the precomputed plan, not the engine budget
    const run = computeRun(simulation, NONSEP, 5000);
    expect(run.telemetry.failedAtStep).toBe(plan.updates + 1);
    expect(run.telemetry.failureReason).toMatch(/oscillat/i);
  });
});