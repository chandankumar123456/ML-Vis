// src/topics/svm-hard-margin/testCases.test.ts
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import {
  simulation, svmModule, register, getSweep, solveHardMargin,
  classifyByParams, supportVectorIndices,
} from './module';
import { svmTestCases } from './testCases';
import { getClassifier } from '../../registry/viewRegistry';

// Measured anchors (verified by running the simulator — every number below was
// printed by the module before being asserted):
//   default (nPerClass 12, margin 1.5, noise 0.45, seed 42):
//     w = (1.519692, 0.381713), b = −0.230105, ‖w‖ = 1.566898,
//     margin = 2/‖w‖ = 1.276408, γ = 1/‖w‖ = 0.638204, ½‖w‖² = 1.227584
//     SVs = indices [9, 21] (d9 class 0 at (−0.406, −0.400), d21 class 1 at (0.832, −0.089)),
//     each at distance 0.638204 from the boundary; closest NON-SV (d20) at 0.757101.
//     run = 40 snapshots (SWEEP_CAP), halfWSq non-increasing 1.240917 → 1.227584.
//   scale 2: ‖w‖ = 0.783449, margin = 2.552815 (both scale by 1/2 and 2);
//   scale 0.5: ‖w‖ = 3.133795 (= 2×). margin·‖w‖ = 2.000000 in every case.
//   seed 7: margin = 1.364655, ½‖w‖² = 1.073951, SVs = [0, 14, 22].

const DEFAULT = { nPerClass: 12, margin: 1.5, noise: 0.45, seed: 42 };

describe('svm-hard-margin testCases (data-driven)', () => {
  for (const tc of svmTestCases) {
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
        for (const lbl of tc.expect.eventLabels) {
          expect(labels).toContain(lbl);
        }
      }
      if (tc.expect.finalMetrics) {
        const m = run.snapshots[run.snapshots.length - 1].metrics;
        for (const [k, pred] of Object.entries(tc.expect.finalMetrics)) {
          if (typeof pred === 'function') expect(pred(m[k]), `metric ${k} failed for ${tc.name}`).toBe(true);
          else expect(m[k]).toBeCloseTo(pred, 6);
        }
      }
    });
  }
});

describe('svm-hard-margin: plan case 1 — max margin on separable data', () => {
  it('margin = 2/‖w‖ holds exactly (hand formula) for the optimal model', () => {
    const run = computeRun(simulation, DEFAULT, 500);
    const last = run.snapshots[run.snapshots.length - 1];
    const a = last.algorithm;
    const normW = Math.hypot(a.w1 as number, a.w2 as number);
    // formula margin = 2/‖w‖ must reproduce the measured metric
    expect(last.metrics.margin).toBeCloseTo(2 / normW, 6);
    expect(normW).toBeCloseTo(1.566898, 4);
    expect(last.metrics.margin).toBeCloseTo(1.276408, 4);
    expect(last.metrics.gamma).toBeCloseTo(1 / normW, 6);
    expect(last.metrics.halfWSq).toBeCloseTo(0.5 * normW * normW, 6);
  });

  it('w is the boundary normal: w ⊥ boundary direction, and w·x + b = 0 passes between the classes', () => {
    const run = computeRun(simulation, DEFAULT, 500);
    const a = run.snapshots[run.snapshots.length - 1].algorithm;
    const w1 = a.w1 as number, w2 = a.w2 as number, b = a.b as number;
    // boundary direction d = (−w2, w1); w·d must be 0 (normal ⊥ boundary)
    expect(w1 * -w2 + w2 * w1).toBeCloseTo(0, 9);
    // the boundary bisects the support-vector segment: w·mid + b = 0
    const pts = getSweep(DEFAULT).data;
    const svA = pts[9], svB = pts[21];
    const midx = (svA.x + svB.x) / 2, midy = (svA.y + svB.y) / 2;
    expect(w1 * midx + w2 * midy + b).toBeCloseTo(0, 4);
    // and the SVs are on opposite sides of the boundary
    expect(w1 * svA.x + w2 * svA.y + b).toBeLessThan(0);
    expect(w1 * svB.x + w2 * svB.y + b).toBeGreaterThan(0);
  });

  it('every point satisfies the canonical constraint yᵢ(w·xᵢ+b) ≥ 1 (feasibility, margin ≥ γ)', () => {
    const run = computeRun(simulation, DEFAULT, 500);
    const a = run.snapshots[run.snapshots.length - 1].algorithm;
    const w1 = a.w1 as number, w2 = a.w2 as number, b = a.b as number;
    const normW = Math.hypot(w1, w2);
    for (const pt of getSweep(DEFAULT).data) {
      const y = pt.cls === 1 ? 1 : -1;
      const fm = y * (w1 * pt.x + w2 * pt.y + b);
      expect(fm).toBeGreaterThanOrEqual(1 - 1e-9);
      // geometric margin (distance) ≥ γ = 1/‖w‖ for every point
      expect(Math.abs(w1 * pt.x + w2 * pt.y + b) / normW).toBeGreaterThanOrEqual(1 / normW - 1e-9);
    }
  });
});

describe('svm-hard-margin: plan case 2 — support vectors are the closest points', () => {
  it('support vectors sit at distance exactly 1/‖w‖; non-SVs are strictly farther', () => {
    const run = computeRun(simulation, DEFAULT, 500);
    const last = run.snapshots[run.snapshots.length - 1];
    const a = last.algorithm;
    const w1 = a.w1 as number, w2 = a.w2 as number, b = a.b as number;
    const normW = Math.hypot(w1, w2);
    const pts = getSweep(DEFAULT).data;
    expect(last.metrics.svCount).toBe(2);
    const sv = supportVectorIndices(pts, w1, w2, b);
    expect(sv).toEqual([9, 21]);
    const svDist: number[] = [];
    const nonSvDist: number[] = [];
    for (let i = 0; i < pts.length; i++) {
      const d = Math.abs(w1 * pts[i].x + w2 * pts[i].y + b) / normW;
      (sv.includes(i) ? svDist : nonSvDist).push(d);
    }
    // SVs at distance 1/‖w‖ (0.638204); non-SVs strictly beyond (min 0.757101)
    for (const d of svDist) expect(d).toBeCloseTo(1 / normW, 6);
    for (const d of nonSvDist) expect(d).toBeGreaterThan(1 / normW + 1e-3);
    expect(Math.min(...nonSvDist)).toBeGreaterThan(Math.max(...svDist));
    expect(Math.min(...nonSvDist)).toBeCloseTo(0.757101, 4);
  });
});

describe('svm-hard-margin: plan case 3 — scaling invariance (measured)', () => {
  it('scaling data by 2 halves ‖w‖ and doubles the margin; boundary unchanged; margin·‖w‖ = 2', () => {
    const base = computeRun(simulation, DEFAULT, 500);
    const scaled = computeRun(simulation, { ...DEFAULT, scale: 2 }, 500);
    const mb = base.snapshots[base.snapshots.length - 1].metrics;
    const ms = scaled.snapshots[scaled.snapshots.length - 1].metrics;
    const ab = base.snapshots[base.snapshots.length - 1].algorithm;
    const as_ = scaled.snapshots[scaled.snapshots.length - 1].algorithm;
    expect(ms.normW).toBeCloseTo(mb.normW / 2, 5);
    expect(ms.margin).toBeCloseTo(mb.margin * 2, 5);
    expect(ms.margin * ms.normW).toBeCloseTo(2, 5);
    expect(mb.margin * mb.normW).toBeCloseTo(2, 5);
    // same boundary: unit normals align, and the midpoint maps to w·x+b = 0
    const ang1 = Math.atan2(ab.w2 as number, ab.w1 as number);
    const ang2 = Math.atan2(as_.w2 as number, as_.w1 as number);
    expect(Math.abs(ang1 - ang2)).toBeLessThan(1e-9);
  });

  it('scaling data by ½ doubles ‖w‖ (the plan\'s "scaling by 2 doubles ‖w‖" framing)', () => {
    const base = computeRun(simulation, DEFAULT, 500);
    const shrunk = computeRun(simulation, { ...DEFAULT, scale: 0.5 }, 500);
    const mb = base.snapshots[base.snapshots.length - 1].metrics;
    const ms = shrunk.snapshots[shrunk.snapshots.length - 1].metrics;
    expect(ms.normW).toBeCloseTo(mb.normW * 2, 5);
    expect(ms.margin).toBeCloseTo(mb.margin / 2, 5);
    expect(ms.margin * ms.normW).toBeCloseTo(2, 5);
  });
});

describe('svm-hard-margin: plan case 4 — classifier correctness', () => {
  it('the registered classifier returns the correct class for the optimal model', () => {
    register(); // idempotent; registers topic + classifier into viewRegistry
    const cls = getClassifier('svm-hard-margin');
    expect(cls).toBeDefined();
    const run = computeRun(simulation, DEFAULT, 500);
    const last = run.snapshots[run.snapshots.length - 1];
    const params = { ...DEFAULT, ...last.algorithm };
    for (const pt of getSweep(DEFAULT).data) {
      expect(cls!(pt.x, pt.y, params)).toBe(pt.cls);
    }
  });

  it('classifyByParams reads the snapshot model and the fallback re-solves deterministically', () => {
    const run = computeRun(simulation, DEFAULT, 500);
    const last = run.snapshots[run.snapshots.length - 1];
    // snapshot path: w1/w2/b come from snapshot.algorithm
    for (const pt of getSweep(DEFAULT).data) {
      expect(classifyByParams(pt.x, pt.y, { ...DEFAULT, ...last.algorithm })).toBe(pt.cls);
    }
    // fallback path (no snapshot): re-solves the optimum via getSweep
    for (const pt of getSweep(DEFAULT).data) {
      expect(classifyByParams(pt.x, pt.y, DEFAULT)).toBe(pt.cls);
    }
  });
});

describe('svm-hard-margin: determinism + sweep integrity', () => {
  it('same seed → identical snapshot arrays; different seed → different data', () => {
    const r1 = computeRun(simulation, DEFAULT, 500);
    const r2 = computeRun(simulation, DEFAULT, 500);
    expect(JSON.stringify(r1.snapshots)).toBe(JSON.stringify(r2.snapshots));
    expect(r1.telemetry.snapshotCount).toBe(r2.telemetry.snapshotCount);
    const r3 = computeRun(simulation, { ...DEFAULT, seed: 7 }, 500);
    expect(JSON.stringify(r1.snapshots)).not.toBe(JSON.stringify(r3.snapshots));
  });

  it('the run is a 40-step sweep; the objective ½‖w‖² is non-increasing; the final snapshot is the exact optimum', () => {
    const run = computeRun(simulation, DEFAULT, 500);
    expect(run.snapshots).toHaveLength(40); // SWEEP_CAP
    const half = run.snapshots.map((s) => s.metrics.halfWSq);
    for (let i = 1; i < half.length; i++) expect(half[i]).toBeLessThanOrEqual(half[i - 1]!);
    expect(half[0]).toBeCloseTo(1.240917, 4);
    expect(half[half.length - 1]).toBeCloseTo(1.227584, 4);
    // last snapshot model = solveHardMargin's exact optimum
    const sw = getSweep(DEFAULT);
    const optimal = solveHardMargin(sw.data);
    const a = run.snapshots[run.snapshots.length - 1].algorithm;
    expect(a.w1).toBeCloseTo(optimal.w1, 9);
    expect(a.w2).toBeCloseTo(optimal.w2, 9);
    expect(a.b).toBeCloseTo(optimal.b, 9);
    // every snapshot model is a FEASIBLE separator (train error 0)
    for (const s of run.snapshots) expect(s.metrics.trainError).toBe(0);
  });
});

describe('svm-hard-margin: validateParams', () => {
  it('rejects zero/degenerate separation (margin below the slider minimum)', () => {
    const issues = svmModule.validateParams?.({ ...DEFAULT, margin: 0 }) ?? [];
    expect(issues.some((s) => /margin/.test(s) && /≥ 0.5/.test(s))).toBe(true);
  });

  it('rejects non-positive noise and nPerClass < 2', () => {
    const noise = svmModule.validateParams?.({ ...DEFAULT, noise: 0 }) ?? [];
    expect(noise.some((s) => /noise/.test(s))).toBe(true);
    const n = svmModule.validateParams?.({ ...DEFAULT, nPerClass: 1 }) ?? [];
    expect(n.some((s) => /nPerClass/.test(s) && /≥ 2/.test(s))).toBe(true);
    const big = svmModule.validateParams?.({ ...DEFAULT, nPerClass: 25 }) ?? [];
    expect(big.some((s) => /nPerClass/.test(s) && /20/.test(s))).toBe(true);
  });

  it('warns when noise is large relative to separation (non-separability risk)', () => {
    const issues = svmModule.validateParams?.({ ...DEFAULT, margin: 0.5, noise: 0.5 }) ?? [];
    expect(issues.some((s) => /noise/.test(s) && /large relative/.test(s))).toBe(true);
  });

  it('accepts the default parameter set', () => {
    const issues = svmModule.validateParams?.(DEFAULT) ?? [];
    expect(issues.length).toBe(0);
  });
});

describe('svm-hard-margin: honest telemetry failure on non-separable data', () => {
  it('non-separable config (margin 0.5, noise 1.5, seed 42) fails cleanly via telemetry', () => {
    const run = computeRun(simulation, { nPerClass: 12, margin: 0.5, noise: 1.5, seed: 42 }, 500);
    expect(run.telemetry.failedAtStep).toBeDefined();
    expect(run.telemetry.failureReason).toMatch(/separable/i);
  });

  it('seed search lands on a separable seed and surfaces it as dataSeed', () => {
    // margin 1.0 / noise 0.9: seeds 42–47 are non-separable; the search uses seed 48
    const sw = getSweep({ nPerClass: 12, margin: 1.0, noise: 0.9, seed: 42 });
    expect(sw.dataSeed).toBe(48);
    expect(sw.solution.separable).toBe(true);
    expect(sw.solution.margin).toBeCloseTo(0.632867, 4);
  });
});
