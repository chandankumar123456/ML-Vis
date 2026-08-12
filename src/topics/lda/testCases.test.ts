// src/topics/lda/testCases.test.ts
// Plan Task 14 (Fisher's LDA) — the four prescribed test cases at UNIT level
// against the module's own math core, plus the run-level loop over
// testCases.ts, the classifier contract, determinism, the singular-S_W
// telemetry path and the validateParams guards.
//
// Hand-derived toy anchors (also printed in module.ts / testCases.ts). The
// scatter matrices, worked out BY HAND — class 0: (0,0)(2,0)(0,2)(2,2),
// class 1: (4,0)(4,2)(6,2)(6,4):
//
//   μ₀ = (1,1),  μ₁ = (5,2),  d = μ₁−μ₀ = (4,1)
//   C₀ = (1/4)·Σ(x−μ₀)(x−μ₀)ᵀ = [[1,0],[0,1]]        (deviations (±1,±1): s₁₁=4/4, s₁₂=0, s₂₂=4/4)
//   C₁ = (1/4)·Σ(x−μ₁)(x−μ₁)ᵀ = [[1,1],[1,2]]        (deviations (−1,−2)(−1,0)(1,0)(1,2):
//                                                       s₁₁=4/4, s₁₂=4/4, s₂₂=8/4)
//   S_W = C₀ + C₁ = [[2,1],[1,3]]                     det S_W = 2·3 − 1·1 = 5
//   S_W⁻¹ = (1/5)·[[3,−1],[−1,2]]                    (2×2 adjugate)
//   S_B = d·dᵀ = [[16,4],[4,1]]                      (rank 1)
//   w = S_W⁻¹d = (1/5)(3·4−1·1, −1·4+2·1) = (11/5, −2/5) = (2.2, −0.4)
//   ŵ = w/‖w‖ = (11,−2)/(5√5)                         ‖w‖ = √(11²+2²)/5 = √125/5 = √5 → ŵ = (11,−2)/(5√5)
//   J(w*) = wᵀS_Bw/wᵀS_Ww = dᵀS_W⁻¹d = (4·11−1·2)/5 = 42/5 = 8.4  (= λ: the eigen-link)
//   τ = ŵᵀ(μ₀+μ₁)/2 = (9 + 51)/(2·5√5) = 60/(10√5) = 6/√5 ≈ 2.6833
//   s₀² = ŵᵀC₀ŵ = ‖ŵ‖² = 1;   s₁² = ŵᵀC₁ŵ = (121 − 44 + 8)/125 = 85/125 = 17/25 = 0.68
//   θ* = angle of ŵ in [0,π) = atan2(−0.4, 2.2) + π (wrapped) ≈ 169.6952°
//   Threshold rule (measured z on ŵ): class 0 → {0, 22, −4, 18}/(5√5), all < τ;
//   class 1 → {44, 40, 62, 58}/(5√5), all > τ → trainError = 0, every point won.
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import {
  simulation, computeLdaStats, TOY_POINTS, getSweep, angleEvalAt, quadForm,
  classifyByParams, ldaModule, mulberry32,
} from './module';
import { ldaTestCases } from './testCases';
import type { LdaPoint } from './module';

const S5 = Math.sqrt(5);

// ===== Run-level loop over testCases.ts =====

describe('lda testCases (run level)', () => {
  for (const tc of ldaTestCases) {
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

// ===== Plan case 1 — unit level: w = S_W⁻¹(μ₁−μ₂) on the hand-derived toy set =====

describe('lda plan case 1: projection direction known for 2 classes (hand math, 1e-9)', () => {
  const s = computeLdaStats({ points: TOY_POINTS });

  it('scatter matrices match the hand derivation', () => {
    // C₀, C₁, S_W, S_B, det — the "write the scatter matrices" requirement
    expect(s.mu0).toEqual([1, 1]);
    expect(s.mu1).toEqual([5, 2]);
    expect(s.d).toEqual([4, 1]);
    for (const [r, c] of [[0, 0], [0, 1], [1, 0], [1, 1]] as const) {
      expect(s.C0[r][c]).toBeCloseTo((r === c ? 1 : 0) as number, 12); // C₀ = I
      expect(s.C1[r][c]).toBeCloseTo(r === 0 && c === 0 ? 1 : r === 1 && c === 1 ? 2 : 1, 12); // [[1,1],[1,2]]
      expect(s.SW[r][c]).toBeCloseTo((r === 0 && c === 0) ? 2 : (r === 1 && c === 1) ? 3 : 1, 12); // [[2,1],[1,3]]
      expect(s.SB[r][c]).toBeCloseTo((r === 0 ? 4 : 1) * (c === 0 ? 4 : 1), 12); // d·dᵀ = [[16,4],[4,1]]
    }
    expect(s.detSW).toBeCloseTo(5, 12);
  });

  it('closed form w = S_W⁻¹(μ₁−μ₂) matches (11/5, −2/5) to 1e-12', () => {
    expect(s.wRaw[0]).toBeCloseTo(11 / 5, 12);
    expect(s.wRaw[1]).toBeCloseTo(-2 / 5, 12);
    // unit direction ŵ = (11,−2)/(5√5), oriented so ŵᵀ(μ₁−μ₀) ≥ 0 (already true here)
    expect(s.wx).toBeCloseTo(11 / (5 * S5), 12);
    expect(s.wy).toBeCloseTo(-2 / (5 * S5), 12);
    expect(s.thetaAxisDeg).toBeCloseTo(169.695153531234, 6);
  });

  it('J(w*) = dᵀS_W⁻¹d = 42/5 = λ (the eigen-link identity)', () => {
    expect(s.jOpt).toBeCloseTo(42 / 5, 12);
    const lambda = s.d[0] * s.wRaw[0] + s.d[1] * s.wRaw[1]; // trace of rank-1 S_W⁻¹S_B
    expect(lambda).toBeCloseTo(42 / 5, 12);
    expect(s.jOpt).toBeCloseTo(lambda, 12);
    // direct Rayleigh ratio J = (ŵᵀd)²/(s₀²+s₁²) agrees with the matrix form
    const wd = s.wx * s.d[0] + s.wy * s.d[1];
    expect((wd * wd) / (s.withinVar0 + s.withinVar1)).toBeCloseTo(42 / 5, 12);
  });

  it('projected within-class variances s₀² = 1, s₁² = 17/25', () => {
    expect(s.withinVar0).toBeCloseTo(1, 12);
    expect(s.withinVar1).toBeCloseTo(17 / 25, 12);
  });

  it('threshold τ = 6/√5 and the affine rule b = −τ', () => {
    expect(s.threshold).toBeCloseTo(6 / S5, 12);
    expect(s.b).toBeCloseTo(-6 / S5, 12);
    expect(s.b).toBeCloseTo(-s.threshold, 12);
  });
});

// ===== Plan case 2 — unit level: J(θ*) beats every measured sweep direction =====

describe('lda plan case 2: LDA projection maximizes class separation', () => {
  const runs: { label: string; params: Record<string, number | boolean> }[] = [
    { label: 'default (15,2,30,2,42)', params: { nPerClass: 15, separation: 2, covAngleDeg: 30, covShape: 2, seed: 42 } },
    { label: 'close clusters (15,1,30,2,42)', params: { nPerClass: 15, separation: 1, covAngleDeg: 30, covShape: 2, seed: 42 } },
    { label: 'circular, seed 7 (15,3,0,1,7)', params: { nPerClass: 15, separation: 3, covAngleDeg: 0, covShape: 1, seed: 7 } },
    { label: 'elongated 60°, seed 123 (25,2,60,3,123)', params: { nPerClass: 25, separation: 2, covAngleDeg: 60, covShape: 3, seed: 123 } },
    { label: 'wide gap, seed 1 (10,4,90,2,1)', params: { nPerClass: 10, separation: 4, covAngleDeg: 90, covShape: 2, seed: 1 } },
    { label: 'toy hand set', params: { toy: true, seed: 42 } },
  ];

  for (const { label, params } of runs) {
    it(`${label}: J(θ*) strictly greater than J along every grid direction`, () => {
      const run = computeRun(simulation, params, 60);
      const snapshots = run.snapshots;
      expect(snapshots.length).toBe(37); // 36 grid directions + closed form
      const last = snapshots[snapshots.length - 1];
      expect(last.metrics.isOptimal).toBe(1);
      // The sweep occupies snapshots 0..35 (steps 1..36); the closed form is 36.
      for (let k = 0; k < 36; k++) {
        const jK = snapshots[k].metrics.jFisher as number;
        expect(jK, `${label}: grid snapshot ${k} J=${jK} not below J(θ*)=${last.metrics.jFisher}`)
          .toBeLessThan(last.metrics.jFisher as number);
      }
      // measured grid-max anchors on the default config (loss-curve peak)
      if (label === 'default (15,2,30,2,42)') {
        expect(last.metrics.gridMaxJ).toBeCloseTo(3.861638793, 6);
        expect(last.metrics.jFisher).toBeCloseTo(3.869000715, 6);
      }
    });
  }

  it('the eigen identity J(θ*) = dᵀS_W⁻¹d is exact on the Gaussian config too', () => {
    const sweep = getSweep({ nPerClass: 15, separation: 2, covAngleDeg: 30, covShape: 2, seed: 42 });
    const { stats } = sweep;
    const lambda = stats.d[0] * stats.wRaw[0] + stats.d[1] * stats.wRaw[1];
    expect(stats.jOpt).toBeCloseTo(lambda, 10);
    // and the grid's best angle (140° here) is NOT the analytic optimum
    expect(stats.thetaAxisDeg).toBeCloseTo(138.455844, 4);
  });
});

// ===== Plan case 3 — unit level: 2-class LDA = threshold decision rule =====

describe('lda plan case 3: threshold decision rule', () => {
  const s = computeLdaStats({ points: TOY_POINTS });

  it('every toy point: z > τ ⟺ class 1 (measured z-values in the file header)', () => {
    const zOf = (d: LdaPoint) => s.wx * d.x + s.wy * d.y;
    // exact projected coordinates, hand-checked: class 0 → {0, 22, −4, 18}/(5√5),
    // class 1 → {44, 40, 62, 58}/(5√5); all class-0 z < τ < all class-1 z.
    const class0 = TOY_POINTS.filter((d) => d.cls === 0);
    const class1 = TOY_POINTS.filter((d) => d.cls === 1);
    for (const d of class0) {
      expect(zOf(d)).toBeLessThan(s.threshold);
      expect(zOf(d) > s.threshold ? 1 : 0).toBe(0);
    }
    for (const d of class1) {
      expect(zOf(d)).toBeGreaterThan(s.threshold);
      expect(zOf(d) > s.threshold ? 1 : 0).toBe(1);
    }
    // spot fraction checks against 5√5 denominators
    const f = (num: number) => num / (5 * S5);
    expect(zOf({ x: 2, y: 0, cls: 0 })).toBeCloseTo(f(22), 12);
    expect(zOf({ x: 0, y: 2, cls: 0 })).toBeCloseTo(f(-4), 12);
    expect(zOf({ x: 4, y: 0, cls: 1 })).toBeCloseTo(f(44), 12);
    expect(zOf({ x: 6, y: 4, cls: 1 })).toBeCloseTo(f(58), 12);
    expect(s.trainError).toBe(0);
  });

  it('classifyByParams = wx·x + wy·y + b > 0 (threshold in affine form), snapshot weights win', () => {
    // DecisionBoundary merges the CURRENT snapshot's algorithm into params —
    // the boundary must reflect the exact step being scrubbed.
    const atFinal = { toy: true, seed: 42, wx: s.wx, wy: s.wy, b: s.b };
    for (const d of TOY_POINTS) {
      expect(classifyByParams(d.x, d.y, atFinal), `point (${d.x}, ${d.y}) cls ${d.cls}`).toBe(d.cls);
    }
    // an arbitrary sweep axis (θ = 30°) with its own threshold — a lever check
    const sweep = getSweep({ nPerClass: 15, separation: 2, covAngleDeg: 30, covShape: 2, seed: 42 });
    const ev = sweep.evals[6]; // θ = 30° (grid k=6)
    const atSweep = { ...atFinal, wx: ev.wx, wy: ev.wy, b: ev.b };
    const pt = sweep.data.points[0];
    const z = ev.wx * pt.x + ev.wy * pt.y;
    expect(classifyByParams(pt.x, pt.y, atSweep)).toBe(z > ev.threshold ? 1 : 0);
  });

  it('classifyByParams falls back to the closed-form solution when no snapshot exists', () => {
    expect(classifyByParams(4, 0, { toy: true, seed: 42 })).toBe(1);  // z = 44/(5√5) > τ
    expect(classifyByParams(0, 0, { toy: true, seed: 42 })).toBe(0);  // z = 0 < τ
    const p = { nPerClass: 15, separation: 2, covAngleDeg: 30, covShape: 2, seed: 42 };
    const sw = getSweep(p);
    const q = sw.data.points[0];
    expect(classifyByParams(q.x, q.y, p)).toBe(sw.stats.wx * q.x + sw.stats.wy * q.y + sw.stats.b > 0 ? 1 : 0);
  });

  it('run timeline: Data → Scatter → Project → Evaluate → LDA Solution, events tell the arc', () => {
    const run = computeRun(simulation, { toy: true, seed: 42 }, 60);
    const first = run.snapshots[0].timeline;
    const last = run.snapshots[run.snapshots.length - 1];
    expect(first).toEqual(['Data', 'Scatter', 'Project', 'Evaluate']);
    expect(last.timeline[last.timeline.length - 1]).toBe('LDA Solution');
    expect(last.algorithm.mode).toBe('lda-optimal');
    expect(last.algorithm.isOptimal).toBe(1);
    const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
    expect(labels).toContain('lda-data-generated');
    expect(labels).toContain('closed-form-lda-solution');
    const gridEvents = labels.filter((l) => /^direction-g\d+-of-36$/.test(l));
    expect(gridEvents.length).toBe(36);
  });
});

// ===== Plan case 4 — unit level: reduces to 1-D, within-class variance =====

describe('lda plan case 4: reduces to 1-D (measured within-class variance)', () => {
  it('the final axis is the ANALYTIC optimum, not a grid angle (angleDeg === thetaOptDeg)', () => {
    const run = computeRun(simulation, { toy: true, seed: 42 }, 60);
    const last = run.snapshots[run.snapshots.length - 1];
    expect(last.metrics.angleDeg).toBeCloseTo(last.metrics.thetaOptDeg as number, 9);
    expect(last.metrics.thetaOptDeg).toBeCloseTo(169.695154, 6); // between grid angles 165°/170°
    expect(last.metrics.isOptimal).toBe(1);
  });

  it('projection of variance = variance of projection (the 1-D collapse is exact)', () => {
    const s = computeLdaStats({ points: TOY_POINTS });
    for (const cls of [0, 1] as const) {
      const pts = TOY_POINTS.filter((d) => d.cls === cls);
      const zs = pts.map((d) => s.wx * d.x + s.wy * d.y);
      const zMean = zs.reduce((a, b) => a + b, 0) / zs.length;
      const sampleVar = zs.reduce((a, z) => a + (z - zMean) ** 2, 0) / zs.length;
      const form = quadForm(cls === 0 ? s.C0 : s.C1, [s.wx, s.wy]);
      expect(sampleVar, `class ${cls} projected variance`).toBeCloseTo(form, 10);
    }
    // measured: s₀² = 1, s₁² = 17/25
    expect(s.withinVar0).toBeCloseTo(1, 12);
    expect(s.withinVar1).toBeCloseTo(17 / 25, 12);
  });

  it('whitening COMPRESSES the classes: within-class variance along θ* < along the naive mean-difference axis', () => {
    // The honest "minimal" reading (plan wording): Fisher optimizes the
    // between/within RATIO. The LDA axis reduces the projected within-class
    // variance BELOW the raw μ₁−μ₂ direction (the S_W⁻¹ whitening step). It
    // is NOT the absolute within-variance minimum — a bare minor-axis
    // direction is tighter but destroys separation (J there is much smaller).
    for (const [p, label] of [
      [{ toy: true, seed: 42 }, 'toy'],
      [{ nPerClass: 15, separation: 2, covAngleDeg: 30, covShape: 2, seed: 42 }, 'gaussian'],
    ] as const) {
      const sw = getSweep(p as Record<string, number | boolean>);
      const { stats } = sw;
      const d = stats.d;
      const nd = Math.hypot(d[0], d[1]);
      const wRawAxis: [number, number] = [d[0] / nd, d[1] / nd];
      const withinRaw = quadForm(stats.C0, wRawAxis) + quadForm(stats.C1, wRawAxis);
      const withinOpt = stats.withinVar0 + stats.withinVar1;
      // measured: toy 1.68 vs 2.5294; gaussian 0.6287 vs 2.1685
      expect(withinOpt, `${label} LDA within`).toBeLessThan(withinRaw);
      // ...and the Fisher ratio prefers it: J(θ*) > J(raw axis), measured
      const evRaw = angleEvalAt(stats, sw.data, Math.atan2(d[1], d[0]));
      expect(evRaw.jFisher, `${label} J(raw axis)`).toBeLessThan(stats.jOpt);
      if (label === 'toy') {
        expect(withinOpt).toBeCloseTo(42 / 25, 12); // s₀²+s₁² = 1 + 17/25
        expect(withinRaw).toBeCloseTo(2.529411765, 6); // 43/17 measured
      } else {
        expect(withinOpt).toBeCloseTo(0.628655308, 6);   // measured
        expect(withinRaw).toBeCloseTo(2.168514425, 6);   // measured
        // the sweep's tightest direction is even smaller — the ratio, not the
        // bare minimum, is what the closed form maximizes (documented honest claim)
        const gridMin = Math.min(...sw.evals.map((e) => e.withinVar0 + e.withinVar1));
        expect(gridMin).toBeLessThan(withinOpt);
      }
    }
  });
});

// ===== Determinism + classifier caching + singular-S_W telemetry =====

describe('lda determinism, failure path and guards', () => {
  it('same params → byte-identical snapshot arrays', () => {
    const p = { nPerClass: 15, separation: 2, covAngleDeg: 30, covShape: 2, seed: 42 };
    const a = computeRun(simulation, p, 60);
    const b = computeRun(simulation, p, 60);
    expect(JSON.stringify(a.snapshots)).toBe(JSON.stringify(b.snapshots));
    expect(a.telemetry.snapshotCount).toBe(37);
    expect(a.telemetry.failedAtStep).toBeUndefined();
  });

  it('collinear classes → singular S_W is an HONEST run failure (no NaN)', () => {
    const run = computeRun(
      simulation,
      { points: '[[0,0,0],[1,1,0],[2,2,0],[3,3,1],[4,4,1],[5,5,1]]', seed: 42 },
      60,
    );
    expect(run.telemetry.failedAtStep).toBeDefined();
    expect(run.telemetry.failureReason ?? '').toMatch(/singular/i);
    // every emitted metric/algorithm value stays finite (sandbox never sees NaN)
    for (const snap of run.snapshots) {
      for (const v of Object.values(snap.metrics)) expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('validateParams flags every degenerate-data guard', () => {
    const vp = ldaModule.validateParams!;
    expect(vp({ nPerClass: 2, separation: 2, covAngleDeg: 30, covShape: 2, seed: 42 }).length).toBeGreaterThan(0);
    expect(vp({ nPerClass: 15, separation: 0, covAngleDeg: 30, covShape: 2, seed: 42 }).some((s) => /separation/.test(s))).toBe(true);
    expect(vp({ nPerClass: 15, separation: 2, covAngleDeg: 180, covShape: 2, seed: 42 }).some((s) => /covAngleDeg/.test(s))).toBe(true);
    expect(vp({ nPerClass: 15, separation: 2, covAngleDeg: 30, covShape: 0.5, seed: 42 }).some((s) => /covShape/.test(s))).toBe(true);
    expect(vp({ nPerClass: 15, separation: 2, covAngleDeg: 30, covShape: 2, seed: 10000 }).some((s) => /seed/.test(s))).toBe(true);
    expect(vp({ points: 'not json', seed: 42 }).length).toBeGreaterThan(0);
    // ≥ 3 per class required (rank-deficient within-class scatter otherwise)
    expect(vp({ points: '[[0,0,0],[1,0,0],[5,0,1],[6,0,1]]', seed: 42 }).some((s) => /points/.test(s))).toBe(true);
    expect(vp({ nPerClass: 15, separation: 2, covAngleDeg: 30, covShape: 2, seed: 42 }).length).toBe(0);
  });

  it('numeric claims converge: every metric finite across the whole default run', () => {
    const run = computeRun(simulation, { nPerClass: 15, separation: 2, covAngleDeg: 30, covShape: 2, seed: 42 }, 60);
    expect(run.telemetry.failedAtStep).toBeUndefined();
    for (const snap of run.snapshots) {
      for (const v of Object.values(snap.metrics)) expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('mulberry32 is deterministic (seeded stream)', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 10; i++) expect(a()).toBe(b());
  });
});