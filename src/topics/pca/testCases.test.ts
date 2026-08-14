// src/topics/pca/testCases.test.ts
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import {
  simulation, pcaModule, register, getSweep,
  centeredCov, rawCov, eigen2x2, varianceAlong, reconError, quadForm,
} from './module';
import { pcaTestCases } from './testCases';

// Measured anchors (ALL verified by running the module — every number below was
// printed by the module before being asserted):
//   default (n 40, corr 0.7, rotDeg 30, noise 0.15, seed 42):
//     Σ = [[0.37011827663219543, 0.37911198787583555],
//          [0.37911198787583555, 1.8463033092217696]]
//     λ₁ = 1.9379736879527472, λ₂ = 0.2784478979012177,
//     total = 2.216421585853965, ratio₁ = 0.8743705170178919,
//     PC1 = (0.23502957205216105, 0.9719882202274769) at 76.40663404901252°,
//     v₁·v₂ = 0 exactly, reconErrK1 = 0.2784478979012177 (= λ₂), reconErrK2 = 0,
//     run = 37 snapshots (36 sweep + 1 closed-form final).
//     varianceAlong(v₁) = 1.9379736879527472 = λ₁ exactly; varianceAlong(v₂) =
//     0.27844789790121793 = λ₂; quadForm(Σ, v₁) = λ₁ to 1e-15.
//     grid sweep peaks at 75° with variance 1.9369736563193656 — λ₁ exceeds it
//     by 0.00100; dense 0.25° scan over 720 directions tops out at
//     1.9379692812332503 — λ₁ exceeds it by 4.4067e-6.
//   rotDeg 80 (centering contrast): centered PC1 at 127.15210735118922° vs
//     RAW (uncentered) PC1 at 33.29706869064782° (points at the mean μ of the
//     rotDeg-80 draw; mean direction ≈ 34.06°, measured); raw PC1 explains only
//     0.12582945811006516 of the CENTERED variance (true PC1: 0.8775840456844758).
//   seed 7: λ₁ = 2.23747379244006, λ₂ = 0.2641044311347427, PC1 at 68.85°.
//   cfg2 (n 60, corr 0.3, rotDeg 60, noise 0.25, seed 123): λ₁ = 1.4315445585568694,
//     λ₂ = 0.678435059535849, PC1 at 112.13°.
//   near-degenerate (noise 0, corr 0.98): λ₂ = 0.01600865622759695, ratio₂ = 0.005884164837050432.

const DEFAULT = { n: 40, corr: 0.7, rotDeg: 30, noise: 0.15, seed: 42 };

describe('pca testCases (data-driven)', () => {
  for (const tc of pcaTestCases) {
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

describe('pca: plan case 1 — the first PC maximizes variance', () => {
  it('λ₁ ≥ uᵀΣu along EVERY 5° sweep direction (all 36 grid angles)', () => {
    const sw = getSweep(DEFAULT);
    for (const ev of sw.evals) {
      expect(sw.eig.lambda1).toBeGreaterThanOrEqual(ev.variance - 1e-9);
    }
    // measured: grid peak = 1.9369736563193656 at 75°, λ₁ = 1.9379736879527472
    const gridMax = Math.max(...sw.evals.map((e) => e.variance));
    expect(gridMax).toBeCloseTo(1.9369736563193656, 6);
    expect(sw.eig.lambda1 - gridMax).toBeCloseTo(0.0010000316333815817, 9);
  });

  it('λ₁ ≥ uᵀΣu along a DENSE 0.25° scan (720 directions) — beats any other direction', () => {
    const sw = getSweep(DEFAULT);
    let denseMax = -Infinity;
    for (let k = 0; k < 720; k++) {
      const th = (k * Math.PI) / 720;
      denseMax = Math.max(denseMax, quadForm(sw.Sigma, [Math.cos(th), Math.sin(th)]));
    }
    expect(denseMax).toBeCloseTo(1.9379692812332503, 6);
    expect(sw.eig.lambda1).toBeGreaterThanOrEqual(denseMax - 1e-9);
  });

  it('the loss curve (axisVariance) peaks exactly at the closed-form final snapshot', () => {
    const run = computeRun(simulation, DEFAULT, 500);
    const curve = run.snapshots.map((s) => s.metrics.axisVariance);
    const maxAt = curve.indexOf(Math.max(...curve));
    expect(maxAt).toBe(run.snapshots.length - 1); // final snapshot is the peak
    expect(run.snapshots[maxAt].metrics.axisVariance).toBeCloseTo(1.9379736879527472, 9);
    // and the final snapshot is marked optimal
    expect(run.snapshots[maxAt].metrics.isOptimal).toBe(1);
    expect(run.snapshots[maxAt].metrics.angleDeg).toBeCloseTo(76.40663404901252, 6);
  });
});

describe('pca: plan case 2 — PCs are orthogonal', () => {
  it('v₁·v₂ = 0 exactly (90°-rotation construction), within 1e-6 as the plan requires', () => {
    const sw = getSweep(DEFAULT);
    const dot = sw.eig.v1[0] * sw.eig.v2[0] + sw.eig.v1[1] * sw.eig.v2[1];
    expect(Math.abs(dot)).toBeLessThan(1e-12); // measured exactly 0
    expect(dot).toBeCloseTo(0, 12);
  });

  it('both PCs are unit vectors, and Σ·v = λ·v holds (eigen-decomposition check to 1e-9)', () => {
    const sw = getSweep(DEFAULT);
    const { Sigma, eig } = sw;
    // unit norms
    expect(Math.hypot(eig.v1[0], eig.v1[1])).toBeCloseTo(1, 12);
    expect(Math.hypot(eig.v2[0], eig.v2[1])).toBeCloseTo(1, 12);
    // Σv₁ = λ₁v₁  and  Σv₂ = λ₂v₂, entrywise
    const Mv1 = [
      Sigma[0][0] * eig.v1[0] + Sigma[0][1] * eig.v1[1],
      Sigma[1][0] * eig.v1[0] + Sigma[1][1] * eig.v1[1],
    ];
    const Mv2 = [
      Sigma[0][0] * eig.v2[0] + Sigma[0][1] * eig.v2[1],
      Sigma[1][0] * eig.v2[0] + Sigma[1][1] * eig.v2[1],
    ];
    expect(Mv1[0]).toBeCloseTo(eig.lambda1 * eig.v1[0], 9);
    expect(Mv1[1]).toBeCloseTo(eig.lambda1 * eig.v1[1], 9);
    expect(Mv2[0]).toBeCloseTo(eig.lambda2 * eig.v2[0], 9);
    expect(Mv2[1]).toBeCloseTo(eig.lambda2 * eig.v2[1], 9);
  });
});

describe('pca: plan case 3 — eigenvalues equal the explained variance', () => {
  it('λ_k = empirical variance of the data projected on v_k (to 1e-9)', () => {
    const sw = getSweep(DEFAULT);
    const var1 = varianceAlong(sw.data, sw.mu, sw.eig.v1);
    const var2 = varianceAlong(sw.data, sw.mu, sw.eig.v2);
    expect(var1).toBeCloseTo(sw.eig.lambda1, 9);
    expect(var2).toBeCloseTo(sw.eig.lambda2, 9);
    // measured: varianceAlong(v₁) = 1.9379736879527472 = λ₁ exactly
    expect(var1).toBeCloseTo(1.9379736879527472, 12);
    expect(var2).toBeCloseTo(0.2784478979012177, 12);
  });

  it('λ_k = the Rayleigh quotient v_kᵀΣv_k (the sweep objective at the PC directions)', () => {
    const sw = getSweep(DEFAULT);
    expect(quadForm(sw.Sigma, sw.eig.v1)).toBeCloseTo(sw.eig.lambda1, 9);
    expect(quadForm(sw.Sigma, sw.eig.v2)).toBeCloseTo(sw.eig.lambda2, 9);
    // and the eigenvalues sum to the total variance (trace)
    expect(sw.eig.lambda1 + sw.eig.lambda2).toBeCloseTo(sw.eig.totalVariance, 12);
    expect(sw.eig.totalVariance).toBeCloseTo(2.216421585853965, 9);
  });

  it('explained-variance ratios are measured and consistent (ratio₁ + ratio₂ = 1)', () => {
    const sw = getSweep(DEFAULT);
    const r1 = sw.eig.lambda1 / sw.eig.totalVariance;
    const r2 = sw.eig.lambda2 / sw.eig.totalVariance;
    expect(r1).toBeCloseTo(0.8743705170178919, 9);
    expect(r2).toBeCloseTo(0.1256294829821081, 9);
    expect(r1 + r2).toBeCloseTo(1, 12);
  });
});

describe('pca: plan case 4 — centering matters (measured contrast)', () => {
  const CENTER = { n: 40, corr: 0.7, rotDeg: 80, noise: 0.15, seed: 42 };

  it('centered PCA finds PC1 at 127.15°; RAW (uncentered) PCA finds 33.30° — different (wrong) PCs', () => {
    const sw = getSweep(CENTER);
    const data = sw.data;
    const mu = sw.mu;
    const eigC = eigen2x2(centeredCov(data, mu));
    const eigRaw = eigen2x2(rawCov(data));
    // centered (correct) PC angle
    expect(eigC.angleDeg).toBeCloseTo(127.15210735118922, 6);
    // raw (uncentered) PC angle — points at the mean, NOT the variance
    expect(eigRaw.angleDeg).toBeCloseTo(33.29706869064782, 6);
    // the mean direction is ≈ 34.1° — the raw PC points at the mean
    const meanAngle = (Math.atan2(mu[1], mu[0]) * 180) / Math.PI;
    expect(meanAngle).toBeCloseTo(34.063263145264045, 3);
    expect(Math.abs(eigRaw.angleDeg - meanAngle)).toBeLessThan(2); // raw PC ≈ mean direction
    expect(Math.abs(eigC.angleDeg - eigRaw.angleDeg)).toBeGreaterThan(50);
    expect(Math.abs(eigC.angleDeg - eigRaw.angleDeg)).toBeCloseTo(93.8550386605414, 3);
  });

  it('the raw PC1 explains only 12.6% of the CENTERED variance — quantitatively worse', () => {
    const sw = getSweep(CENTER);
    const data = sw.data;
    const mu = sw.mu;
    const eigC = eigen2x2(centeredCov(data, mu));
    const eigRaw = eigen2x2(rawCov(data));
    const centeredVarAlongRawPC1 = varianceAlong(data, mu, eigRaw.v1);
    // raw PC1's variance measured on the centered data
    expect(centeredVarAlongRawPC1).toBeCloseTo(0.2750802289310812, 6);
    // fraction of centered variance it captures
    expect(centeredVarAlongRawPC1 / eigC.totalVariance).toBeCloseTo(0.12582945811006516, 6);
    // vs the true PC1's 87.76%
    expect(eigC.lambda1 / eigC.totalVariance).toBeCloseTo(0.8775840456844758, 6);
    // and the raw covariance's dominant eigenvalue is the mean-offset term ‖μ‖² ≈ 8.9
    expect(eigRaw.lambda1).toBeCloseTo(8.555894201495349, 6);
  });

  it('rawCov = centeredCov + μμᵀ holds exactly (the algebraic source of the failure)', () => {
    const sw = getSweep(CENTER);
    const { data, mu } = sw;
    const rc = rawCov(data);
    const cc = centeredCov(data, mu);
    // (1/n)XᵀX = Σ + μμᵀ
    expect(rc[0][0]).toBeCloseTo(cc[0][0] + mu[0] * mu[0], 9);
    expect(rc[0][1]).toBeCloseTo(cc[0][1] + mu[0] * mu[1], 9);
    expect(rc[1][1]).toBeCloseTo(cc[1][1] + mu[1] * mu[1], 9);
  });
});

describe('pca: plan case 5 — reconstruction error = sum of dropped eigenvalues', () => {
  it('reconError(k=1) = λ₂ exactly; reconError(k=2) = 0 (both to 1e-12)', () => {
    const sw = getSweep(DEFAULT);
    const e1 = reconError(sw.data, sw.mu, sw.eig, 1);
    const e2 = reconError(sw.data, sw.mu, sw.eig, 2);
    expect(e1).toBeCloseTo(sw.eig.lambda2, 12);
    expect(e1).toBeCloseTo(0.2784478979012177, 9);
    expect(Math.abs(e2)).toBeLessThan(1e-12);
  });

  it('the run\'s final metrics report reconErrK1 = λ₂ and reconErrK2 = 0', () => {
    const run = computeRun(simulation, DEFAULT, 500);
    const m = run.snapshots[run.snapshots.length - 1].metrics;
    expect(m.reconErrK1).toBeCloseTo(m.lambda2, 12);
    expect(m.reconErrK2).toBe(0);
  });

  it('reconstruction error identity holds on the other measured configs too', () => {
    const cfg2 = { n: 60, corr: 0.3, rotDeg: 60, noise: 0.25, seed: 123 };
    const sw2 = getSweep(cfg2);
    expect(reconError(sw2.data, sw2.mu, sw2.eig, 1)).toBeCloseTo(sw2.eig.lambda2, 12);
    const sw7 = getSweep({ ...DEFAULT, seed: 7 });
    expect(reconError(sw7.data, sw7.mu, sw7.eig, 1)).toBeCloseTo(sw7.eig.lambda2, 12);
  });
});

describe('pca: determinism + sweep integrity', () => {
  it('same seed → identical snapshot arrays; different seed → different data', () => {
    const r1 = computeRun(simulation, DEFAULT, 500);
    const r2 = computeRun(simulation, DEFAULT, 500);
    expect(JSON.stringify(r1.snapshots)).toBe(JSON.stringify(r2.snapshots));
    expect(r1.telemetry.snapshotCount).toBe(r2.telemetry.snapshotCount);
    const r3 = computeRun(simulation, { ...DEFAULT, seed: 7 }, 500);
    expect(JSON.stringify(r1.snapshots)).not.toBe(JSON.stringify(r3.snapshots));
  });

  it('the run is a 37-step sweep (36 grid directions + 1 closed-form final), converging with the exact-solution event', () => {
    const run = computeRun(simulation, DEFAULT, 500);
    expect(run.snapshots).toHaveLength(37); // ANGLE_COUNT + 1
    expect(run.telemetry.failedAtStep).toBeUndefined();
    const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
    expect(labels).toContain('exact-2x2-pca-solution');
    // the first snapshot is grid angle 0° (x-direction): uᵀΣu = Σ₁₁
    expect(run.snapshots[0].metrics.angleDeg).toBeCloseTo(0, 9);
    expect(run.snapshots[0].metrics.axisVariance).toBeCloseTo(0.37011827663219543, 9);
    // every sweep snapshot reports the running best so far
    for (let i = 0; i < 36; i++) {
      expect(run.snapshots[i].metrics.step).toBe(i + 1);
      expect(run.snapshots[i].metrics.isOptimal).toBe(0);
    }
  });

  it('the centered covariance and PC1 match the measured Σ exactly', () => {
    const sw = getSweep(DEFAULT);
    expect(sw.Sigma[0][0]).toBeCloseTo(0.37011827663219543, 12);
    expect(sw.Sigma[0][1]).toBeCloseTo(0.37911198787583555, 12);
    expect(sw.Sigma[1][1]).toBeCloseTo(1.8463033092217696, 12);
    expect(sw.eig.v1[0]).toBeCloseTo(0.23502957205216105, 12);
    expect(sw.eig.v1[1]).toBeCloseTo(0.9719882202274769, 12);
  });
});

describe('pca: validateParams', () => {
  it('rejects degenerate sizes and ranges', () => {
    const issues = pcaModule.validateParams?.({ ...DEFAULT, n: 2 }) ?? [];
    expect(issues.some((s) => /n must be an integer ≥ 3/.test(s))).toBe(true);
    const big = pcaModule.validateParams?.({ ...DEFAULT, n: 81 }) ?? [];
    expect(big.some((s) => /n ≤ 80/.test(s))).toBe(true);
    const corr = pcaModule.validateParams?.({ ...DEFAULT, corr: 1 }) ?? [];
    expect(corr.some((s) => /corr must be in/.test(s))).toBe(true);
    const rot = pcaModule.validateParams?.({ ...DEFAULT, rotDeg: 180 }) ?? [];
    expect(rot.some((s) => /rotDeg must be in/.test(s))).toBe(true);
    const noise = pcaModule.validateParams?.({ ...DEFAULT, noise: -0.1 }) ?? [];
    expect(noise.some((s) => /noise/.test(s))).toBe(true);
  });

  it('rejects bad seeds and warns on the near-degenerate combination', () => {
    const seed = pcaModule.validateParams?.({ ...DEFAULT, seed: 10000 }) ?? [];
    expect(seed.some((s) => /seed must be an integer/.test(s))).toBe(true);
    const warn = pcaModule.validateParams?.({ ...DEFAULT, noise: 0, corr: 0.98 }) ?? [];
    expect(warn.some((s) => /near degenerate/.test(s) || /degenerate/.test(s))).toBe(true);
  });

  it('validates the points override (failure-demo datasets) and rejects malformed JSON', () => {
    const ok = pcaModule.validateParams?.({ ...DEFAULT, points: '[[0,0],[1,1],[2,2]]' }) ?? [];
    expect(ok.length).toBe(0);
    const bad = pcaModule.validateParams?.({ ...DEFAULT, points: '[[0,0],[1]]' }) ?? [];
    expect(bad.some((s) => /JSON array of ≥ 3/.test(s))).toBe(true);
    const zeroVar = pcaModule.validateParams?.({ ...DEFAULT, points: '[[1,1],[1,1],[1,1]]' }) ?? [];
    expect(zeroVar.some((s) => /zero variance/.test(s))).toBe(true);
  });

  it('accepts the default parameter set', () => {
    const issues = pcaModule.validateParams?.(DEFAULT) ?? [];
    expect(issues.length).toBe(0);
  });
});

describe('pca: honest telemetry failure on degenerate data', () => {
  it('zero-variance points (all identical) fail cleanly via telemetry, no NaN', () => {
    const run = computeRun(simulation, {
      ...DEFAULT,
      points: '[[1,1],[1,1],[1,1]]',
    }, 500);
    expect(run.telemetry.failedAtStep).toBeDefined();
    expect(run.telemetry.failureReason).toMatch(/zero-?variance/i);
    // no snapshot carries NaN/Infinity in metrics
    for (const s of run.snapshots) {
      for (const v of Object.values(s.metrics)) expect(Number.isFinite(v)).toBe(true);
    }
  });
});

describe('pca: registration + unsupervised contract', () => {
  it('register() is idempotent and registers the module under id "pca"', () => {
    register();
    register();
    // no throw = idempotent; module is reachable via the registry
    expect(pcaModule.id).toBe('pca');
    expect(pcaModule.title).toBe('Principal Component Analysis');
  });
});