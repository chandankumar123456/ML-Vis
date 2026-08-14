// src/topics/pca-svd/testCases.test.ts
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import {
  simulation, svdModule, register, getSweep,
  gramMatrix, matVec, eigen2x2Symmetric, computeSvd,
  diagMatrix, economyU, economyVt,
} from './module';
import { svdTestCases } from './testCases';

// Measured anchors (ALL verified by running the module — every number below was
// printed by the module before being asserted):
//   default (n 40, corr 0.7, rotDeg 30, noise 0.15, seed 42, rank 2):
//     XᵀX = [[14.804731065287818, 15.164479515033422],
//            [15.164479515033422, 73.85213236887078]]
//     λ(XᵀX) = 77.51894751810988, 11.137915916048712; λ(cov) = 1.9379736879527472, 0.2784478979012178
//     σ₁ = 8.804484511776364, σ₂ = 3.3373516320652685, ratio = 2.6381650729227606
//     V₁ = (0.23502957205216105, 0.9719882202274769) at 76.40663404901252°,
//     V₂ = (0.9719882202274769, −0.23502957205216105); v₁·v₂ = 0 exactly.
//     u₁·u₂ = 3.90e-18; ‖X − UΣVᵀ‖∞ = 4.44e-16; G·v = λ·v EXACTLY (0).
//     rank-1: errFro = errSpectral = σ₂ = 3.337351632065269, errMse = 0.278447897901218 = σ₂²/n.
//     rank-2: errFro = 1.28e-15, errSpec = 0, errMse = 4.12e-32.
//     rank-0 baseline (loss plateau) = ‖X‖²_F/n = 2.216421585853965 = λ₁/n + λ₂/n.
//     run = 6 snapshots (4 build-up + rank 1 + rank 2); rank-1 run = 5 snapshots.
//     mu = (2.5120649580032963, 1.6169993082874292).
//   seed 7: σ₁ = 9.460388559546717, σ₂ = 3.2502580275094646,
//     λ(cov) = 2.23747379244006, 0.26410443113474286, V₁ at 68.85395927722712°.
//   cfg2 (n 60, corr 0.3, rotDeg 60, noise 0.25, seed 123): σ₁ = 9.26783003261347,
//     σ₂ = 6.380133507392377, λ(cov) = 1.4315445585568694, 0.6784350595358491.
//   flip config (rotDeg 140, seed 2): raw v₁ = (−0.9977754850137721, 0.06666394456924787)
//     at 176.17760258315272° — largest |component| (x) NEGATIVE → sign-fixed to
//     (0.9977754850137721, −0.06666394456924787). σ₁·u₁·v₁ᵀ unchanged by the flip.
//   collinear '[[1,2],[3,4],[5,6],[7,8]]': σ₁ = 6.324555320336759, σ₂ = 0,
//     rankDeficient = 1, singularRatio saturated at 1e9; run completes (6 snapshots).

const DEFAULT = { n: 40, corr: 0.7, rotDeg: 30, noise: 0.15, seed: 42, rank: 2 };

// ---------------------------------------------------------------------------
// Data-driven test cases (from testCases.ts)
// ---------------------------------------------------------------------------
describe('pca-svd testCases (data-driven)', () => {
  for (const tc of svdTestCases) {
    it(tc.name, () => {
      const run = computeRun(simulation, tc.params, tc.maxSteps ?? 500);
      if (tc.expect.converged !== undefined) {
        if (tc.expect.converged) {
          expect(run.telemetry.failedAtStep, tc.name).toBeUndefined();
        } else {
          expect(run.telemetry.failedAtStep, tc.name).toBeDefined();
        }
      }
      if (tc.expect.eventLabels) {
        const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
        for (const lbl of tc.expect.eventLabels) expect(labels, tc.name).toContain(lbl);
      }
      if (tc.expect.finalMetrics) {
        const m = run.snapshots[run.snapshots.length - 1].metrics;
        for (const [k, pred] of Object.entries(tc.expect.finalMetrics)) {
          if (typeof pred === 'function') expect(pred(m[k]), `metric ${k} failed for ${tc.name}`).toBe(true);
          else expect(m[k], `metric ${k} failed for ${tc.name}`).toBeCloseTo(pred, 6);
        }
      }
      if (tc.expect.finalAlgorithm) {
        const a = run.snapshots[run.snapshots.length - 1].algorithm;
        for (const [k, pred] of Object.entries(tc.expect.finalAlgorithm)) {
          if (typeof pred === 'function') expect(pred(a[k]), `algorithm ${k} failed for ${tc.name}`).toBe(true);
          else expect(a[k], `algorithm ${k} failed for ${tc.name}`).toBe(pred);
        }
      }
    });
  }
});

// ---------------------------------------------------------------------------
// Plan case 1 — SVD gives the same "PCs" as the eigen-decomposition of XᵀX
// ---------------------------------------------------------------------------
describe('pca-svd: plan case 1 — V = eigenvectors of XᵀX (numeric)', () => {
  it('G·v_k = λ_k·v_k holds for both right singular vectors (exactly 0, measured)', () => {
    const sw = getSweep(DEFAULT);
    const { Gram, V1, V2, gramLambda1, gramLambda2 } = sw.core;
    const Gv1 = [
      Gram[0][0] * V1[0] + Gram[0][1] * V1[1],
      Gram[1][0] * V1[0] + Gram[1][1] * V1[1],
    ];
    const Gv2 = [
      Gram[0][0] * V2[0] + Gram[0][1] * V2[1],
      Gram[1][0] * V2[0] + Gram[1][1] * V2[1],
    ];
    expect(Gv1[0]).toBeCloseTo(gramLambda1 * V1[0], 12);
    expect(Gv1[1]).toBeCloseTo(gramLambda1 * V1[1], 12);
    expect(Gv2[0]).toBeCloseTo(gramLambda2 * V2[0], 12);
    expect(Gv2[1]).toBeCloseTo(gramLambda2 * V2[1], 12);
    // measured: the closed form gives exact eigenvectors (residual 0)
    expect(Math.abs(Gv1[0] - gramLambda1 * V1[0])).toBeLessThan(1e-15);
  });

  it('the closed-form eigen-solver and the module produce the SAME V (wiring check)', () => {
    const sw = getSweep(DEFAULT);
    const eig = eigen2x2Symmetric(sw.core.Gram);
    expect(eig.lambda1).toBeCloseTo(sw.core.gramLambda1, 12);
    expect(eig.lambda2).toBeCloseTo(sw.core.gramLambda2, 12);
    // sign convention applied identically
    expect(eig.rawV1[0]).toBeCloseTo(sw.core.rawV1[0], 12);
    expect(eig.rawV1[1]).toBeCloseTo(sw.core.rawV1[1], 12);
  });

  it('V is orthonormal and X = UΣVᵀ reconstructs the data exactly', () => {
    const sw = getSweep(DEFAULT);
    const c = sw.core;
    expect(Math.hypot(c.V1[0], c.V1[1])).toBeCloseTo(1, 12);
    expect(Math.hypot(c.V2[0], c.V2[1])).toBeCloseTo(1, 12);
    expect(c.V1[0] * c.V2[0] + c.V1[1] * c.V2[1]).toBeCloseTo(0, 12);
    // U orthonormal (u_k = Xv_k/σ_k is unit and u₁⊥u₂ by construction)
    const u1Norm = Math.sqrt(c.U1.reduce((a, x) => a + x * x, 0));
    expect(u1Norm).toBeCloseTo(1, 9);
    expect(Math.abs(c.U1.reduce((a, x, i) => a + x * c.U2[i], 0))).toBeLessThan(1e-12);
    // ‖X − UΣVᵀ‖∞ (measured 4.44e-16)
    let maxAbs = 0;
    for (let i = 0; i < c.X.length; i++) {
      const r0 = c.X[i][0] - (c.sigma1 * c.U1[i] * c.V1[0] + c.sigma2 * c.U2[i] * c.V2[0]);
      const r1 = c.X[i][1] - (c.sigma1 * c.U1[i] * c.V1[1] + c.sigma2 * c.U2[i] * c.V2[1]);
      maxAbs = Math.max(maxAbs, Math.abs(r0), Math.abs(r1));
    }
    expect(maxAbs).toBeLessThan(1e-12);
  });

  it('‖X·v_k‖ = σ_k — the norm identity behind u_k = Xv_k/σ_k', () => {
    const sw = getSweep(DEFAULT);
    const Xv1 = matVec(sw.core.X, sw.core.V1);
    const norm = Math.sqrt(Xv1.reduce((a, x) => a + x * x, 0));
    expect(norm).toBeCloseTo(sw.core.sigma1, 9);
    expect(norm).toBeCloseTo(8.804484511776364, 9);
  });

  it('V·Σ²·Vᵀ = XᵀX reconstructs the Gram matrix (the XᵀX = VΣ²Vᵀ identity)', () => {
    const sw = getSweep(DEFAULT);
    const c = sw.core;
    let maxDiff = 0;
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        let sum = 0;
        for (let r = 0; r < 2; r++) sum += c.Vt[r][i] * (r === 0 ? c.gramLambda1 : c.gramLambda2) * c.Vt[r][j];
        maxDiff = Math.max(maxDiff, Math.abs(sum - c.Gram[i][j]));
      }
    }
    // measured: 1.42e-14 (floating-point round-trip)
    expect(maxDiff).toBeLessThan(1e-9);
  });
});

// ---------------------------------------------------------------------------
// Plan case 2 — singular values relate to eigenvalues
// ---------------------------------------------------------------------------
describe('pca-svd: plan case 2 — σ_k² = λ_k of XᵀX = n·λ_k of covariance', () => {
  it('σ_k² = λ_k(XᵀX) exactly (measured: 77.51894751810988 / 11.137915916048712)', () => {
    const sw = getSweep(DEFAULT);
    expect(sw.core.sigma1 * sw.core.sigma1).toBeCloseTo(sw.core.gramLambda1, 9);
    expect(sw.core.sigma2 * sw.core.sigma2).toBeCloseTo(sw.core.gramLambda2, 9);
    expect(sw.core.sigma1 * sw.core.sigma1).toBeCloseTo(77.51894751810988, 9);
    expect(sw.core.sigma2 * sw.core.sigma2).toBeCloseTo(11.137915916048712, 9);
  });

  it('σ_k²/n = λ_k of the covariance (1/n)XᵀX — the PCA link', () => {
    const sw = getSweep(DEFAULT);
    const n = sw.core.X.length;
    expect((sw.core.sigma1 * sw.core.sigma1) / n).toBeCloseTo(sw.core.covLambda1, 12);
    expect((sw.core.sigma2 * sw.core.sigma2) / n).toBeCloseTo(sw.core.covLambda2, 12);
    expect(sw.core.covLambda1).toBeCloseTo(1.9379736879527472, 12);
    expect(sw.core.covLambda2).toBeCloseTo(0.2784478979012178, 12);
    // and the measured v₁ matches the PCA principal direction at 76.40663404901252°
    expect(Math.atan2(sw.core.V1[1], sw.core.V1[0]) * 180 / Math.PI).toBeCloseTo(76.40663404901252, 6);
  });

  it('singular values are ordered σ₁ ≥ σ₂ and both non-negative on every config', () => {
    for (const params of [
      DEFAULT,
      { ...DEFAULT, seed: 7 },
      { n: 60, corr: 0.3, rotDeg: 60, noise: 0.25, seed: 123, rank: 2 },
    ]) {
      const c = getSweep(params).core;
      expect(c.sigma1).toBeGreaterThanOrEqual(0);
      expect(c.sigma2).toBeGreaterThanOrEqual(0);
      expect(c.sigma1).toBeGreaterThanOrEqual(c.sigma2);
      expect(c.sigma1 / c.sigma2).toBeGreaterThanOrEqual(1);
    }
  });
});

// ---------------------------------------------------------------------------
// Plan case 3 — low-rank approximation: rank-k error = σ_{k+1} (Eckart-Young)
// ---------------------------------------------------------------------------
describe('pca-svd: plan case 3 — Eckart-Young low-rank approximation', () => {
  it('rank-1 error: ‖X−X̂₁‖_F = ‖X−X̂₁‖₂ = σ₂ and MSE = σ₂²/n (measured)', () => {
    const sw = getSweep(DEFAULT);
    const r = sw.recon[0];
    expect(r.errFro).toBeCloseTo(sw.core.sigma2, 9);
    expect(r.errSpectral).toBeCloseTo(sw.core.sigma2, 9);
    expect(r.errMse).toBeCloseTo((sw.core.sigma2 * sw.core.sigma2) / sw.core.X.length, 9);
    // measured anchors (full precision)
    expect(r.errFro).toBeCloseTo(3.33735163206527, 12);
    expect(r.errMse).toBeCloseTo(0.278447897901218, 12);
    expect(r.errMse).toBeCloseTo(sw.core.covLambda2, 12);
  });

  it('rank-2 error: the reconstruction is EXACT (σ₃ = 0, measured errMse = 4.12e-32)', () => {
    const sw = getSweep(DEFAULT);
    const r = sw.recon[1];
    expect(r.errFro).toBeLessThan(1e-12);
    expect(r.errSpectral).toBe(0);
    expect(r.errMse).toBeLessThan(1e-25);
  });

  it('Eckart-Young holds on the other measured configs (seed 7, cfg2)', () => {
    const sw7 = getSweep({ ...DEFAULT, seed: 7 });
    expect(sw7.recon[0].errFro).toBeCloseTo(sw7.core.sigma2, 9);
    expect(sw7.recon[0].errMse).toBeCloseTo(sw7.core.covLambda2, 9);
    expect(sw7.recon[0].errFro).toBeCloseTo(3.2502580275094646, 12);

    const cfg2 = { n: 60, corr: 0.3, rotDeg: 60, noise: 0.25, seed: 123, rank: 2 };
    const swC = getSweep(cfg2);
    expect(swC.recon[0].errFro).toBeCloseTo(swC.core.sigma2, 9);
    expect(swC.recon[0].errMse).toBeCloseTo(swC.core.covLambda2, 9);
    expect(swC.recon[0].errFro).toBeCloseTo(6.380133507392378, 12);
  });

  it('the loss curve (reconstructionError) is the Eckart-Young series with the rank-0 baseline', () => {
    const run = computeRun(simulation, DEFAULT, 500);
    const curve = run.snapshots.map((s) => s.metrics.reconstructionError);
    // build-up steps 1–4: rank-0 baseline ‖X‖²_F/n = (λ₁+λ₂)/n (measured)
    for (let i = 0; i < 4; i++) expect(curve[i]).toBeCloseTo(2.216421585853965, 9);
    // rank 1: σ₂²/n; rank 2: exactly 0
    expect(curve[4]).toBeCloseTo(0.278447897901218, 9);
    expect(curve[5]).toBeLessThan(1e-20);
    // the curve is monotonically non-increasing (Eckart-Young: more rank, less error)
    for (let i = 1; i < curve.length; i++) expect(curve[i]).toBeLessThanOrEqual(curve[i - 1] + 1e-12);
  });
});

// ---------------------------------------------------------------------------
// Plan case 4 — economy SVD shapes
// ---------------------------------------------------------------------------
describe('pca-svd: plan case 4 — economy SVD U (n×k), Σ (k×k), Vᵀ (k×d)', () => {
  it('k = 1: U₁ (40×1), Σ₁ (1×1), V₁ᵀ (1×2)', () => {
    const sw = getSweep(DEFAULT);
    const c = sw.core;
    const U1m = economyU(c, 1);
    const S1m = diagMatrix([c.sigma1, c.sigma2], 1);
    const Vt1 = economyVt(c, 1);
    expect(U1m).toHaveLength(40);
    expect(U1m[0]).toHaveLength(1);
    expect(S1m).toHaveLength(1);
    expect(S1m[0]).toHaveLength(1);
    expect(S1m[0][0]).toBeCloseTo(c.sigma1, 12);
    expect(Vt1).toHaveLength(1);
    expect(Vt1[0]).toHaveLength(2);
    expect(Vt1[0][0]).toBeCloseTo(c.V1[0], 12);
    expect(Vt1[0][1]).toBeCloseTo(c.V1[1], 12);
  });

  it('k = 2: U₂ (40×2), Σ₂ (2×2 diag), V₂ᵀ (2×2)', () => {
    const sw = getSweep(DEFAULT);
    const c = sw.core;
    const U2m = economyU(c, 2);
    const S2m = diagMatrix([c.sigma1, c.sigma2], 2);
    const Vt2 = economyVt(c, 2);
    expect(U2m).toHaveLength(40);
    expect(U2m[0]).toHaveLength(2);
    expect(S2m).toHaveLength(2);
    expect(S2m[0][0]).toBeCloseTo(c.sigma1, 12);
    expect(S2m[1][1]).toBeCloseTo(c.sigma2, 12);
    expect(S2m[0][1]).toBe(0);
    expect(S2m[1][0]).toBe(0);
    expect(Vt2).toHaveLength(2);
    expect(Vt2[0]).toHaveLength(2);
    expect(Vt2[1]).toHaveLength(2);
  });

  it('economy factors are the leading columns/rows of the full factors', () => {
    const sw = getSweep(DEFAULT);
    const c = sw.core;
    const U2m = economyU(c, 2);
    const Vt2 = economyVt(c, 2);
    for (let i = 0; i < c.U1.length; i++) {
      expect(U2m[i][0]).toBeCloseTo(c.U1[i], 12);
      expect(U2m[i][1]).toBeCloseTo(c.U2[i], 12);
    }
    expect(Vt2[0][0]).toBeCloseTo(c.V1[0], 12);
    expect(Vt2[0][1]).toBeCloseTo(c.V1[1], 12);
    expect(Vt2[1][0]).toBeCloseTo(c.V2[0], 12);
    expect(Vt2[1][1]).toBeCloseTo(c.V2[1], 12);
  });

  it('the reconstruction matrix X̂_k is n×2 with the measured rank-k error', () => {
    const sw = getSweep(DEFAULT);
    const r1 = sw.recon[0];
    const r2 = sw.recon[1];
    expect(r1.Xhat).toHaveLength(40);
    expect(r1.Xhat[0]).toHaveLength(2);
    expect(r2.Xhat).toHaveLength(40);
    expect(r2.Xhat[0]).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// Sign convention (the plan's "deterministic sign convention")
// ---------------------------------------------------------------------------
describe('pca-svd: deterministic sign convention (largest |component| positive)', () => {
  it('every sign-fixed singular vector has its largest |component| positive', () => {
    const sw = getSweep(DEFAULT);
    for (const v of [sw.core.V1, sw.core.V2]) {
      const largestNeg = (Math.abs(v[0]) >= Math.abs(v[1])) ? v[0] < 0 : v[1] < 0;
      expect(largestNeg).toBe(false);
    }
  });

  it('signFix flips a negative-dominant raw vector and keeps X = UΣVᵀ invariant', () => {
    // measured flip case (rotDeg 140, seed 2): raw v₁ = (−0.9978, 0.0667) at 176.18°
    const sw = getSweep({ ...DEFAULT, rotDeg: 140, seed: 2 });
    const c = sw.core;
    expect(c.rawV1[0]).toBeLessThan(0);                       // raw x-dominant NEGATIVE
    expect(c.rawV1[0]).toBeCloseTo(-0.9977754850137721, 12);
    expect(c.rawV1[1]).toBeCloseTo(0.06666394456924787, 12);
    expect(c.rawAngleDeg).toBeCloseTo(176.17760258315272, 6);
    // the convention flips to the positive orientation
    expect(c.V1[0]).toBeGreaterThan(0);
    expect(c.V1[0]).toBeCloseTo(0.9977754850137721, 12);
    expect(c.V1[1]).toBeCloseTo(-0.06666394456924787, 12);
    // flipping (v₁, u₁) together leaves σ₁u₁v₁ᵀ unchanged: recompute the RAW
    // u₁ = X·rawV₁/σ₁ (the module's U1 already carries the sign-fixed V1) and
    // compare the reconstruction term σ₁·u₁·v₁ᵀ under both orientations.
    const u1Raw = matVec(c.X, c.rawV1).map((v) => v / c.sigma1);
    const termFixed = c.sigma1 * c.U1[0] * c.V1[0];
    const termRaw = c.sigma1 * u1Raw[0] * c.rawV1[0];
    expect(Math.abs(termFixed - termRaw)).toBeLessThan(1e-15);
    // and the raw u₁ is the sign-flip of the fixed u₁ (measured: U1[0] = 0.00127)
    expect(u1Raw[0]).toBeCloseTo(-c.U1[0], 12);
  });

  it('the default config also exposes a flipped V₂ (raw (−0.972, 0.235) → (0.972, −0.235))', () => {
    const sw = getSweep(DEFAULT);
    const c = sw.core;
    expect(c.rawV2[0]).toBeLessThan(0);
    expect(c.rawV2[0]).toBeCloseTo(-0.9719882202274769, 12);
    expect(c.V2[0]).toBeGreaterThan(0);
    expect(c.V2[0]).toBeCloseTo(0.9719882202274769, 12);
  });
});

// ---------------------------------------------------------------------------
// Determinism + run shape
// ---------------------------------------------------------------------------
describe('pca-svd: determinism + sweep integrity', () => {
  it('same params → identical snapshot arrays; different seed → different data', () => {
    const r1 = computeRun(simulation, DEFAULT, 500);
    const r2 = computeRun(simulation, DEFAULT, 500);
    expect(JSON.stringify(r1.snapshots)).toBe(JSON.stringify(r2.snapshots));
    expect(r1.telemetry.snapshotCount).toBe(r2.telemetry.snapshotCount);
    const r3 = computeRun(simulation, { ...DEFAULT, seed: 7 }, 500);
    expect(JSON.stringify(r1.snapshots)).not.toBe(JSON.stringify(r3.snapshots));
  });

  it('rank-2 run: exactly 6 snapshots (4 build-up + rank 1 + rank 2), last = slider rank', () => {
    const run = computeRun(simulation, DEFAULT, 500);
    expect(run.snapshots).toHaveLength(6);
    expect(run.telemetry.failedAtStep).toBeUndefined();
    // build-up phases 1..4 then ranks 1..2 (steps 5..6)
    expect(run.snapshots.map((s) => s.metrics.step)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(run.snapshots.map((s) => s.metrics.rank)).toEqual([0, 0, 0, 0, 1, 2]);
    expect(run.snapshots[5].metrics.rank).toBe(2);
    expect(run.snapshots[5].metrics.isOptimal).toBe(1);
    expect(run.snapshots[4].metrics.isOptimal).toBe(0);
  });

  it('rank-1 run: exactly 5 snapshots; the loss is σ₂²/n and the run is optimal', () => {
    const run = computeRun(simulation, { ...DEFAULT, rank: 1 }, 500);
    expect(run.snapshots).toHaveLength(5);
    expect(run.snapshots[4].metrics.rank).toBe(1);
    expect(run.snapshots[4].metrics.reconstructionError).toBeCloseTo(0.278447897901218, 12);
    expect(run.snapshots[4].metrics.isOptimal).toBe(1);
    // build-up snapshots keep the rank-0 baseline (the loss-curve plateau)
    expect(run.snapshots[3].metrics.reconstructionError).toBeCloseTo(2.216421585853965, 9);
  });

  it('all 8 event labels fire in order on the default run', () => {
    const run = computeRun(simulation, DEFAULT, 500);
    const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
    expect(labels).toEqual([
      'svd-seeded-correlated-gaussian',
      'gram-matrix-XᵀX',
      'XᵀX-eigendecomposition',
      'singular-values-extracted',
      'factorization-complete',
      'rank-1-reconstruction',
      'rank-2-reconstruction',
      'svd-rank-sweep-complete',
    ]);
    // the mean is measured and stable
    const sw = getSweep(DEFAULT);
    expect(sw.mu[0]).toBeCloseTo(2.5120649580032963, 9);
    expect(sw.mu[1]).toBeCloseTo(1.6169993082874292, 9);
  });
});

// ---------------------------------------------------------------------------
// Rank-deficient data (honest handling)
// ---------------------------------------------------------------------------
describe('pca-svd: rank-deficient data handled honestly', () => {
  it('collinear points: σ₂ = 0, rankDeficient flagged, run completes', () => {
    const params = { ...DEFAULT, points: '[[1,2],[3,4],[5,6],[7,8]]' };
    const run = computeRun(simulation, params, 500);
    expect(run.telemetry.failedAtStep).toBeUndefined();
    expect(run.snapshots).toHaveLength(6);
    const m = run.snapshots[5].metrics;
    expect(m.sigma2).toBe(0);
    expect(m.rankDeficient).toBe(1);
    expect(m.singularRatio).toBe(1e9);            // saturated, finite
  });

  it('the null-space completion is a deterministic unit vector orthogonal to u₁', () => {
    const sw = getSweep({ ...DEFAULT, points: '[[1,2],[3,4],[5,6],[7,8]]' });
    const c = sw.core;
    expect(c.sigma1).toBeCloseTo(6.324555320336759, 9);
    // completion is unit and orthogonal
    const n1 = Math.hypot(c.U1[0], c.U1[1], c.U1[2], c.U1[3]);
    const n2 = Math.hypot(c.U2[0], c.U2[1], c.U2[2], c.U2[3]);
    expect(n1).toBeCloseTo(1, 9);
    expect(n2).toBeCloseTo(1, 9);
    const dot = c.U1.reduce((a, x, i) => a + x * c.U2[i], 0);
    expect(Math.abs(dot)).toBeLessThan(1e-9);
    // measured completion (deterministic)
    expect(c.U1[0]).toBeCloseTo(-0.6708203932499369, 9);
    expect(c.U2[0]).toBeCloseTo(0.7416198487095663, 9);
    // the reconstruction is still EXACT at rank 2 even though σ₂ = 0
    expect(sw.recon[1].errFro).toBeLessThan(1e-12);
  });
});

// ---------------------------------------------------------------------------
// Honest telemetry failures
// ---------------------------------------------------------------------------
describe('pca-svd: honest telemetry failures', () => {
  it('zero-variance points fail cleanly via telemetry, no non-finite metric escapes', () => {
    const run = computeRun(simulation, { ...DEFAULT, points: '[[2,3],[2,3],[2,3]]' }, 500);
    expect(run.telemetry.failedAtStep).toBeDefined();
    expect(run.telemetry.failureReason).toMatch(/zero-?variance/i);
    for (const s of run.snapshots) {
      for (const v of Object.values(s.metrics)) expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('a missing row (null element) fails cleanly — SVD needs a complete matrix', () => {
    const run = computeRun(simulation, { ...DEFAULT, points: '[[1,2],[3,4],null,[7,8]]' }, 500);
    expect(run.telemetry.failedAtStep).toBeDefined();
    expect(run.telemetry.failureReason).toBeTruthy();
  });

  it('the math core rejects a NaN Gram matrix explicitly (defense in depth)', () => {
    const G = gramMatrix([[1, 2], [NaN, 3], [4, 5]]);
    expect(Number.isNaN(G[0][0])).toBe(true);
    expect(() => eigen2x2Symmetric(G)).toThrow(/non-finite/);
    // and computeSvd on a NaN design matrix throws too (no silent NaN spread)
    expect(() => computeSvd([[1, 2], [NaN, 3], [4, 5], [6, 7]])).toThrow();
  });
});

// ---------------------------------------------------------------------------
// validateParams
// ---------------------------------------------------------------------------
describe('pca-svd: validateParams', () => {
  it('rejects degenerate sizes and ranges', () => {
    const n = svdModule.validateParams?.({ ...DEFAULT, n: 2 }) ?? [];
    expect(n.some((s) => /n must be an integer ≥ 3/.test(s))).toBe(true);
    const big = svdModule.validateParams?.({ ...DEFAULT, n: 81 }) ?? [];
    expect(big.some((s) => /n ≤ 80/.test(s))).toBe(true);
    const corr = svdModule.validateParams?.({ ...DEFAULT, corr: 1 }) ?? [];
    expect(corr.some((s) => /corr must be in/.test(s))).toBe(true);
    const rot = svdModule.validateParams?.({ ...DEFAULT, rotDeg: 180 }) ?? [];
    expect(rot.some((s) => /rotDeg must be in/.test(s))).toBe(true);
    const noise = svdModule.validateParams?.({ ...DEFAULT, noise: -0.1 }) ?? [];
    expect(noise.some((s) => /noise/.test(s))).toBe(true);
    const rank = svdModule.validateParams?.({ ...DEFAULT, rank: 3 }) ?? [];
    expect(rank.some((s) => /rank must be an integer/.test(s))).toBe(true);
  });

  it('rejects bad seeds and warns on the near-degenerate combination', () => {
    const seed = svdModule.validateParams?.({ ...DEFAULT, seed: 10000 }) ?? [];
    expect(seed.some((s) => /seed must be an integer/.test(s))).toBe(true);
    const warn = svdModule.validateParams?.({ ...DEFAULT, noise: 0, corr: 0.98 }) ?? [];
    expect(warn.some((s) => /degenerate/.test(s))).toBe(true);
  });

  it('validates the points override and rejects malformed / zero-variance input', () => {
    const ok = svdModule.validateParams?.({ ...DEFAULT, points: '[[0,0],[1,1],[2,2]]' }) ?? [];
    expect(ok.length).toBe(0);
    const bad = svdModule.validateParams?.({ ...DEFAULT, points: '[[0,0],[1]]' }) ?? [];
    expect(bad.some((s) => /JSON array of ≥ 3/.test(s))).toBe(true);
    const zeroVar = svdModule.validateParams?.({ ...DEFAULT, points: '[[1,1],[1,1],[1,1]]' }) ?? [];
    expect(zeroVar.some((s) => /zero variance/.test(s))).toBe(true);
    const nanPts = svdModule.validateParams?.({ ...DEFAULT, points: '[[1,2],[3,4],null,[7,8]]' }) ?? [];
    expect(nanPts.some((s) => /valid JSON/.test(s) || /≥ 3/.test(s))).toBe(true);
  });

  it('accepts the default parameter set', () => {
    const issues = svdModule.validateParams?.(DEFAULT) ?? [];
    expect(issues.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Registration + unsupervised contract
// ---------------------------------------------------------------------------
describe('pca-svd: registration + unsupervised contract', () => {
  it('register() is idempotent and the module carries the pca-svd contract', () => {
    register();
    register();
    expect(svdModule.id).toBe('pca-svd');
    expect(svdModule.title).toBe('Singular Value Decomposition (SVD)');
    expect(svdModule.lossMetricKey).toBe('reconstructionError');
    expect(svdModule.validateParams?.(DEFAULT) ?? []).toHaveLength(0);
  });
});