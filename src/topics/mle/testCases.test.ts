// src/topics/mle/testCases.test.ts
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import type { SnapshotRun } from '../../engine/types';
import { getTopic } from '../../registry/topicRegistry';
import {
  simulation, mleModule, register, generateStream, fitPrefix, sweepSizesOf,
  bernoulliMle, gaussianMle, linearMle, bernoulliLogLik, bernoulliLikelihood,
  bernoulliScore, gaussianLogLik, linearScoreVec, linearLogLik, likelihoodGrid,
} from './module';
import { mleTestCases } from './testCases';

// Measured anchors (every number below was printed by running the module):
//   coin seed 42, pTrue 0.7: k = 8 @10, 74 @100, 219 @300, 707 @1000;
//     p̂ = 0.8/0.74/0.73/0.707; pErr = 0.1/0.04/0.03/0.007;
//     score(p̂)@1000 = 1.14e−13; H(0.7) = 0.6108643020548935;
//     nll@10 = 0.5004024235381879, nll@1000 = 0.6048160237512594.
//   gaussian seed 42, μ=1, σ=1.5 (σ²=2.25):
//     n=10: μ̂=0.05971922381826435, σ̂²=0.8504281631796738, σ̂²ᵤ=0.9449201813107487 (10/9);
//     n=100: μ̂=0.7585326658425336, σ̂²=2.317688007784234, σ̂²ᵤ=2.341098997761853;
//     n=1000: μ̂=0.9293916772008841, σ̂²=2.2177338531455018, σ̂²ᵤ=2.219953806952454 (1000/999);
//     ℓ(MLE)@100 = −183.92236034171168 vs ℓ(0, 5) = −201.29634707151646.
//   linear seed 42, slope 1.5, intercept −0.5, noise 0.8:
//     n=100: slopê=1.590870319456517, intercept̂=−0.4672611084691402,
//     rss=71.0527268349805, σ̂²=RSS/n=0.7105272683498051, σ̂²ᵤ=RSS/(n−2)=0.7250278248467399,
//     XᵀX=[[100, 10.494344260543585],[10.494344260543585, 265.7561100467906]],
//     Xᵀy=[−30.030970040656385, 417.8799087558206], score@θ̂ ≈ [8.4e−15, 8.5e−14],
//     ℓ(θ̂,σ̂²)=−124.80645566634281; slopeErr 0.0953@10 → 0.0119@1000.

const COIN = { family: 'coin', n: 1000, seed: 42, pTrue: 0.7 };
const GAUSS = { family: 'gaussian', n: 100, seed: 42, muTrue: 1, sigmaTrue: 1.5 };
const LIN = { family: 'linear', n: 100, seed: 42, slopeTrue: 1.5, interceptTrue: -0.5, noiseSigma: 0.8 };

function lastMetrics(run: SnapshotRun): Record<string, number> {
  return run.snapshots[run.snapshots.length - 1].metrics;
}

describe('mle testCases (data-driven — the plan\'s 5 required cases)', () => {
  for (const tc of mleTestCases) {
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
        const m = lastMetrics(run);
        for (const [k, pred] of Object.entries(tc.expect.finalMetrics)) {
          if (typeof pred === 'function') expect(pred(m[k]), `metric ${k} failed for ${tc.name}`).toBe(true);
          else expect(m[k]).toBeCloseTo(pred, 6);
        }
      }
    });
  }
});

describe('mle: plan case 1 — MLE recovers the true Bernoulli parameter (measured)', () => {
  it('p̂ = k/n exactly, and the seeded run measures k = 707, p̂ = 0.707 at n=1000', () => {
    const run = computeRun(simulation, COIN, 500);
    const m = lastMetrics(run);
    expect(m.k).toBe(707);
    expect(m.pHat).toBeCloseTo(707 / 1000, 12);   // p̂ = k/n by construction
    expect(m.pHat).toBeCloseTo(0.707, 6);
    expect(m.pErr).toBeCloseTo(0.007, 6);
    // the same empirical-frequency identity holds at every sweep prefix
    const stream = generateStream(COIN);
    expect(bernoulliMle(stream.coin.slice(0, 10)).k).toBe(8);
    expect(bernoulliMle(stream.coin.slice(0, 100)).k).toBe(74);
  });

  it('the score (numeric gradient of ℓ) vanishes at p̂: 1.14e−13 on the seeded stream', () => {
    const k = 707, n = 1000, pHat = 0.707;
    expect(bernoulliScore(k, n, pHat)).toBeCloseTo(0, 9);
    // central-difference derivative of ℓ at p̂ — the numeric gradient check
    const h = 1e-6;
    const numDeriv = (bernoulliLogLik(k, n, pHat + h) - bernoulliLogLik(k, n, pHat - h)) / (2 * h);
    expect(numDeriv).toBeCloseTo(0, 6);
    // the score has the correct sign on both sides of the MLE
    expect(bernoulliScore(k, n, pHat - 0.1)).toBeGreaterThan(0);
    expect(bernoulliScore(k, n, pHat + 0.1)).toBeLessThan(0);
  });

  it('nll per sample descends toward the true entropy H(0.7) = 0.6108643020548935', () => {
    const run = computeRun(simulation, COIN, 500);
    const nll = run.snapshots.map((s) => s.metrics.nllPerSample);
    expect(run.snapshots).toHaveLength(5);            // [10, 30, 100, 300, 1000]
    expect(nll[0]).toBeCloseTo(0.5004024235381879, 6);
    expect(nll[4]).toBeCloseTo(0.6048160237512594, 6);
    expect(nll[4]).toBeLessThan(0.6108643020548935);  // below the true entropy
  });
});

describe('mle: plan case 2 — Gaussian MLE: μ̂, and the BIASED ÷n variance (measured)', () => {
  it('μ̂ equals the sample mean exactly and ℓ is maximized at (μ̂, σ̂²)', () => {
    const stream = generateStream(GAUSS);
    const f = gaussianMle(stream.gauss.slice(0, 100));
    const mean = stream.gauss.slice(0, 100).reduce((a, b) => a + b, 0) / 100;
    expect(f.muHat).toBeCloseTo(mean, 12);
    expect(f.muHat).toBeCloseTo(0.7585326658425336, 6);
    expect(f.sigmaHatSq).toBeCloseTo(2.317688007784234, 6);
    // measured: ℓ at the MLE beats ℓ at a wrong (μ, σ²)
    expect(gaussianLogLik(stream.gauss, f.muHat, f.sigmaHatSq)).toBeCloseTo(-183.92236034171168, 6);
    expect(gaussianLogLik(stream.gauss, 0, 5)).toBeCloseTo(-201.29634707151646, 6);
  });

  it('at n=10 the MLE variance is the ÷n estimator: σ̂²ᵤ = (10/9)·σ̂², gap 0.0945', () => {
    const stream = generateStream({ ...GAUSS, n: 10 });
    const f = gaussianMle(stream.gauss.slice(0, 10));
    expect(f.sigmaHatSq).toBeCloseTo(0.8504281631796738, 6);   // Σ(x−μ̂)²/n — the MLE
    expect(f.sigmaUnbSq).toBeCloseTo(0.9449201813107487, 6);   // Σ(x−μ̂)²/(n−1)
    expect(f.sigmaUnbSq).toBeCloseTo(f.sigmaHatSq * (10 / 9), 9); // ÷n vs ÷(n−1) identity
    expect(f.sigmaUnbSq - f.sigmaHatSq).toBeCloseTo(0.09449201813107488, 6);
    const run = computeRun(simulation, { ...GAUSS, n: 10 }, 500);
    expect(lastMetrics(run).biasGapRel).toBeCloseTo(1 / 10, 9); // the ÷n bias is 1/n of σ̂²ᵤ
  });

  it('the bias gap shrinks with n: at n=1000 σ̂²ᵤ = (1000/999)·σ̂² and the gap is 0.0022', () => {
    const stream = generateStream({ ...GAUSS, n: 1000 });
    const f = gaussianMle(stream.gauss.slice(0, 1000));
    expect(f.sigmaHatSq).toBeCloseTo(2.2177338531455018, 6);
    expect(f.sigmaUnbSq).toBeCloseTo(2.219953806952454, 6);
    expect(f.sigmaUnbSq).toBeCloseTo(f.sigmaHatSq * (1000 / 999), 9);
    expect(f.sigmaUnbSq - f.sigmaHatSq).toBeCloseTo(0.0022199538069522795, 6);
  });
});

describe('mle: plan case 3 — log-likelihood maximization: argmax of L and ℓ coincide', () => {
  it('L(p) and ℓ(p) peak at the same p = p̂ = 0.8 (n=10, k=8, seeded)', () => {
    const p = { family: 'coin', n: 10, seed: 42, pTrue: 0.7 };
    const stream = generateStream(p);
    const k = bernoulliMle(stream.coin.slice(0, 10)).k;
    expect(k).toBe(8);
    let bestL = -Infinity, bestLL = -Infinity, argL = -1, argLL = -1;
    for (let i = 0; i <= 20; i++) {
      const pv = i / 20;
      const lv = bernoulliLogLik(k, 10, pv);
      const Lv = bernoulliLikelihood(k, 10, pv);
      if (lv > bestLL) { bestLL = lv; argLL = pv; }
      if (Lv > bestL) { bestL = Lv; argL = pv; }
    }
    expect(argLL).toBeCloseTo(0.8, 6);
    expect(argL).toBeCloseTo(0.8, 6);
    expect(argLL).toBe(argL);          // identical maximizer — the monotonicity claim
  });

  it('numeric gradient check: ℓ rises into p̂ and falls past it; L underflows to 0 at p=0.9 (n=1000)', () => {
    const k = 8, n = 10, pHat = 0.8;
    expect(bernoulliLogLik(k, n, pHat - 0.05)).toBeLessThan(bernoulliLogLik(k, n, pHat));
    expect(bernoulliLogLik(k, n, pHat + 0.05)).toBeLessThan(bernoulliLogLik(k, n, pHat));
    // why we maximize ℓ instead of L: the product L(0.9) underflows to 0 at n=1000
    expect(bernoulliLikelihood(707, 1000, 0.9)).toBe(0);
    expect(Number.isFinite(bernoulliLogLik(707, 1000, 0.9))).toBe(true);
    expect(bernoulliLikelihood(707, 1000, 0.707)).toBeCloseTo(2.1465367000805625e-263, 2);
  });

  it('gaussian: the ℓ(μ, σ²) grid peaks at the μ̂ center row for EVERY σ² column (surface substitution)', () => {
    const stream = generateStream(GAUSS);
    const f = gaussianMle(stream.gauss.slice(0, 100));
    const grid = likelihoodGrid(stream.gauss.slice(0, 100), f.muHat, f.sigmaHatSq);
    expect(grid).toHaveLength(9);
    for (let c = 0; c < 9; c++) {
      let best = -Infinity, bestRow = -1;
      for (let r = 0; r < 9; r++) {
        if (grid[r][c]! > best) { best = grid[r][c]!; bestRow = r; }
      }
      expect(bestRow, `column ${c} argmax row`).toBe(4);   // μ̂ is exactly row 4
    }
    let best = -Infinity, br = -1;
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        if (grid[r][c]! > best) { best = grid[r][c]!; br = r; }
      }
    }
    expect(br).toBe(4);                                     // global argmax row = μ̂
  });
});

describe('mle: plan case 4 — MLE = OLS for Gaussian noise (normal equation, numeric)', () => {
  it('linearMle equals (XᵀX)⁻¹Xᵀy via the 2×2 adjugate on the seeded design', () => {
    const stream = generateStream(LIN);
    const xs = stream.linX.slice(0, 100), ys = stream.linY.slice(0, 100);
    const f = linearMle(xs, ys);
    let sx = 0, sxx = 0, sy = 0, sxy = 0;
    for (let i = 0; i < 100; i++) {
      sx += xs[i]!; sxx += xs[i]! ** 2; sy += ys[i]!; sxy += xs[i]! * ys[i]!;
    }
    expect(sx).toBeCloseTo(10.494344260543585, 9);
    expect(sxx).toBeCloseTo(265.7561100467906, 9);
    expect(sy).toBeCloseTo(-30.030970040656385, 9);
    expect(sxy).toBeCloseTo(417.8799087558206, 9);
    const det = 100 * sxx - sx * sx;
    const slopeHat = (100 * sxy - sx * sy) / det;
    const interceptHat = (sxx * sy - sx * sxy) / det;
    expect(f.slopeHat).toBeCloseTo(slopeHat, 12);
    expect(f.interceptHat).toBeCloseTo(interceptHat, 12);
    expect(f.slopeHat).toBeCloseTo(1.590870319456517, 6);
    expect(f.interceptHat).toBeCloseTo(-0.4672611084691402, 6);
  });

  it('the matrix score Xᵀ(y − Xθ) is zero at θ̂ (measured 8.4e−15) and nonzero elsewhere', () => {
    const stream = generateStream(LIN);
    const xs = stream.linX.slice(0, 100), ys = stream.linY.slice(0, 100);
    const f = linearMle(xs, ys);
    const [s0, s1] = linearScoreVec(xs, ys, f.interceptHat, f.slopeHat);
    expect(s0).toBeCloseTo(0, 9);
    expect(s1).toBeCloseTo(0, 9);
    const [t0, t1] = linearScoreVec(xs, ys, 0, 0);
    expect(t0).toBeCloseTo(-30.030970040656385, 6);   // = Xᵀy at θ = 0
    expect(t1).toBeCloseTo(417.8799087558206, 6);
  });

  it('RSS and ℓ are optimized at θ̂: perturbing θ increases RSS and lowers ℓ', () => {
    const stream = generateStream(LIN);
    const xs = stream.linX.slice(0, 100), ys = stream.linY.slice(0, 100);
    const f = linearMle(xs, ys);
    const llAt = (b0: number, b1: number) => linearLogLik(xs, ys, b0, b1, f.sigmaHatSq);
    expect(f.rss).toBeCloseTo(71.0527268349805, 6);
    expect(f.sigmaHatSq).toBeCloseTo(f.rss / 100, 12);      // σ̂² = RSS/n (÷n)
    expect(llAt(f.interceptHat, f.slopeHat)).toBeCloseTo(-124.80645566634281, 6);
    expect(llAt(f.interceptHat + 0.1, f.slopeHat + 0.1)).toBeLessThan(llAt(f.interceptHat, f.slopeHat));
    expect(llAt(f.interceptHat - 0.1, f.slopeHat - 0.1)).toBeLessThan(llAt(f.interceptHat, f.slopeHat));
  });
});

describe('mle: plan case 5 — consistency: estimates improve with n (seeded)', () => {
  it('coin: |p̂ − p| falls 0.1 → 0.04 → 0.007 over n = 10 → 100 → 1000', () => {
    const base = { family: 'coin', seed: 42, pTrue: 0.7 };
    const e10 = lastMetrics(computeRun(simulation, { ...base, n: 10 }, 500)).pErr;
    const e100 = lastMetrics(computeRun(simulation, { ...base, n: 100 }, 500)).pErr;
    const e1000 = lastMetrics(computeRun(simulation, { ...base, n: 1000 }, 500)).pErr;
    expect(e10).toBeCloseTo(0.1, 6);
    expect(e100).toBeCloseTo(0.04, 6);
    expect(e1000).toBeCloseTo(0.007, 6);
    expect(e100).toBeLessThan(e10);
    expect(e1000).toBeLessThan(e100);
  });

  it('gaussian: μ̂ error 0.940 → 0.071 and σ̂² error 1.400 → 0.032 as n: 10 → 1000', () => {
    const base = { family: 'gaussian', seed: 42, muTrue: 1, sigmaTrue: 1.5 };
    const m10 = lastMetrics(computeRun(simulation, { ...base, n: 10 }, 500));
    const m100 = lastMetrics(computeRun(simulation, { ...base, n: 100 }, 500));
    const m1000 = lastMetrics(computeRun(simulation, { ...base, n: 1000 }, 500));
    expect(m10.muErr).toBeCloseTo(0.9402807761817357, 6);
    expect(m100.muErr).toBeCloseTo(0.24146733415746635, 6);
    expect(m1000.muErr).toBeCloseTo(0.07060832279911589, 6);
    expect(m1000.muErr).toBeLessThan(m100.muErr);
    expect(m10.sigmaErr).toBeCloseTo(1.3995718368203263, 6);
    expect(m1000.sigmaErr).toBeCloseTo(0.032266146854498245, 6);
    expect(m1000.sigmaErr).toBeLessThan(m100.sigmaErr);
    // σ̂² converges on the TRUE 2.25 from below (the ÷n bias disappears)
    expect(m1000.sigmaHatSq).toBeCloseTo(2.2177338531455018, 6);
  });

  it('linear: slope error 0.095 → 0.091 → 0.012 and intercept error 0.322 → 0.039', () => {
    const base = { family: 'linear', seed: 42, slopeTrue: 1.5, interceptTrue: -0.5, noiseSigma: 0.8 };
    const m10 = lastMetrics(computeRun(simulation, { ...base, n: 10 }, 500));
    const m100 = lastMetrics(computeRun(simulation, { ...base, n: 100 }, 500));
    const m1000 = lastMetrics(computeRun(simulation, { ...base, n: 1000 }, 500));
    expect(m10.slopeErr).toBeCloseTo(0.09532705183618151, 6);
    expect(m100.slopeErr).toBeCloseTo(0.09087031945651702, 6);
    expect(m1000.slopeErr).toBeCloseTo(0.011874967755075128, 6);
    expect(m1000.slopeErr).toBeLessThan(m100.slopeErr);
    // intercept: 0.322 at n=10 → 0.039 at n=1000 (consistency is a limit property,
    // not monotone per n — measured: 0.0327@100, 0.0389@1000)
    expect(m10.interceptErr).toBeCloseTo(0.3220066033571674, 6);
    expect(m1000.interceptErr).toBeCloseTo(0.03885314227432041, 6);
    expect(m1000.interceptErr).toBeLessThan(m10.interceptErr);
  });
});

describe('mle: module integrity (determinism, sweep, honest failure)', () => {
  it('same params → identical snapshots; sweep is [10, 30, 100, 300, 1000]; off-grid n=55 → [10, 30, 55]', () => {
    const r1 = computeRun(simulation, COIN, 500);
    const r2 = computeRun(simulation, COIN, 500);
    expect(JSON.stringify(r1.snapshots)).toBe(JSON.stringify(r2.snapshots));
    const r3 = computeRun(simulation, { ...COIN, seed: 7 }, 500);
    expect(JSON.stringify(r1.snapshots)).not.toBe(JSON.stringify(r3.snapshots));
    expect(sweepSizesOf({ n: 1000 })).toEqual([10, 30, 100, 300, 1000]);
    expect(sweepSizesOf({ n: 55 })).toEqual([10, 30, 55]);
    expect(sweepSizesOf({ n: 10 })).toEqual([10]);
    expect(r1.snapshots.map((s) => s.metrics.n)).toEqual([10, 30, 100, 300, 1000]);
  });

  it('honest prefix chain: snapshot k uses a strict prefix of the SAME stream (k300=219, k1000=707); register() lands the module', () => {
    const stream = generateStream(COIN);
    expect(fitPrefix(COIN, stream, 300).k).toBe(219);
    expect(fitPrefix(COIN, stream, 1000).k).toBe(707);
    const tail = stream.coin.slice(300, 1000).reduce((a, b) => a + b, 0);
    expect(219 + tail).toBe(707);   // 300→1000 adds only the new flips — no re-seeding
    register();
    register();                     // idempotent
    expect(getTopic('mle')?.id).toBe('mle');
    expect(getTopic('mle')?.lossMetricKey).toBe('nllPerSample');
  });

  it('non-identifiable linear design (all x equal) fails honestly via telemetry, with a flat likelihood', () => {
    const run = computeRun(simulation, { family: 'linear', points: '[[1,2],[1,5],[1,8]]' }, 500);
    expect(run.telemetry.failedAtStep).toBeDefined();
    expect(run.telemetry.failureReason).toMatch(/singular|identifiable/i);
    // the flat likelihood: every θ with β₀ + β₁·1 = 5 gives the SAME ℓ (measured −6.9445)
    const xs = [1, 1, 1], ys = [2, 5, 8];
    const sig2 = 18 / 3;   // RSS = (2−5)² + (5−5)² + (8−5)² = 18, ÷n
    const llA = linearLogLik(xs, ys, 5, 0, sig2);
    const llB = linearLogLik(xs, ys, 3, 2, sig2);
    const llC = linearLogLik(xs, ys, 0, 5, sig2);
    expect(llA).toBeCloseTo(-6.9444548034561, 6);
    expect(llB).toBeCloseTo(llA, 9);
    expect(llC).toBeCloseTo(llA, 9);
    expect(3 * 3 - 3 * 3).toBe(0);   // det(XᵀX) = n·Σx² − (Σx)² = 0 — not identifiable
  });
});

describe('mle: validateParams', () => {
  it('accepts the default parameter sets for all three families', () => {
    expect(mleModule.validateParams?.(COIN) ?? []).toHaveLength(0);
    expect(mleModule.validateParams?.(GAUSS) ?? []).toHaveLength(0);
    expect(mleModule.validateParams?.(LIN) ?? []).toHaveLength(0);
  });

  it('rejects bad family, n outside [2, 1000], non-positive noise, and malformed points', () => {
    const fam = mleModule.validateParams?.({ ...COIN, family: 'poisson' }) ?? [];
    expect(fam.some((s) => /family/.test(s))).toBe(true);
    const n1 = mleModule.validateParams?.({ ...COIN, n: 1 }) ?? [];
    expect(n1.some((s) => /n/.test(s))).toBe(true);
    const nbig = mleModule.validateParams?.({ ...COIN, n: 1001 }) ?? [];
    expect(nbig.some((s) => /1000/.test(s))).toBe(true);
    const lin2 = mleModule.validateParams?.({ ...LIN, n: 2 }) ?? [];
    expect(lin2.some((s) => /linear/.test(s) && /3/.test(s))).toBe(true);
    const noise = mleModule.validateParams?.({ ...LIN, noiseSigma: 0 }) ?? [];
    expect(noise.some((s) => /noiseSigma/.test(s))).toBe(true);
    const pts = mleModule.validateParams?.({ family: 'linear', points: '[[1,1]]' }) ?? [];
    expect(pts.some((s) => /points/.test(s))).toBe(true);
  });
});