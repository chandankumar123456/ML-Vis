// Pure-helper coverage for the distribution-view registry view (Task 15):
// the Gaussian PDF, sigma, x/y range fitting, the ±1σ shaded band clipped to a
// range, and collecting/sanitizing class densities emitted as visual commands.
import { describe, it, expect } from 'vitest';
import {
  gaussianPdf, sigmaOf, fitXRange, fitYRange, sigmaBandClipped,
  sanitizeDistributions, collectDistributions,
} from './distribution';
import type { VisualCommand } from '../engine/types';

// 1 / sqrt(2pi) — the standard-normal peak
const STD_PEAK = 0.3989422804014327;

describe('gaussianPdf', () => {
  it('peaks at the mean with 1/(σ sqrt(2π)) for N(0,1)', () => {
    expect(gaussianPdf(0, 0, 1)).toBeCloseTo(STD_PEAK, 9);
  });

  it('is symmetric about the mean', () => {
    expect(gaussianPdf(1, 0, 1)).toBeCloseTo(0.24197072451914337, 9);
    expect(gaussianPdf(-1, 0, 1)).toBeCloseTo(gaussianPdf(1, 0, 1), 12);
  });

  it('scales the peak by 1/σ for wider distributions', () => {
    // N(0, 4): peak = 1/(2 sqrt(2π)) = STD_PEAK / 2
    expect(gaussianPdf(0, 0, 4)).toBeCloseTo(STD_PEAK / 2, 9);
  });

  it('evaluates an off-center sample exactly', () => {
    // pdf(5 | N(3,4)) = exp(-(2²)/(2·4)) / (2 sqrt(2π)) = e^-1/2 / (2 sqrt(2π))
    expect(gaussianPdf(5, 3, 4)).toBeCloseTo(0.12098536225957168, 9);
  });

  it('returns 0 for degenerate or corrupted distributions', () => {
    expect(gaussianPdf(0, 0, 0)).toBe(0);
    expect(gaussianPdf(0, 0, -2)).toBe(0);
    expect(gaussianPdf(1, Number.NaN, 1)).toBe(0);
    expect(gaussianPdf(1, 0, Number.NaN)).toBe(0);
  });
});

describe('sigmaOf', () => {
  it('is the square root of the variance', () => {
    expect(sigmaOf(4)).toBe(2);
    expect(sigmaOf(1)).toBe(1);
    expect(sigmaOf(0)).toBe(0);
  });
});

describe('fitXRange', () => {
  it('centers a single distribution on mean ± 3.5σ', () => {
    expect(fitXRange([{ label: 'a', mean: 0, variance: 1 }])).toEqual([-3.5, 3.5]);
  });

  it('spans every distribution’s 3.5σ extent', () => {
    const r = fitXRange([
      { label: 'a', mean: 2, variance: 4 },   // ±7 → [-5, 9]
      { label: 'b', mean: -1, variance: 1 },  // ±3.5 → [-4.5, 2.5]
    ]);
    expect(r).toEqual([-5, 9]);
  });

  it('falls back to a default window when no distribution is valid', () => {
    expect(fitXRange([])).toEqual([-1, 1]);
    expect(fitXRange([{ label: 'bad', mean: 0, variance: 0 }])).toEqual([-1, 1]);
  });
});

describe('fitYRange', () => {
  it('fits [0, 1.1 × peak] for a single distribution', () => {
    const y = fitYRange([{ label: 'a', mean: 0, variance: 1 }]);
    expect(y[0]).toBe(0);
    expect(y[1]).toBeCloseTo(1.1 * STD_PEAK, 6);
  });

  it('uses the highest peak across distributions', () => {
    const y = fitYRange([
      { label: 'narrow', mean: 0, variance: 1 },    // peak STD_PEAK
      { label: 'wide', mean: 3, variance: 4 },      // peak STD_PEAK / 2
    ]);
    expect(y[1]).toBeCloseTo(1.1 * STD_PEAK, 6);
  });

  it('ignores peaks outside the given xRange (sampled max instead)', () => {
    const y = fitYRange([{ label: 'a', mean: 3, variance: 1 }], [0, 1]);
    // grid includes x=1 exactly; pdf(1 | N(3,1)) = e⁻² / sqrt(2π) ≈ 0.053991
    expect(y[1]).toBeCloseTo(1.1 * 0.05399096651318806, 4);
  });

  it('falls back to a default height for an empty set', () => {
    expect(fitYRange([])).toEqual([0, 1]);
  });
});

describe('sigmaBandClipped', () => {
  it('is mean ± σ inside the visible range', () => {
    expect(sigmaBandClipped({ label: 'a', mean: 0, variance: 1 }, [-5, 5])).toEqual([-1, 1]);
    expect(sigmaBandClipped({ label: 'a', mean: 0, variance: 4 }, [-5, 5])).toEqual([-2, 2]);
  });

  it('clips to the visible range edges', () => {
    // band [1, 5] on range [0, 4] → visible [1, 4]
    expect(sigmaBandClipped({ label: 'a', mean: 3, variance: 4 }, [0, 4])).toEqual([1, 4]);
  });

  it('returns null when the band lies entirely outside the range', () => {
    expect(sigmaBandClipped({ label: 'a', mean: 10, variance: 1 }, [-5, 5])).toBeNull();
  });
});

describe('collectDistributions', () => {
  it('extracts distribution commands and skips other visuals', () => {
    const cmds: VisualCommand[] = [
      { type: 'distribution', label: 'class 0', mean: -1, variance: 1.5, color: '#2563eb' },
      { type: 'point', x: 1, y: 1 },
      { type: 'distribution', label: 'class 1', mean: 2, variance: 0.5 },
    ];
    expect(collectDistributions(cmds)).toEqual([
      { label: 'class 0', mean: -1, variance: 1.5, color: '#2563eb' },
      { label: 'class 1', mean: 2, variance: 0.5 },
    ]);
  });

  it('filters malformed distributions (missing label, non-finite, non-positive variance)', () => {
    const cmds: VisualCommand[] = [
      { type: 'distribution', label: 'ok', mean: 0, variance: 1 },
      { type: 'distribution', label: '', mean: 0, variance: 1 },
      { type: 'distribution', label: 'zero-var', mean: 0, variance: 0 },
      { type: 'distribution', label: 'neg-var', mean: 0, variance: -2 },
      { type: 'distribution', label: 'nan-mean', mean: Number.NaN, variance: 1 },
      { type: 'distribution', label: 'nan-var', mean: 0, variance: Number.NaN },
    ];
    expect(collectDistributions(cmds)).toEqual([
      { label: 'ok', mean: 0, variance: 1 },
    ]);
  });

  it('returns an empty list for an empty command set', () => {
    expect(collectDistributions([])).toEqual([]);
  });
});

describe('sanitizeDistributions', () => {
  it('keeps valid distributions and drops corrupted ones', () => {
    expect(sanitizeDistributions([
      { label: 'ok', mean: 0, variance: 1, color: '#2563eb' },
      { label: '', mean: 0, variance: 1 },
      { label: 'neg', mean: 0, variance: -1 },
      { label: 'bad', mean: Number.NaN, variance: 2 },
    ])).toEqual([
      { label: 'ok', mean: 0, variance: 1, color: '#2563eb' },
    ]);
  });
});