// src/visualizers/distribution.ts
// Pure, DOM-free helpers for the distribution-view registry view (Task 15):
// the Gaussian PDF, sigma, x/y range fitting, the ±1σ band clipped to the
// visible range, and collecting densities emitted as visual commands (the
// channel Wave-4 topics use to feed the view through the registry).
import type { Distribution, VisualCommand } from '../engine/types';

/** How many σ to extend the x-range around each mean (covers 99.95%). */
const X_SIGMA = 3.5;
/** Headroom above the highest sampled PDF peak when fitting the y-range. */
const Y_HEADROOM = 1.1;
/** Curve sampling resolution — must stay in sync with DistributionView.tsx. */
export const CURVE_SAMPLES = 200;
/** Fallback window when no valid distribution exists. */
const DEFAULT_X_RANGE: [number, number] = [-1, 1];
const DEFAULT_Y_RANGE: [number, number] = [0, 1];

/**
 * Univariate Gaussian density. Returns 0 for degenerate or corrupted inputs
 * (non-positive / non-finite variance, non-finite mean) so a hostile topic can
 * never poison the plotted domain.
 */
export function gaussianPdf(x: number, mean: number, variance: number): number {
  if (!(variance > 0) || !Number.isFinite(variance) || !Number.isFinite(mean)) return 0;
  const z = (x - mean) / Math.sqrt(variance);
  return Math.exp(-(z * z) / 2) / Math.sqrt(2 * Math.PI * variance);
}

/** Standard deviation of a distribution. */
export function sigmaOf(variance: number): number {
  return Math.sqrt(variance);
}

/**
 * Fits the x-range over every distribution's mean ± 3.5σ extent.
 * Falls back to a default window when nothing valid exists.
 */
export function fitXRange(distributions: Distribution[]): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const d of distributions) {
    if (!(d.variance > 0) || !Number.isFinite(d.mean)) continue;
    const s = sigmaOf(d.variance);
    lo = Math.min(lo, d.mean - X_SIGMA * s);
    hi = Math.max(hi, d.mean + X_SIGMA * s);
  }
  return Number.isFinite(lo) ? [lo, hi] : DEFAULT_X_RANGE;
}

/**
 * Fits the y-range as [0, headroom × max(pdf)] over the visible domain. Peaks
 * whose mean lies inside the range contribute their exact maximum; otherwise
 * the sampled curve within the range bounds the height. Deterministic: uses
 * the same sampling grid as the renderer.
 */
export function fitYRange(distributions: Distribution[], xRange?: [number, number]): [number, number] {
  const rx = xRange ?? fitXRange(distributions);
  let max = 0;
  const visible = (d: Distribution) => d.mean >= rx[0] && d.mean <= rx[1];
  // Exact peak at a visible mean: the CURVE_SAMPLES grid below samples evenly
  // spaced x, and d.mean can fall BETWEEN grid points, so the grid alone would
  // clip the true peak (and the mean marker drawn up to it).
  for (const d of distributions) {
    if (!(d.variance > 0) || !Number.isFinite(d.mean)) continue;
    if (visible(d)) max = Math.max(max, gaussianPdf(d.mean, d.mean, d.variance));
  }
  // sample the rendered grid so the fit never under-covers the drawn curves
  for (const d of distributions) {
    if (!(d.variance > 0)) continue;
    for (let i = 0; i < CURVE_SAMPLES; i++) {
      const x = rx[0] + (i / (CURVE_SAMPLES - 1)) * (rx[1] - rx[0]);
      max = Math.max(max, gaussianPdf(x, d.mean, d.variance));
    }
  }
  return [0, max > 0 ? max * Y_HEADROOM : DEFAULT_Y_RANGE[1]];
}

/**
 * The ±1σ band of a distribution clipped to the visible x-range, or null when
 * the band lies entirely outside it (nothing to shade).
 */
export function sigmaBandClipped(d: Distribution, xRange: [number, number]): [number, number] | null {
  const s = sigmaOf(d.variance);
  const lo = Math.max(d.mean - s, xRange[0]);
  const hi = Math.min(d.mean + s, xRange[1]);
  return lo < hi ? [lo, hi] : null;
}

/**
 * Drops malformed densities (missing label, non-finite mean, non-positive or
 * non-finite variance) so the view never plots or legends a broken class.
 */
export function sanitizeDistributions(distributions: Distribution[]): Distribution[] {
  return distributions.filter((d) =>
    typeof d.label === 'string' && d.label.length > 0 &&
    Number.isFinite(d.mean) && Number.isFinite(d.variance) && d.variance > 0
  );
}

/**
 * Collects `{type:'distribution', label, mean, variance, color}` visual
 * commands into Distribution[] (the registry wrapper's channel — topics emit
 * densities in their snapshots). Malformed entries are dropped.
 */
export function collectDistributions(visuals: VisualCommand[]): Distribution[] {
  const out: Distribution[] = [];
  for (const cmd of visuals) {
    if (cmd.type !== 'distribution') continue;
    const label = cmd.label;
    const mean = cmd.mean;
    const variance = cmd.variance;
    if (typeof label !== 'string' || label.length === 0) continue;
    if (typeof mean !== 'number' || !Number.isFinite(mean)) continue;
    if (typeof variance !== 'number' || !Number.isFinite(variance) || variance <= 0) continue;
    out.push({
      label,
      mean,
      variance,
      color: typeof cmd.color === 'string' ? cmd.color : undefined,
    });
  }
  return out;
}