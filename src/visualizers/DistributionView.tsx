// src/visualizers/DistributionView.tsx
// Gaussian PDF plot per class (registered as 'distribution-view'). Props per
// the plan: { distributions, xRange?, yRange? } — the registry wrapper feeds
// densities collected from the snapshot's visuals
// ({type:'distribution', label, mean, variance, color} commands); x/y ranges
// are fitted when the topic omits them (mean ± 3.5σ x-cover, 1.1× peak
// y-headroom — see ./distribution). Renders each class's curve with ±1σ
// shading, a mean marker at the peak, and a legend; consumed by the naive-bayes,
// mle and lda topics.
import { useEffect, useMemo, useRef } from 'react';
import { fitBounds } from '../lib/canvas/CanvasStage';
import { useContainerSize } from '../lib/canvas/useContainerSize';
import {
  CURVE_SAMPLES, gaussianPdf, sanitizeDistributions, sigmaBandClipped,
  fitXRange, fitYRange,
} from './distribution';
import type { Distribution } from '../engine/types';

const PAD = 30;
const BAND_SAMPLES = 40;
const BASELINE_COLOR = '#e2e8f0';
// Classes without an explicit color cycle this fixed palette.
const FALLBACK_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#f59e0b', '#0891b2'];

export function DistributionView({ distributions, xRange, yRange }: {
  distributions: Distribution[];
  xRange?: [number, number];
  yRange?: [number, number];
}) {
  const [ref, size] = useContainerSize(600, 320);
  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<{ host: HTMLDivElement; canvas: HTMLCanvasElement } | null>(null);

  const dists = useMemo(() => sanitizeDistributions(distributions), [distributions]);
  const xRangeEff = useMemo<[number, number]>(() => xRange ?? fitXRange(dists), [xRange, dists]);
  const yRangeEff = useMemo<[number, number]>(() => yRange ?? fitYRange(dists, xRangeEff), [yRange, dists, xRangeEff]);

  // Paint lazily like DecisionBoundary: create the canvas on first use, then
  // resize + redraw whenever the densities or the resolved ranges change.
  useEffect(() => {
    const host = hostRef.current;
    if (!host || dists.length === 0) return;
    const dpr = window.devicePixelRatio || 1;
    let canvas = canvasRef.current?.host === host ? canvasRef.current.canvas : null;
    if (!canvas) {
      const created = document.createElement('canvas');
      created.className = 'distribution-canvas';
      created.setAttribute('data-testid', 'distribution-canvas');
      created.setAttribute('aria-label',
        'Gaussian probability density curves per class, with ±1σ shading and mean markers');
      created.style.display = 'block';
      host.replaceChildren(created);
      canvasRef.current = { host, canvas: created };
      canvas = created;
    }
    canvas.width = Math.max(1, size.w * dpr);
    canvas.height = Math.max(1, size.h * dpr);
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${size.h}px`;
    // expose the resolved domain to tests / consumers
    canvas.dataset.x0 = xRangeEff[0].toFixed(2);
    canvas.dataset.x1 = xRangeEff[1].toFixed(2);
    canvas.dataset.y0 = '0.00';
    canvas.dataset.y1 = yRangeEff[1].toFixed(2);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    draw(ctx, dists, xRangeEff, yRangeEff, size, dpr);
  }, [dists, xRangeEff, yRangeEff, size]);

  if (dists.length === 0) {
    return (
      <div className="distribution-view distribution-empty" role="status">
        distribution-view: no distributions
      </div>
    );
  }

  return (
    <div ref={ref} className="distribution-view" data-distribution-count={dists.length}
      style={{ width: '100%', height: 360, position: 'relative' }}>
      <div ref={hostRef} style={{ width: '100%', height: 320 }} />
      <div className="distribution-legend distribution-legend-overlay"
        style={{ position: 'absolute', top: 4, right: 8, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {dists.map((d, i) => (
          <span key={d.label} className="distribution-legend-item"
            data-testid="distribution-legend-item" data-label={d.label}>
            <span className="distribution-swatch"
              style={{ background: d.color ?? FALLBACK_COLORS[i % FALLBACK_COLORS.length] }} />
            {d.label}
            <small> μ={d.mean} σ²={d.variance}</small>
          </span>
        ))}
      </div>
    </div>
  );
}

function draw(ctx: CanvasRenderingContext2D, dists: Distribution[],
  xr: [number, number], yr: [number, number],
  size: { w: number; h: number }, dpr: number): void {
  const t = fitBounds({ x: xr, y: yr }, size.w, size.h, PAD);
  const w2s = (x: number, y: number): [number, number] => [x * t.scale + t.tx, y * t.scale + t.ty];
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size.w, size.h);

  // baseline at pdf = 0
  ctx.beginPath();
  const [bx0, by0] = w2s(xr[0], 0);
  const [bx1, by1] = w2s(xr[1], 0);
  ctx.moveTo(bx0, by0);
  ctx.lineTo(bx1, by1);
  ctx.strokeStyle = BASELINE_COLOR;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  dists.forEach((d, di) => {
    const color = d.color ?? FALLBACK_COLORS[di % FALLBACK_COLORS.length];

    // ±1σ shaded band: curve up the band, back along the baseline
    const band = sigmaBandClipped(d, xr);
    if (band) {
      const pts: [number, number][] = [];
      for (let k = 0; k <= BAND_SAMPLES; k++) {
        const x = band[0] + ((band[1] - band[0]) * k) / BAND_SAMPLES;
        pts.push([x, gaussianPdf(x, d.mean, d.variance)]);
      }
      for (let k = BAND_SAMPLES; k >= 0; k--) {
        const x = band[0] + ((band[1] - band[0]) * k) / BAND_SAMPLES;
        pts.push([x, 0]);
      }
      ctx.beginPath();
      const [sx0, sy0] = w2s(pts[0][0], pts[0][1]);
      ctx.moveTo(sx0, sy0);
      for (let k = 1; k < pts.length; k++) {
        const [sx, sy] = w2s(pts[k][0], pts[k][1]);
        ctx.lineTo(sx, sy);
      }
      ctx.closePath();
      ctx.fillStyle = hexAlpha(color);
      ctx.fill();
    }

    // PDF curve sampled over the visible x-range
    ctx.beginPath();
    for (let k = 0; k < CURVE_SAMPLES; k++) {
      const x = xr[0] + ((xr[1] - xr[0]) * k) / (CURVE_SAMPLES - 1);
      const [sx, sy] = w2s(x, gaussianPdf(x, d.mean, d.variance));
      if (k === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    // mean marker: vertical line from the baseline to the peak
    const [mx0, my0] = w2s(d.mean, 0);
    const [mx1, my1] = w2s(d.mean, gaussianPdf(d.mean, d.mean, d.variance));
    ctx.beginPath();
    ctx.moveTo(mx0, my0);
    ctx.lineTo(mx1, my1);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

/** 25%-alpha variant of a #rrggbb fill; passes non-hex colors through. */
function hexAlpha(hex: string): string {
  if (!/^#[0-9a-fA-F]{6}$/.test(hex)) return hex;
  return `${hex}40`;
}