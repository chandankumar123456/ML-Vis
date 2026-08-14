// Component coverage for the eigenviewer registry view (Task 15): empty state
// without a snapshot, the 2-D cloud + candidate axis render, snapshot-axis and
// params-angle fallbacks, slider-driven rotation with recomputed projections,
// variance bars, the 1-D projection strip, snapshot projection-command
// consumption, and the project/reconstruct mode toggle (canvas draw calls are
// recorded on a stubbed context — jsdom ships no 2D canvas, LossCurve/Decision
// Boundary convention).
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Eigenviewer } from './Eigenviewer';
import type { SimState, VisualCommand } from '../engine/types';

type CtxRecorder = {
  setTransform: ReturnType<typeof vi.fn>;
  fillRect: ReturnType<typeof vi.fn>;
  beginPath: ReturnType<typeof vi.fn>;
  moveTo: ReturnType<typeof vi.fn>;
  lineTo: ReturnType<typeof vi.fn>;
  stroke: ReturnType<typeof vi.fn>;
  fill: ReturnType<typeof vi.fn>;
  arc: ReturnType<typeof vi.fn>;
  lineWidth: number;
  fillStyle: string | null;
  strokeStyle: string | null;
  strokeStyles: (string | null)[];
};

let ctx: CtxRecorder;

function createCtxRecorder(): CtxRecorder {
  const strokeStyles: (string | null)[] = [];
  const recorder: CtxRecorder = {
    setTransform: vi.fn(), fillRect: vi.fn(), beginPath: vi.fn(),
    moveTo: vi.fn(), lineTo: vi.fn(), stroke: vi.fn(), fill: vi.fn(),
    arc: vi.fn(), lineWidth: 1, fillStyle: null, strokeStyle: null, strokeStyles,
  };
  // CanvasStage assigns strokeStyle right before stroke() — record each to
  // prove which colors were actually drawn (axis, guides, error lines).
  Object.defineProperty(recorder, 'strokeStyle', {
    get: () => strokeStyles[strokeStyles.length - 1] ?? null,
    set: (v: string | null) => { strokeStyles.push(v); },
  });
  return recorder;
}

const mkSnapshot = (visuals: VisualCommand[]): SimState => ({
  algorithm: {},
  visuals,
  math: [],
  narration: '',
  explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
  highlights: [],
  metrics: {},
  events: [],
  timeline: [],
});

describe('Eigenviewer', () => {
  beforeEach(() => {
    ctx = createCtxRecorder();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext')
      .mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
    class ResizeObserverStub {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    vi.stubGlobal('ResizeObserver', ResizeObserverStub);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('renders an empty state when no snapshot is provided', () => {
    render(<Eigenviewer params={{}} />);
    expect(screen.getByRole('status')).toHaveTextContent('eigenviewer: no data');
  });

  it('renders an empty state when the snapshot has no point visuals', () => {
    render(<Eigenviewer params={{}} snapshot={mkSnapshot([])} />);
    expect(screen.getByRole('status')).toHaveTextContent('eigenviewer: no data');
  });

  it('draws the 2-D cloud, the axis and the projected dots for a filled snapshot', () => {
    const snap = mkSnapshot([
      { type: 'point', x: 0, y: 0, color: '#2563eb' },
      { type: 'point', x: 4, y: 0, color: '#2563eb' },
      { type: 'axis', id: 'axis', angle: 0, color: '#64748b' },
    ]);
    const { container } = render(<Eigenviewer params={{}} snapshot={snap} />);
    const root = container.querySelector('[data-testid="eigenview"]');
    expect(root).not.toBeNull();
    expect(root!.getAttribute('data-angle-deg')).toBe('0');
    // project mode (default): 2 data points + 2 projected dots = 4 arcs
    expect(ctx.arc).toHaveBeenCalledTimes(4);
    // axis path + 2 guide lines + 2 white dot outlines = 5 stroked paths
    expect(ctx.stroke).toHaveBeenCalledTimes(5);
  });

  it('uses the snapshot axis command angle when the user has not touched the slider', () => {
    const snap = mkSnapshot([
      { type: 'point', x: 0, y: 0 },
      { type: 'point', x: 4, y: 0 },
      { type: 'axis', id: 'axis', angle: Math.PI / 4, color: '#64748b' },
    ]);
    const { container } = render(<Eigenviewer params={{}} snapshot={snap} />);
    expect(container.querySelector('[data-testid="eigenview"]')!.getAttribute('data-angle-deg')).toBe('45');
  });

  it('falls back to the params angleDeg hint when the snapshot has no axis command', () => {
    const snap = mkSnapshot([
      { type: 'point', x: 0, y: 0 },
      { type: 'point', x: 4, y: 0 },
    ]);
    const { container } = render(<Eigenviewer params={{ angleDeg: 45 }} snapshot={snap} />);
    expect(container.querySelector('[data-testid="eigenview"]')!.getAttribute('data-angle-deg')).toBe('45');
  });

  it('shows variance-explained bars and a 1-D strip for the axis', () => {
    const snap = mkSnapshot([
      { type: 'point', x: 0, y: 0 },
      { type: 'point', x: 4, y: 0 },
    ]);
    const { container } = render(<Eigenviewer params={{}} snapshot={snap} />);
    // angle 0 default: all variance on the horizontal axis → bars 1.0 / 0.0
    const bars = container.querySelectorAll('[data-testid="eigen-bar"]');
    expect(bars.length).toBe(2);
    expect(bars[0].getAttribute('data-angle')).toBe('0');
    expect(bars[0].getAttribute('data-fraction')).toBe('1.0000');
    expect(bars[1].getAttribute('data-angle')).toBe('90');
    expect(bars[1].getAttribute('data-fraction')).toBe('0.0000');
    // strip: projections onto the x-axis through (2,0) → t = −2 and +2
    const dots = container.querySelectorAll('[data-testid="eigen-strip-dot"]');
    expect(dots.length).toBe(2);
    expect(Array.from(dots).map((d) => d.getAttribute('data-t'))).toEqual(['-2.00', '2.00']);
  });

  it('rotates the axis and recomputes projections when the slider is dragged', () => {
    const snap = mkSnapshot([
      { type: 'point', x: 0, y: 0 },
      { type: 'point', x: 4, y: 0 },
    ]);
    const { container } = render(<Eigenviewer params={{}} snapshot={snap} />);
    const slider = container.querySelector('[data-testid="eigen-axis-slider"]')!;
    const root = () => container.querySelector('[data-testid="eigenview"]')!;
    expect(root().getAttribute('data-angle-deg')).toBe('0');
    expect(root().getAttribute('data-using-snapshot')).toBe('true');

    fireEvent.change(slider, { target: { value: '90' } });

    expect(root().getAttribute('data-angle-deg')).toBe('90');
    expect(root().getAttribute('data-using-snapshot')).toBe('false');
    // vertical axis through (2,0): no variance → bars flip to 0.0 / 1.0
    const bars = container.querySelectorAll('[data-testid="eigen-bar"]');
    expect(bars[0].getAttribute('data-angle')).toBe('90');
    expect(bars[0].getAttribute('data-fraction')).toBe('0.0000');
    // the orthogonal axis is 180° ≡ 0° (a line axis: θ ≡ θ + 180°)
    expect(bars[1].getAttribute('data-angle')).toBe('0');
    expect(bars[1].getAttribute('data-fraction')).toBe('1.0000');
    // every point projects to the axis center → all strip dots at t = 0
    const dots = container.querySelectorAll('[data-testid="eigen-strip-dot"]');
    expect(Array.from(dots).map((d) => d.getAttribute('data-t'))).toEqual(['0.00', '0.00']);

    // "follow topic" escapes the local override and returns to the snapshot angle
    fireEvent.click(container.querySelector('[data-testid="eigen-follow-button"]')!);
    expect(root().getAttribute('data-angle-deg')).toBe('0');
    expect(root().getAttribute('data-using-snapshot')).toBe('true');
    expect(container.querySelector('[data-testid="eigen-follow-button"]')).toBeNull();
  });

  it('repaints from a second snapshot and keeps local mode state across the scrub', () => {
    const snap0 = mkSnapshot([
      { type: 'point', x: 0, y: 0 },
      { type: 'point', x: 4, y: 0 },
      { type: 'axis', id: 'axis', angle: 0, color: '#64748b' },
    ]);
    const snap45 = mkSnapshot([
      { type: 'point', x: 0, y: 0 },
      { type: 'point', x: 4, y: 0 },
      { type: 'axis', id: 'axis', angle: Math.PI / 4, color: '#64748b' },
    ]);
    const { container, rerender } = render(<Eigenviewer params={{}} snapshot={snap0} />);
    const root = () => container.querySelector('[data-testid="eigenview"]')!;
    expect(root().getAttribute('data-angle-deg')).toBe('0');

    // scrubbing to a second snapshot repaints the scene from the new axis angle
    rerender(<Eigenviewer params={{}} snapshot={snap45} />);
    expect(root().getAttribute('data-angle-deg')).toBe('45');
    // the variance bars repaint too: 45° splits the variance evenly
    const bars = container.querySelectorAll('[data-testid="eigen-bar"]');
    expect(bars[0].getAttribute('data-angle')).toBe('45');
    expect(bars[0].getAttribute('data-fraction')).toBe('0.5000');

    // local mode is not reset by the scrub: reconstruct persists across snapshots
    fireEvent.click(container.querySelector('[data-testid="eigen-mode-reconstruct"]')!);
    rerender(<Eigenviewer params={{}} snapshot={snap0} />);
    expect(root().getAttribute('data-mode')).toBe('reconstruct');
    expect(root().getAttribute('data-angle-deg')).toBe('0');
  });

  it('consumes snapshot projection commands (no override) instead of recomputing', () => {
    // Points [(0,2),(2,0)] → centroid (1,1); axis angle 0 → u = (1,0).
    // The topic emits projections to odd onto targets; the strip must reflect
    // those (t = (onto − c)·u = 4 and −4), NOT the recomputed ones (t = ±1).
    const snap = mkSnapshot([
      { type: 'point', x: 0, y: 2 },
      { type: 'point', x: 2, y: 0 },
      { type: 'axis', id: 'axis', angle: 0, color: '#64748b' },
      { type: 'projection', point: [0, 2], onto: [5, 5], residual: 1 },
      { type: 'projection', point: [2, 0], onto: [-3, 7], residual: 1 },
    ]);
    const { container } = render(<Eigenviewer params={{}} snapshot={snap} />);
    const dots = container.querySelectorAll('[data-testid="eigen-strip-dot"]');
    expect(Array.from(dots).map((d) => d.getAttribute('data-t'))).toEqual(['4.00', '-4.00']);
  });

  it('reconstruct mode draws the projected points back with residual error lines', () => {
    const snap = mkSnapshot([
      { type: 'point', x: 0, y: 0 },
      { type: 'point', x: 4, y: 0 },
      { type: 'axis', id: 'axis', angle: Math.PI / 4, color: '#64748b' },
    ]);
    const { container } = render(<Eigenviewer params={{}} snapshot={snap} />);
    const root = container.querySelector('[data-testid="eigenview"]')!;
    expect(root.getAttribute('data-mode')).toBe('project');
    // project mode draws no residual error color
    expect(ctx.strokeStyles.filter((s) => s === '#f43f5e').length).toBe(0);

    fireEvent.click(container.querySelector('[data-testid="eigen-mode-reconstruct"]')!);
    expect(root.getAttribute('data-mode')).toBe('reconstruct');
    // axis + 2 residual error lines stroked in the reconstruction color
    expect(ctx.strokeStyles.filter((s) => s === '#f43f5e').length).toBe(2);
    // 4 arcs per draw × 2 draws (project on mount, reconstruct after the
    // click): 2 faded originals + 2 reconstructed dots both times
    expect(ctx.arc).toHaveBeenCalledTimes(8);
  });
});