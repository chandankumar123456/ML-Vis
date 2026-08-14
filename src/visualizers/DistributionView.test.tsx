// Component coverage for the distribution-view registry view (Task 15): the
// empty state, PDF curve + ±1σ shading + mean markers + legend rendering for
// class densities, fitted vs explicit x/y ranges, and malformed-distribution
// filtering. Canvas draw calls are recorded on a stubbed context (jsdom ships
// no 2D canvas — LossCurve/DecisionBoundary convention).
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DistributionView } from './DistributionView';
import type { Distribution } from '../engine/types';

// mirrors CURVE_SAMPLES in DistributionView.tsx — the curve alone traces 199
// lineTo per distribution
const CURVE_SAMPLES = 200;
const PEAK_N01 = 0.3989422804014327; // 1 / sqrt(2π)

type CtxRecorder = {
  setTransform: ReturnType<typeof vi.fn>;
  clearRect: ReturnType<typeof vi.fn>;
  beginPath: ReturnType<typeof vi.fn>;
  moveTo: ReturnType<typeof vi.fn>;
  lineTo: ReturnType<typeof vi.fn>;
  stroke: ReturnType<typeof vi.fn>;
  fill: ReturnType<typeof vi.fn>;
  closePath: ReturnType<typeof vi.fn>;
  lineWidth: number;
  fillStyle: string | null;
  strokeStyle: string | null;
  strokeStyles: (string | null)[];
};

let ctx: CtxRecorder;

function createCtxRecorder(): CtxRecorder {
  const strokeStyles: (string | null)[] = [];
  const recorder: CtxRecorder = {
    setTransform: vi.fn(), clearRect: vi.fn(), beginPath: vi.fn(),
    moveTo: vi.fn(), lineTo: vi.fn(), stroke: vi.fn(), fill: vi.fn(),
    closePath: vi.fn(), lineWidth: 1, fillStyle: null, strokeStyle: null, strokeStyles,
  };
  Object.defineProperty(recorder, 'strokeStyle', {
    get: () => strokeStyles[strokeStyles.length - 1] ?? null,
    set: (v: string | null) => { strokeStyles.push(v); },
  });
  return recorder;
}

const DISTS: Distribution[] = [
  { label: 'class A', mean: 0, variance: 1, color: '#2563eb' },
  { label: 'class B', mean: 3, variance: 4, color: '#dc2626' },
];

describe('DistributionView', () => {
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

  it('renders an empty state for an empty distribution list', () => {
    render(<DistributionView distributions={[]} />);
    expect(screen.getByRole('status')).toHaveTextContent('distribution-view: no distributions');
  });

  it('draws PDF curves with ±1σ shading, mean markers and a legend per class', () => {
    const { container } = render(<DistributionView distributions={DISTS} />);
    const canvas = container.querySelector('[data-testid="distribution-canvas"]');
    expect(canvas).not.toBeNull();
    // fitted xRange spans every 3.5σ extent: A → [-3.5, 3.5], B (σ=2) → [-4, 10]
    expect(canvas!.getAttribute('data-x0')).toBe('-4.00');
    expect(canvas!.getAttribute('data-x1')).toBe('10.00');
    // fitted yRange = 1.1 × highest peak (class A) with 2 decimals
    expect(canvas!.getAttribute('data-y0')).toBe('0.00');
    expect(canvas!.getAttribute('data-y1')).toBe((1.1 * PEAK_N01).toFixed(2));
    // baseline + (curve + mean marker) per distribution = 1 + 2·2 strokes
    expect(ctx.stroke).toHaveBeenCalledTimes(5);
    // one ±1σ shaded band fill per distribution
    expect(ctx.fill).toHaveBeenCalledTimes(2);
    // curves are actually traced at curve-sampling resolution
    expect(ctx.lineTo.mock.calls.length).toBeGreaterThanOrEqual(CURVE_SAMPLES * DISTS.length);
    // each distribution stroked in its own color
    expect(ctx.strokeStyles).toContain('#2563eb');
    expect(ctx.strokeStyles).toContain('#dc2626');
    // legend lists every class with its label and the μ / σ² summary
    expect(screen.getByText('class A')).toBeInTheDocument();
    expect(screen.getByText('class B')).toBeInTheDocument();
    expect(container.querySelector('[data-label="class A"]')).not.toBeNull();
    expect(container.querySelector('[data-label="class B"]')).not.toBeNull();
  });

  it('honors explicit xRange and yRange props instead of fitting them', () => {
    const { container } = render(
      <DistributionView distributions={[DISTS[0]]} xRange={[-2, 2]} yRange={[0, 1]} />
    );
    const canvas = container.querySelector('[data-testid="distribution-canvas"]');
    expect(canvas!.getAttribute('data-x0')).toBe('-2.00');
    expect(canvas!.getAttribute('data-x1')).toBe('2.00');
    expect(canvas!.getAttribute('data-y0')).toBe('0.00');
    expect(canvas!.getAttribute('data-y1')).toBe('1.00');
  });

  it('filters malformed distributions and renders only the valid ones', () => {
    const { container } = render(<DistributionView distributions={[
      { label: 'zero-var', mean: 0, variance: 0 },
      { label: 'ok', mean: 0, variance: 1, color: '#2563eb' },
    ]} />);
    expect(screen.queryByText('zero-var')).toBeNull();
    expect(screen.getByText('ok')).toBeInTheDocument();
    expect(container.querySelector('[data-testid="distribution-canvas"]')!.getAttribute('data-x1')).toBe('3.50');
    // only the valid distribution is listed in the legend
    expect(container.querySelectorAll('[data-testid="distribution-legend-item"]').length).toBe(1);
  });

  it('renders both legend items when classes share a label (keys never collide)', () => {
    const { container } = render(<DistributionView distributions={[
      { label: 'class', mean: 0, variance: 1, color: '#2563eb' },
      { label: 'class', mean: 3, variance: 4, color: '#dc2626' },
    ]} />);
    const items = container.querySelectorAll('[data-testid="distribution-legend-item"]');
    expect(items.length).toBe(2);
    // both entries keep their own label and summary
    expect(items[0].getAttribute('data-label')).toBe('class');
    expect(items[1].getAttribute('data-label')).toBe('class');
    expect(screen.getAllByText('class')).toHaveLength(2);
  });

  it('cycles the fallback palette for classes without an explicit color', () => {
    render(<DistributionView distributions={[
      { label: 'a', mean: 0, variance: 1 },
      { label: 'b', mean: 3, variance: 4 },
    ]} />);
    // FALLBACK_COLORS[0] and FALLBACK_COLORS[1] are used for the curves
    expect(ctx.strokeStyles).toContain('#2563eb');
    expect(ctx.strokeStyles).toContain('#dc2626');
  });
});