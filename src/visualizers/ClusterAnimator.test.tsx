// Component coverage for the cluster-animator registry view (Task 19): empty
// state, points + centroids + assignment lines (counted via a stubbed canvas
// context), centroid-color propagation onto assigned points, the loss readout
// (metrics-driven, topic.lossMetricKey-preferred, and {type:'text'} command
// fallback), non-finite / dangling-centroid skipping, and the faint centroid
// trail that updates across two consecutive snapshot renders (the "animated
// convergence" visual). Pure helpers (buildClusterScene / resolveLoss /
// clusterBounds) are unit-tested directly.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ClusterAnimator, buildClusterScene, resolveLoss, clusterBounds } from './ClusterAnimator';
import type { SimState, TopicModule, VisualCommand } from '../engine/types';

const TRAIL_COLOR = 'rgba(148,163,184,0.45)';

type CtxRecorder = {
  setTransform: ReturnType<typeof vi.fn>;
  clearRect: ReturnType<typeof vi.fn>;
  beginPath: ReturnType<typeof vi.fn>;
  moveTo: ReturnType<typeof vi.fn>;
  lineTo: ReturnType<typeof vi.fn>;
  arc: ReturnType<typeof vi.fn>;
  fill: ReturnType<typeof vi.fn>;
  stroke: ReturnType<typeof vi.fn>;
  lineWidth: number;
  fillStyle: string | null;
  strokeStyle: string | null;
  fillStyles: (string | null)[];
  strokeStyles: (string | null)[];
};

let ctx: CtxRecorder;

function createCtxRecorder(): CtxRecorder {
  const fillStyles: (string | null)[] = [];
  const strokeStyles: (string | null)[] = [];
  const recorder: CtxRecorder = {
    setTransform: vi.fn(), clearRect: vi.fn(), beginPath: vi.fn(),
    moveTo: vi.fn(), lineTo: vi.fn(), arc: vi.fn(),
    fill: vi.fn(), stroke: vi.fn(), lineWidth: 1,
    fillStyle: null, strokeStyle: null, fillStyles, strokeStyles,
  };
  Object.defineProperty(recorder, 'fillStyle', {
    get: () => fillStyles[fillStyles.length - 1] ?? null,
    set: (v: string | null) => { fillStyles.push(v); },
  });
  Object.defineProperty(recorder, 'strokeStyle', {
    get: () => strokeStyles[strokeStyles.length - 1] ?? null,
    set: (v: string | null) => { strokeStyles.push(v); },
  });
  return recorder;
}

const mkSnapshot = (visuals: VisualCommand[], metrics: Record<string, number> = {}): SimState => ({
  algorithm: {},
  visuals,
  math: [],
  narration: '',
  explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
  highlights: [],
  metrics,
  events: [],
  timeline: [],
});

describe('ClusterAnimator (component)', () => {
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
    render(<ClusterAnimator params={{}} />);
    expect(screen.getByRole('status')).toHaveTextContent('cluster-animator: no data');
  });

  it('renders an empty state when the snapshot has no points or centroids', () => {
    render(<ClusterAnimator params={{}} snapshot={mkSnapshot([
      { type: 'assignment', point: [0, 0], centroidId: 'c1' },
    ])} />);
    expect(screen.getByRole('status')).toHaveTextContent('cluster-animator: no data');
  });

  it('draws points, centroids and assignment lines with the right counts', () => {
    const snap = mkSnapshot([
      { type: 'point', id: 'p1', x: 0, y: 0 },
      { type: 'point', id: 'p2', x: 2, y: 0 },
      { type: 'centroid', id: 'c1', x: 1, y: 1 },
      { type: 'centroid', id: 'c2', x: 3, y: 1 },
      { type: 'assignment', point: [0, 0], centroidId: 'c1' },
      { type: 'assignment', point: [2, 0], centroidId: 'c2' },
    ]);
    const { container } = render(<ClusterAnimator params={{}} snapshot={snap} />);
    expect(container.querySelector('[data-testid="cluster-canvas"]')).not.toBeNull();
    // arcs: 2 assignment points + 2 data points + 2 centroid fills + 2 centroid outlines = 8
    expect(ctx.arc).toHaveBeenCalledTimes(8);
    // strokes: 2 assignment lines + 2 centroid outlines = 4
    expect(ctx.stroke).toHaveBeenCalledTimes(4);
  });

  it('colors each assigned point with its centroid color (propagation)', () => {
    const snap = mkSnapshot([
      { type: 'centroid', id: 'c1', x: 1, y: 1, color: '#16a34a' },
      { type: 'assignment', point: [0, 0], centroidId: 'c1' },
    ]);
    render(<ClusterAnimator params={{}} snapshot={snap} />);
    // the centroid fill and the assigned-point fill both use #16a34a
    expect(ctx.fillStyles.filter((s) => s === '#16a34a').length).toBeGreaterThanOrEqual(2);
    // the assignment line strokes in the same color
    expect(ctx.strokeStyles).toContain('#16a34a');
  });

  it('renders the loss readout from a j-prefixed metric', () => {
    const snap = mkSnapshot(
      [{ type: 'point', id: 'p', x: 0, y: 0 }],
      { j: 123.456 },
    );
    render(<ClusterAnimator params={{}} snapshot={snap} />);
    expect(screen.getByText('J = 123.46')).toBeInTheDocument();
  });

  it('prefers the topic lossMetricKey over other loss/cost/j metrics', () => {
    const topic = { lossMetricKey: 'inertia' } as TopicModule;
    const snap = mkSnapshot(
      [{ type: 'point', id: 'p', x: 0, y: 0 }],
      { inertia: 42, loss: 5 },
    );
    render(<ClusterAnimator params={{}} snapshot={snap} topic={topic} />);
    expect(screen.getByText('Inertia = 42.00')).toBeInTheDocument();
  });

  it('falls back to a {type:"text"} command when no loss/cost/j metric exists', () => {
    const snap = mkSnapshot([
      { type: 'point', id: 'p', x: 0, y: 0 },
      { type: 'text', id: 'loss', text: 'J = 9.87' },
    ], { other: 1 });
    render(<ClusterAnimator params={{}} snapshot={snap} />);
    expect(screen.getByText('J = 9.87')).toBeInTheDocument();
  });

  it('skips NaN points/centroids/assignment points and dangling assignments', () => {
    const snap = mkSnapshot([
      { type: 'point', id: 'ok', x: 0, y: 0 },
      { type: 'point', id: 'nan', x: NaN, y: 0 },
      { type: 'centroid', id: 'bad', x: NaN, y: NaN },
      { type: 'assignment', point: [NaN, 0], centroidId: 'c1' },
      { type: 'assignment', point: [0, 0], centroidId: 'missing' },
    ]);
    render(<ClusterAnimator params={{}} snapshot={snap} />);
    // only the one valid point is drawn
    expect(ctx.arc).toHaveBeenCalledTimes(1);
    expect(ctx.stroke).not.toHaveBeenCalled();
  });

  it('draws a faint trail at the previous snapshot centroid positions', () => {
    const snapA = mkSnapshot([
      { type: 'point', id: 'p', x: 0, y: 0 },
      { type: 'centroid', id: 'c1', x: 1, y: 1 },
    ]);
    const snapB = mkSnapshot([
      { type: 'point', id: 'p', x: 0, y: 0 },
      { type: 'centroid', id: 'c1', x: 3, y: 3 },
    ]);
    const { rerender } = render(<ClusterAnimator params={{}} snapshot={snapA} />);
    // snapshot A: point + centroid fill + outline = 3 arcs, no trail yet
    expect(ctx.arc).toHaveBeenCalledTimes(3);
    rerender(<ClusterAnimator params={{}} snapshot={snapB} />);
    // snapshot B: trail (c1 @ A's position) + point + centroid fill + outline = 4 arcs
    expect(ctx.arc).toHaveBeenCalledTimes(7);
    expect(ctx.fillStyles).toContain(TRAIL_COLOR);
  });
});

describe('ClusterAnimator (pure helpers)', () => {
  it('buildClusterScene filters non-finite coords and keeps assignments', () => {
    const scene = buildClusterScene(mkSnapshot([
      { type: 'point', id: 'p', x: 0, y: 0 },
      { type: 'point', id: 'bad', x: NaN, y: 1 },
      { type: 'centroid', id: 'c1', x: 1, y: 1 },
      { type: 'assignment', point: [0, 0], centroidId: 'c1' },
    ]));
    expect(scene.points.map((p) => p.id)).toEqual(['p']);
    expect(scene.centroids.map((c) => c.id)).toEqual(['c1']);
    expect(scene.assignments).toHaveLength(1);
  });

  it('resolveLoss prefers the topic key, then loss/cost/j metrics, then text', () => {
    expect(resolveLoss(mkSnapshot([], { j: 2 }))!.text).toBe('J = 2.00');
    expect(resolveLoss(mkSnapshot([], { cost: 3.5 }))!.text).toBe('Cost = 3.50');
    expect(resolveLoss(mkSnapshot([], { loss: 0.125 }))!.text).toBe('Loss = 0.13');
    expect(resolveLoss(mkSnapshot([{ type: 'text', id: 't', text: 'hi' }], { other: 1 }))!.text).toBe('hi');
    expect(resolveLoss(mkSnapshot([{ type: 'text', id: 't', text: 'hi' }], {}))!.text).toBe('hi');
    expect(resolveLoss(null)).toBeNull();
  });

  it('clusterBounds pads the data extent and falls back to a default domain', () => {
    const b = clusterBounds([[0, 0], [2, 4]], [[1, 2]]);
    expect(b.x[0]).toBeLessThan(0);
    expect(b.x[1]).toBeGreaterThan(2);
    expect(b.y[1]).toBeGreaterThan(4);
    const empty = clusterBounds([], []);
    expect(empty.x).toEqual([-7, 7]);
    expect(empty.y).toEqual([-7, 7]);
  });
});