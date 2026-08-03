// Component coverage for the decision-boundary registry view (Task 5): empty
// state when no classifier is registered, offscreen grid + drawImage upscale
// when one is, re-classification on snapshot change (with the snapshot's
// algorithm state merged into params), support-vector / margin-line overlays,
// and the topic-lookup path.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DecisionBoundary } from './DecisionBoundary';
import { registerClassifier } from '../registry/viewRegistry';
import type { Params, SimState, TopicModule } from '../engine/types';

const GRID = 50;

// minimal topic shell — the view only reads topic.id for the classifier lookup
const mkTopic = (id: string): TopicModule => ({ id }) as TopicModule;

const mkSnapshot = (algorithm: Record<string, number | string | boolean> = {}): SimState => ({
  algorithm,
  visuals: [],
  math: [],
  narration: '',
  explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
  highlights: [],
  metrics: {},
  events: [],
  timeline: [],
});

// jsdom ships no 2D canvas context, no ImageData global and no ResizeObserver —
// draw calls are recorded on a stubbed context (LossCurve convention) and the
// ImageData global is stubbed so the grid buffer can be constructed.
type CtxRecorder = {
  setTransform: ReturnType<typeof vi.fn>;
  clearRect: ReturnType<typeof vi.fn>;
  putImageData: ReturnType<typeof vi.fn>;
  drawImage: ReturnType<typeof vi.fn>;
  beginPath: ReturnType<typeof vi.fn>;
  moveTo: ReturnType<typeof vi.fn>;
  lineTo: ReturnType<typeof vi.fn>;
  stroke: ReturnType<typeof vi.fn>;
  fill: ReturnType<typeof vi.fn>;
  arc: ReturnType<typeof vi.fn>;
  setLineDash: ReturnType<typeof vi.fn>;
  lineWidth: number;
  fillStyle: string | null;
  strokeStyle: string | null;
};

let ctx: CtxRecorder;

function createCtxRecorder(): CtxRecorder {
  return {
    setTransform: vi.fn(), clearRect: vi.fn(), putImageData: vi.fn(),
    drawImage: vi.fn(), beginPath: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(),
    stroke: vi.fn(), fill: vi.fn(), arc: vi.fn(), setLineDash: vi.fn(),
    lineWidth: 1, fillStyle: null, strokeStyle: null,
  };
}

class ImageDataStub {
  width: number;
  height: number;
  data: Uint8ClampedArray;
  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.data = new Uint8ClampedArray(width * height * 4);
  }
}

describe('DecisionBoundary', () => {
  beforeEach(() => {
    ctx = createCtxRecorder();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext')
      .mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
    vi.stubGlobal('ImageData', ImageDataStub);
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

  it('renders the empty state when no classifier is registered for the topic', () => {
    render(<DecisionBoundary topic={mkTopic('db-empty-1')} params={{}} />);
    expect(screen.getByText('decision-boundary: no classifier')).toBeInTheDocument();
    expect(screen.getByRole('status').className).toContain('decision-empty');
  });

  it('renders the empty state when the topic id is unknown', () => {
    render(<DecisionBoundary topic={mkTopic('db-empty-2')} params={{}} snapshot={mkSnapshot()} />);
    expect(screen.getByText('decision-boundary: no classifier')).toBeInTheDocument();
  });

  it('renders the empty state when no topic is provided', () => {
    render(<DecisionBoundary params={{}} />);
    expect(screen.getByText('decision-boundary: no classifier')).toBeInTheDocument();
  });

  it('classifies the whole grid, paints ImageData and upscales to the canvas', () => {
    const calls: number[] = [];
    registerClassifier('db-grid', (x, _y, _p) => { calls.push(x); return x > 0 ? 1 : 0; });
    const { container } = render(
      <DecisionBoundary topic={mkTopic('db-grid')} params={{}} snapshot={mkSnapshot()} />
    );
    expect(ctx.putImageData).toHaveBeenCalledTimes(1);
    expect(ctx.drawImage).toHaveBeenCalledTimes(1);
    // ≥ GRID²: the grid is GRID² samples; the decision-line fitting probes a
    // handful more (midline scan + gradient), so assert the lower bound
    expect(calls.length).toBeGreaterThanOrEqual(GRID * GRID);
    expect(container.querySelector('canvas[data-decision-grid="50"]')).not.toBeNull();
  });

  it('re-classifies with the merged algorithm state when the snapshot changes', () => {
    const seen: Params[] = [];
    registerClassifier('db-snap', (_x, _y, p) => { seen.push(p); return 0; });
    const { rerender } = render(
      <DecisionBoundary topic={mkTopic('db-snap')} params={{ lr: 0.1 }} snapshot={mkSnapshot({ w: 1 })} />
    );
    const first = seen.length;
    expect(first).toBeGreaterThanOrEqual(GRID * GRID);
    rerender(
      <DecisionBoundary topic={mkTopic('db-snap')} params={{ lr: 0.1 }} snapshot={mkSnapshot({ w: 5 })} />
    );
    // snapshot identity change → a full re-classification of the grid
    expect(seen.length).toBeGreaterThanOrEqual(first + GRID * GRID);
    // topic params preserved, current algorithm state merged on top
    expect(seen[seen.length - 1].lr).toBe(0.1);
    expect(seen[seen.length - 1].w).toBe(5);
  });

  it('renders support vectors as highlighted points', () => {
    registerClassifier('db-sv', (_x, _y, _p) => 1);
    render(
      <DecisionBoundary topic={mkTopic('db-sv')} params={{}} snapshot={mkSnapshot()}
        supportVectors={[[1, 2], [-3, -4]]} />
    );
    // one arc per support vector (grid is ImageData; the decision line uses
    // moveTo/lineTo), so 2 arcs == exactly the two highlighted points
    expect(ctx.arc).toHaveBeenCalledTimes(2);
  });

  it('draws no overlay when supportVectors/marginLines props are absent', () => {
    registerClassifier('db-plain', (x, _y, _p) => (x > 0 ? 1 : 0));
    render(<DecisionBoundary topic={mkTopic('db-plain')} params={{}} snapshot={mkSnapshot()} />);
    expect(ctx.arc).not.toHaveBeenCalled();
    // only the solid decision line is stroked — never with the dashed pattern
    expect(ctx.setLineDash).not.toHaveBeenCalledWith([6, 4]);
  });

  it('renders the decision line and dashed margin lines parallel to the boundary', () => {
    // signed-distance classifier whose boundary (2x − 3y = 0) passes through the
    // default-domain center — exercises the midline scan + gradient normal probe
    registerClassifier('db-margin', (x, y, _p) => 2 * x - 3 * y);
    render(
      <DecisionBoundary topic={mkTopic('db-margin')} params={{}} snapshot={mkSnapshot()}
        marginLines={[{ offset: 1 }]} />
    );
    // decision line + 1 margin line
    expect(ctx.stroke).toHaveBeenCalledTimes(2);
    // the margin line uses the dashed pattern
    expect(ctx.setLineDash).toHaveBeenCalledWith([6, 4]);
  });
});
