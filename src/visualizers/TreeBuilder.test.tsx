// Component coverage for the tree-builder registry view (Task 19): the empty
// state, node + edge + label rendering, purity-bar fills, highlight rings on a
// highlighted node and its ancestor path, non-finite coordinate skipping and
// last-wins duplicate-id resolution. Canvas draw calls are recorded on a
// stubbed context (jsdom ships no 2D canvas — LossCurve/DecisionBoundary
// convention); the pure scene builder (buildTreeScene) is unit-tested directly.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TreeBuilder, buildTreeScene } from './TreeBuilder';
import type { SimState, VisualCommand } from '../engine/types';

// mirror TreeBuilder.tsx constants the purity-bar test depends on
const BAR_W = 40;
const BAR_H = 4;
const HL_COLOR = '#f59e0b';

type CtxRecorder = {
  setTransform: ReturnType<typeof vi.fn>;
  clearRect: ReturnType<typeof vi.fn>;
  beginPath: ReturnType<typeof vi.fn>;
  moveTo: ReturnType<typeof vi.fn>;
  lineTo: ReturnType<typeof vi.fn>;
  arc: ReturnType<typeof vi.fn>;
  fill: ReturnType<typeof vi.fn>;
  stroke: ReturnType<typeof vi.fn>;
  fillRect: ReturnType<typeof vi.fn>;
  fillText: ReturnType<typeof vi.fn>;
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
    moveTo: vi.fn(), lineTo: vi.fn(), arc: vi.fn(),
    fill: vi.fn(), stroke: vi.fn(), fillRect: vi.fn(), fillText: vi.fn(),
    lineWidth: 1, fillStyle: null, strokeStyle: null, strokeStyles,
  };
  Object.defineProperty(recorder, 'strokeStyle', {
    get: () => strokeStyles[strokeStyles.length - 1] ?? null,
    set: (v: string | null) => { strokeStyles.push(v); },
  });
  return recorder;
}

const mkSnapshot = (visuals: VisualCommand[], highlights: { panel: string; id: string; intensity: number }[] = []): SimState => ({
  algorithm: {},
  visuals,
  math: [],
  narration: '',
  explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
  highlights,
  metrics: {},
  events: [],
  timeline: [],
});

describe('TreeBuilder (component)', () => {
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
    render(<TreeBuilder params={{}} />);
    expect(screen.getByRole('status')).toHaveTextContent('tree-builder: no nodes');
  });

  it('renders an empty state when the snapshot has no node visuals', () => {
    render(<TreeBuilder params={{}} snapshot={mkSnapshot([])} />);
    expect(screen.getByRole('status')).toHaveTextContent('tree-builder: no nodes');
  });

  it('draws node circles with labels on an aria-labelled canvas', () => {
    const snap = mkSnapshot([
      { type: 'node', id: 'root', x: 0.5, y: 0.2, label: 'x1 < 3' },
      { type: 'node', id: 'a', x: 0.2, y: 0.7, label: 'yes' },
      { type: 'node', id: 'b', x: 0.8, y: 0.7, label: 'no' },
    ]);
    const { container } = render(<TreeBuilder params={{}} snapshot={snap} />);
    const canvas = container.querySelector('[data-testid="tree-canvas"]');
    expect(canvas).not.toBeNull();
    expect(canvas!.getAttribute('aria-label')).toMatch(/decision tree/i);
    // one filled circle per node
    expect(ctx.arc).toHaveBeenCalledTimes(3);
    const labels = ctx.fillText.mock.calls.map((c) => c[0]);
    expect(labels).toEqual(expect.arrayContaining(['x1 < 3', 'yes', 'no']));
  });

  it('draws one edge per child of a non-leaf node', () => {
    const snap = mkSnapshot([
      { type: 'node', id: 'root', x: 0.5, y: 0.2, label: 'split', children: ['a', 'b'] },
      { type: 'node', id: 'a', x: 0.2, y: 0.7, label: 'yes' },
      { type: 'node', id: 'b', x: 0.8, y: 0.7, label: 'no' },
    ]);
    render(<TreeBuilder params={{}} snapshot={snap} />);
    // 2 edges, no highlight rings → 2 stroked paths; 3 node circles
    expect(ctx.stroke).toHaveBeenCalledTimes(2);
    expect(ctx.arc).toHaveBeenCalledTimes(3);
    expect(ctx.strokeStyles.every((s) => s === '#94a3b8')).toBe(true);
  });

  it('fills the purity bar proportionally to the node purity (0..1)', () => {
    // single node at normalized (0.5, 0.5) in the 600x400 default container:
    // px = 30 + 0.5*(600-60) = 300, py = 30 + 0.5*(400-60) = 200
    const snap = mkSnapshot([
      { type: 'node', id: 'n', x: 0.5, y: 0.5, label: 'leaf', purity: 0.5 },
    ]);
    render(<TreeBuilder params={{}} snapshot={snap} />);
    // background bar [280, 214, 40, 4], then fill [280, 214, 20, 4]
    expect(ctx.fillRect.mock.calls).toContainEqual([280, 214, BAR_W, BAR_H]);
    expect(ctx.fillRect.mock.calls).toContainEqual([280, 214, BAR_W * 0.5, BAR_H]);
  });

  it('draws a highlight ring on the node AND its ancestor path when highlighted', () => {
    const snap = mkSnapshot(
      [
        { type: 'node', id: 'r', x: 0.5, y: 0.2, label: 'root', children: ['a'] },
        { type: 'node', id: 'a', x: 0.5, y: 0.7, label: 'leaf' },
      ],
      [{ panel: 'canvas', id: 'a', intensity: 1 }],
    );
    render(<TreeBuilder params={{}} snapshot={snap} />);
    // edge r→a + ring on r + ring on a = 3 strokes, all in the highlight color
    expect(ctx.stroke).toHaveBeenCalledTimes(3);
    // 2 node circles + 2 rings
    expect(ctx.arc).toHaveBeenCalledTimes(4);
    expect(ctx.strokeStyles.every((s) => s === HL_COLOR)).toBe(true);
  });

  it('ignores highlights on other panels (only panel "canvas" lights nodes)', () => {
    const snap = mkSnapshot(
      [{ type: 'node', id: 'n', x: 0.5, y: 0.5, label: 'leaf' }],
      [{ panel: 'metrics', id: 'n', intensity: 1 }],
    );
    render(<TreeBuilder params={{}} snapshot={snap} />);
    // no rings → exactly 1 node circle arc, 0 strokes
    expect(ctx.arc).toHaveBeenCalledTimes(1);
    expect(ctx.stroke).not.toHaveBeenCalled();
  });

  it('skips nodes with non-finite coordinates', () => {
    const snap = mkSnapshot([
      { type: 'node', id: 'ok', x: 0.5, y: 0.5, label: 'ok' },
      { type: 'node', id: 'bad-x', x: NaN, y: 0.5, label: 'bad' },
      { type: 'node', id: 'bad-y', x: 0.5, y: Infinity, label: 'bad2' },
    ]);
    render(<TreeBuilder params={{}} snapshot={snap} />);
    expect(ctx.arc).toHaveBeenCalledTimes(1);
    const labels = ctx.fillText.mock.calls.map((c) => c[0]);
    expect(labels).toEqual(['ok']);
  });

  it('resolves duplicate node ids with the last command winning', () => {
    const snap = mkSnapshot([
      { type: 'node', id: 'n', x: 0.2, y: 0.5, label: 'first' },
      { type: 'node', id: 'n', x: 0.8, y: 0.5, label: 'second' },
    ]);
    render(<TreeBuilder params={{}} snapshot={snap} />);
    // only ONE circle is drawn (the surviving id)
    expect(ctx.arc).toHaveBeenCalledTimes(1);
    const labels = ctx.fillText.mock.calls.map((c) => c[0]);
    expect(labels).toEqual(['second']);
  });
});

describe('TreeBuilder (buildTreeScene unit)', () => {
  it('excludes NaN nodes, resolves duplicates last-wins and tolerates dangling children', () => {
    const scene = buildTreeScene(mkSnapshot([
      { type: 'node', id: 'root', x: 0.5, y: 0.2, label: 'root', children: ['ghost', 'a'] },
      { type: 'node', id: 'a', x: 0.2, y: 0.7, label: 'a' },
      { type: 'node', id: 'dup', x: 0.1, y: 0.1, label: 'old' },
      { type: 'node', id: 'dup', x: 0.9, y: 0.9, label: 'new' },
      { type: 'node', id: 'nan', x: NaN, y: 0.5, label: 'nan' },
    ]));
    expect(scene.nodes.map((n) => n.id)).toEqual(['root', 'a', 'dup']);
    expect(scene.nodes.find((n) => n.id === 'dup')!.label).toBe('new');
    // 'ghost' is not a node: the edge is skipped, but the scene still builds
    expect(scene.nodes.find((n) => n.id === 'root')!.children).toEqual(['ghost', 'a']);
  });

  it('collects the full ancestor chain for a canvas highlight', () => {
    const scene = buildTreeScene(mkSnapshot(
      [
        { type: 'node', id: 'r', x: 0.5, y: 0.2, label: 'r', children: ['m'] },
        { type: 'node', id: 'm', x: 0.5, y: 0.5, label: 'm', children: ['a'] },
        { type: 'node', id: 'a', x: 0.5, y: 0.8, label: 'a' },
      ],
      [{ panel: 'canvas', id: 'a', intensity: 0.7 }],
    ));
    expect([...scene.highlighted].sort()).toEqual(['a', 'm', 'r']);
  });

  it('returns an empty scene for a null snapshot', () => {
    const scene = buildTreeScene(null);
    expect(scene.nodes).toEqual([]);
    expect(scene.highlighted.size).toBe(0);
  });
});
