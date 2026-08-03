// Component-level coverage for Task 12 visualizers (Task 18 coverage note):
// MatrixAnimator subscribe lifecycle + highlight round-trip, TimelineView
// stage-finding (cursor = -1 sentinel, boundary, beyond-last), FormulaExplorer
// derivesFrom navigation, MistakeView toggle.
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { MatrixAnimator } from './MatrixAnimator';
import { TimelineView } from './TimelineView';
import { FormulaExplorer } from './FormulaExplorer';
import { MistakeView } from './MistakeView';
import { DerivationPlayer } from './DerivationPlayer';
import { LossCurve } from './LossCurve';
import { eventBus } from '../bus/eventBus';
import { usePlaybackStore } from '../store/playbackStore';
import { gdFormulas } from '../topics/gradient-descent/formulas';
import { gdMistakes } from '../topics/gradient-descent/mistakes';
import type { VisualCommand, SimState, SnapshotRun, Derivation } from '../engine/types';

describe('MatrixAnimator', () => {
  const matrix: VisualCommand = {
    type: 'matrix', id: 'W', rows: 2, cols: 2, cells: [[1, 2], [3, 4]],
  };

  it('subscribes to the bus on mount and unsubscribes on unmount', () => {
    const unsub = vi.fn();
    const subscribeSpy = vi.spyOn(eventBus, 'subscribe').mockReturnValue(unsub);
    const { unmount } = render(<MatrixAnimator commands={[matrix]} />);
    expect(subscribeSpy).toHaveBeenCalledTimes(1);
    unmount();
    expect(unsub).toHaveBeenCalled();
    subscribeSpy.mockRestore();
  });

  it('highlight emit → consume round-trip via eventBus', () => {
    render(<MatrixAnimator commands={[matrix]} />);
    const cell = screen.getByTestId('W:0,0');
    expect(cell).not.toHaveClass('active');
    act(() => { eventBus.emit({ type: 'highlight', payload: { panel: 'matrix', id: 'W:0,0', intensity: 1 } }); });
    expect(cell).toHaveClass('active');
    act(() => { eventBus.emit({ type: 'clear-highlights' }); });
    expect(cell).not.toHaveClass('active');
  });
});

describe('TimelineView', () => {
  const mkState = (timeline: string[]): SimState => ({
    algorithm: {},
    visuals: [],
    math: [],
    narration: '',
    explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
    highlights: [],
    metrics: {},
    events: [],
    timeline,
  });

  // stages via timelineStages: {init,0} {phase-a,1} {phase-b,2}
  const run: SnapshotRun = {
    params: {},
    snapshots: [
      mkState(['init']),
      mkState(['init', 'phase-a']),
      mkState(['init', 'phase-a', 'phase-b']),
    ],
    telemetry: { snapshotCount: 3, genMs: 0, memBytes: 0, failedAtStep: undefined, failureReason: undefined },
  };

  const activeStages = (container: HTMLElement) => container.querySelectorAll('.tl-stage.active');

  beforeEach(() => {
    usePlaybackStore.setState({ run, playback: null, cursor: -1 });
  });

  it('cursor = -1 sentinel marks NO stage active', () => {
    usePlaybackStore.setState({ cursor: -1 });
    const { container } = render(<TimelineView />);
    expect(activeStages(container).length).toBe(0);
  });

  it('cursor exactly on a stage boundary belongs to the NEXT stage', () => {
    // boundary between {init,0} and {phase-a,1}: cursor 1 → phase-a active
    usePlaybackStore.setState({ cursor: 1 });
    const { container } = render(<TimelineView />);
    expect(activeStages(container).length).toBe(1);
    expect(activeStages(container)[0].textContent).toContain('phase-a');
  });

  it('cursor beyond the last stage keeps the last stage active', () => {
    usePlaybackStore.setState({ cursor: 10 });
    const { container } = render(<TimelineView />);
    expect(activeStages(container).length).toBe(1);
    expect(activeStages(container)[0].textContent).toContain('phase-b');
  });
});

describe('FormulaExplorer', () => {
  it('navigates to a derived-from formula and back', () => {
    render(<FormulaExplorer formulas={gdFormulas} />);
    // initial selection is the first formula
    expect(screen.getByRole('button', { name: 'f' })).toHaveClass('active');

    // select 'grad' (derivesFrom ['f']) → its pill activates
    fireEvent.click(screen.getByRole('button', { name: 'grad' }));
    expect(screen.getByRole('button', { name: 'grad' })).toHaveClass('active');
    expect(screen.getByRole('button', { name: 'f' })).not.toHaveClass('active');

    // the detail shows a "← f" link; clicking it navigates back to 'f'
    fireEvent.click(screen.getByRole('button', { name: '← f' }));
    expect(screen.getByRole('button', { name: 'f' })).toHaveClass('active');
    expect(screen.getByRole('button', { name: 'grad' })).not.toHaveClass('active');
  });
});

describe('MistakeView', () => {
  it('toggles a mistake body open and closed', () => {
    render(<MistakeView mistakes={gdMistakes} />);
    expect(screen.queryByText(/The plus sign climbs/i)).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: /wrong sign/i }));
    expect(screen.getByText(/The plus sign climbs/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /wrong sign/i }));
    expect(screen.queryByText(/The plus sign climbs/i)).toBeNull();
  });
});

describe('DerivationPlayer', () => {
  const derivations: Derivation[] = [
    {
      id: 'update-rule',
      title: 'Update Rule',
      steps: [
        { latex: 'x_{1} = x_0 - \\eta \\nabla f', justification: 'Move opposite the gradient.' },
        { latex: '\\nabla f = 2x', justification: 'Derivative of x squared.' },
      ],
    },
    {
      id: 'convergence',
      title: 'Convergence',
      steps: [{ latex: '\\lim_k x_k = x^*', justification: 'Reaches a fixed point.' }],
    },
  ];

  it('renders the derivation title and first step, hiding later steps', () => {
    render(<DerivationPlayer derivations={derivations} />);
    expect(screen.getByText('Update Rule')).toBeInTheDocument();
    expect(screen.getByText('Move opposite the gradient.')).toBeInTheDocument();
    expect(screen.queryByText('Derivative of x squared.')).toBeNull();
  });

  it('reveals the next step on click', () => {
    render(<DerivationPlayer derivations={derivations} />);
    fireEvent.click(screen.getByRole('button', { name: /Reveal next step/i }));
    expect(screen.getByText('Derivative of x squared.')).toBeInTheDocument();
  });

  it('navigates to the next derivation and resets the reveal', () => {
    render(<DerivationPlayer derivations={derivations} />);
    fireEvent.click(screen.getByRole('button', { name: /Next derivation/i }));
    expect(screen.getByText('Convergence')).toBeInTheDocument();
    // first step of the new derivation shown, second not revealed
    expect(screen.getByText('Reaches a fixed point.')).toBeInTheDocument();
  });
});

describe('LossCurve', () => {
  // jsdom ships no 2D canvas context and no ResizeObserver — draw calls are
  // recorded on a stubbed context instead of asserting canvas pixels.
  type CtxRecorder = {
    setTransform: ReturnType<typeof vi.fn>;
    fillRect: ReturnType<typeof vi.fn>;
    beginPath: ReturnType<typeof vi.fn>;
    moveTo: ReturnType<typeof vi.fn>;
    lineTo: ReturnType<typeof vi.fn>;
    arc: ReturnType<typeof vi.fn>;
    stroke: ReturnType<typeof vi.fn>;
    fill: ReturnType<typeof vi.fn>;
    lineWidth: number;
    fillStyle: string | null;
    strokeStyle: string | null;
    strokeStyles: (string | null)[];
  };

  // mirror LOSS_SERIES_COLORS in LossCurve.tsx
  const SERIES1 = '#3b82f6';
  const SERIES2 = '#22c55e';
  let ctx: CtxRecorder;

  function createCtxRecorder(): CtxRecorder {
    const strokeStyles: (string | null)[] = [];
    const recorder: CtxRecorder = {
      setTransform: vi.fn(), fillRect: vi.fn(), beginPath: vi.fn(),
      moveTo: vi.fn(), lineTo: vi.fn(), arc: vi.fn(),
      stroke: vi.fn(), fill: vi.fn(), lineWidth: 1,
      fillStyle: null, strokeStyle: null, strokeStyles,
    };
    // drawPath() assigns strokeStyle right before stroke() — record each to
    // prove which series colors were actually stroked.
    Object.defineProperty(recorder, 'strokeStyle', {
      get: () => strokeStyles[strokeStyles.length - 1] ?? null,
      set: (v: string | null) => { strokeStyles.push(v); },
    });
    return recorder;
  }

  function mkRun(rows: Array<Record<string, number>>): SnapshotRun {
    return {
      params: {},
      snapshots: rows.map((metrics) => ({
        algorithm: {}, visuals: [], math: [], narration: '',
        explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
        highlights: [], metrics, events: [], timeline: [],
      })),
      telemetry: { snapshotCount: rows.length, genMs: 0, memBytes: 0, failedAtStep: undefined, failureReason: undefined },
    };
  }

  beforeEach(() => {
    // cursor 999 = beyond any test run → no cursor dot drawn, keeping stroke
    // counts exact. (cursor -1 would crash: run.snapshots[-1] is undefined.)
    usePlaybackStore.setState({ run: null, playback: null, cursor: 999 });
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

  it('draws both series as lines when metricKey2 is set on a multi-snapshot run', () => {
    const run = mkRun([
      { f: 9, g: 10 },
      { f: 4, g: 6 },
      { f: 1, g: 2 },
    ]);
    const { container } = render(<LossCurve run={run} metricKey="f" metricKey2="g" />);
    expect(container.querySelector('[data-loss-mode="line"]')).not.toBeNull();
    // legend names both series
    expect(screen.getByText('f')).toBeInTheDocument();
    expect(screen.getByText('g')).toBeInTheDocument();
    // both paths stroked, in series order, with the two series colors
    expect(ctx.stroke).toHaveBeenCalledTimes(2);
    expect(ctx.strokeStyles).toEqual([SERIES1, SERIES2]);
  });

  it('renders grouped bars for a two-snapshot run with both metric keys', () => {
    const run = mkRun([
      { train: 5, test: 7 },
      { train: 3, test: 4 },
    ]);
    const { container } = render(<LossCurve run={run} metricKey="train" metricKey2="test" />);
    expect(container.querySelector('[data-loss-mode="bars"]')).not.toBeNull();
    // bar mode draws no canvas lines
    expect(ctx.stroke).not.toHaveBeenCalled();
    expect(screen.getByText('step 0')).toBeInTheDocument();
    expect(screen.getByText('step 1')).toBeInTheDocument();
    // one pair of bars per snapshot → 4 bars, keyed by series
    const bars = container.querySelectorAll('[data-key]');
    expect(bars.length).toBe(4);
    expect(container.querySelector('[data-key="train"]')).not.toBeNull();
    expect(container.querySelector('[data-key="test"]')).not.toBeNull();
    // value labels above the bars
    expect(screen.getByText('5.00')).toBeInTheDocument();
    expect(screen.getByText('4.00')).toBeInTheDocument();
  });

  it('renders a bar pair for a single-snapshot run (single-shot topics)', () => {
    const run = mkRun([{ trainMse: 2.5, testMse: 8.25 }]);
    const { container } = render(<LossCurve run={run} metricKey="trainMse" metricKey2="testMse" />);
    expect(container.querySelector('[data-loss-mode="bars"]')).not.toBeNull();
    expect(screen.getByText('step 0')).toBeInTheDocument();
    const bars = container.querySelectorAll('[data-key]');
    expect(bars.length).toBe(2);
    expect(container.querySelector('[data-key="trainMse"]')?.getAttribute('data-value')).toBe('2.5');
    expect(container.querySelector('[data-key="testMse"]')?.getAttribute('data-value')).toBe('8.25');
  });

  it('keeps the original single-line rendering when metricKey2 is absent', () => {
    const run = mkRun([{ f: 9 }, { f: 4 }, { f: 1 }]);
    const { container } = render(<LossCurve run={run} metricKey="f" />);
    expect(container.querySelector('[data-loss-mode="line"]')).not.toBeNull();
    // no legend for a single series
    expect(screen.queryByText('f')).toBeNull();
    expect(ctx.stroke).toHaveBeenCalledTimes(1);
  });

  it('early-returns without drawing for a single snapshot and no metricKey2', () => {
    const run = mkRun([{ f: 9 }]);
    const { container } = render(<LossCurve run={run} metricKey="f" />);
    expect(container.querySelector('[data-loss-mode="line"]')).not.toBeNull();
    // original behavior: <2 finite points → return before clear()/drawPath()
    expect(ctx.fillRect).not.toHaveBeenCalled();
    expect(ctx.stroke).not.toHaveBeenCalled();
  });
});
