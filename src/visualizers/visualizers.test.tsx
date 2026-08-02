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
