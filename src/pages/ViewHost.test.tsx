import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { ViewHost } from './ViewHost';
import { registerView, type ViewProps } from '../registry/viewRegistry';
import { __resetPlaybackLoop, usePlaybackStore } from '../store/playbackStore';
import type { TopicModule, SimulationDef, Params } from '../engine/types';
import type { ComponentType } from 'react';

// controllable rAF: capture the callback + count scheduling instead of running it
let rafCb: FrameRequestCallback | null = null;
let scheduleCount = 0;

beforeEach(() => {
  rafCb = null;
  scheduleCount = 0;
  __resetPlaybackLoop(); // module-level loop bookkeeping persists across tests
  vi.useFakeTimers();
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    scheduleCount++;
    rafCb = cb;
    return scheduleCount;
  });
  usePlaybackStore.setState({ run: null, playback: null, cursor: 0, playing: false, speed: 1 });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const quadratic: SimulationDef = {
  initialState: (p: Params) => ({
    algorithm: { x: p.x0 as number },
    visuals: [], math: [], narration: '',
    explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
    highlights: [], metrics: {}, events: [], timeline: ['init'],
  }),
  step: (_p, s) => {
    const x = s.algorithm.x as number;
    if (Math.abs(x) < 1e-6) return null;
    return { ...s, algorithm: { x: x - 0.1 * 2 * x }, timeline: [...s.timeline, 'step'] };
  },
};

const topic = {
  id: 'vh', title: 'VH', version: 1,
  metadata: { gateWeightage: 'High', revisionPriority: 'P0' },
  layers: { foundation: [], core: [], advanced: [] },
  params: [],
  simulation: quadratic,
  formulas: [], derivations: [], questions: [], comparisons: [], failureDemos: [], mistakes: [], testCases: [],
} as unknown as TopicModule;

const SnapView: ComponentType<ViewProps> = ({ snapshot }) => (
  <span data-testid="x">
    {snapshot ? String((snapshot.algorithm as { x?: number }).x ?? '') : 'none'}
  </span>
);
registerView('vh-snap', SnapView);

describe('ViewHost', () => {
  it('debounces computeAndSet by 150ms and renders the snapshot', () => {
    render(<ViewHost topic={topic} component="vh-snap" params={{ x0: 5 }} />);
    expect(screen.getByTestId('x').textContent).toBe('none');
    act(() => { vi.advanceTimersByTime(149); });
    expect(screen.getByTestId('x').textContent).toBe('none');
    act(() => { vi.advanceTimersByTime(1); });
    expect(screen.getByTestId('x').textContent).toBe('5');
  });

  it('shares ONE animation loop across hosts — one tick per frame', () => {
    render(<ViewHost topic={topic} component="vh-snap" params={{ x0: 5 }} />);
    act(() => { vi.advanceTimersByTime(150); });
    expect(scheduleCount).toBe(1); // first host schedules the singleton loop
    // second host must NOT schedule a second loop (per-host loops → speed ×N)
    render(<ViewHost topic={topic} component="vh-snap" params={{ x0: 5 }} />);
    expect(scheduleCount).toBe(1);
    // play + drive 2 frames → exactly 2 advances, not 2×hosts
    usePlaybackStore.getState().play();
    act(() => { rafCb?.(0); });
    act(() => { rafCb?.(0); });
    expect(usePlaybackStore.getState().cursor).toBe(2);
  });

  it('renders an unknown-view fallback', () => {
    render(<ViewHost topic={topic} component="vh-missing" params={{ x0: 5 }} />);
    expect(screen.getByText('Unknown view: vh-missing')).toBeTruthy();
  });
});
