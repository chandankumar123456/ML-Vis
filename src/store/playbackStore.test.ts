import { describe, it, expect, beforeEach } from 'vitest';
import { usePlaybackStore } from './playbackStore';
import type { SimulationDef, Params } from '../engine/types';

const quadratic: SimulationDef = {
  initialState: (p: Params) => ({
    algorithm: { x: p.x0 as number },
    visuals: [], math: [], narration: '',
    explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
    highlights: [], metrics: { f: (p.x0 as number) ** 2 }, events: [],
    timeline: ['init'],
  }),
  step: (_p, s) => {
    const x = s.algorithm.x as number;
    if (Math.abs(x) < 1e-6) return null;
    return {
      ...s,
      algorithm: { x: x - 0.1 * 2 * x },
      metrics: { f: x ** 2 },
      timeline: [...s.timeline, 'step'],
    };
  },
};

// engine failure in initialState (throws) → empty run with 0 snapshots
const broken: SimulationDef = {
  initialState: () => {
    throw new Error('boom');
  },
  step: (_p, s) => ({ ...s }),
};

beforeEach(() => {
  usePlaybackStore.setState({
    run: null, playback: null, cursor: 0, playing: false, speed: 1,
  });
});

describe('playbackStore', () => {
  it('computes a run and mirrors the engine cursor', () => {
    usePlaybackStore.getState().computeAndSet(quadratic, { x0: 5 });
    const s = usePlaybackStore.getState();
    expect(s.run).not.toBeNull();
    expect(s.cursor).toBe(0);
    expect(s.playing).toBe(false);
  });

  it('never reports playing on an empty (failed) run', () => {
    usePlaybackStore.getState().computeAndSet(broken, { x0: 1 });
    usePlaybackStore.getState().play();
    expect(usePlaybackStore.getState().playing).toBe(false);
  });

  it('propagates the empty-run sentinel cursor (-1)', () => {
    usePlaybackStore.getState().computeAndSet(broken, { x0: 1 });
    expect(usePlaybackStore.getState().cursor).toBe(-1);
  });

  it('play + tick advances cursor, auto-stop flips playing at run end', () => {
    const st = usePlaybackStore.getState();
    st.computeAndSet(quadratic, { x0: 5 });
    st.setSpeed(2);
    st.play();
    expect(usePlaybackStore.getState().playing).toBe(true);
    usePlaybackStore.getState().tick();
    usePlaybackStore.getState().tick();
    const mid = usePlaybackStore.getState();
    expect(mid.cursor).toBeGreaterThan(0);
    expect(mid.playing).toBe(true);
    // drain to the end — engine auto-stops, store must mirror it
    const run = usePlaybackStore.getState().run!;
    for (let i = mid.cursor; i < run.snapshots.length; i++) {
      usePlaybackStore.getState().tick();
    }
    expect(usePlaybackStore.getState().cursor).toBe(run.snapshots.length - 1);
    expect(usePlaybackStore.getState().playing).toBe(false);
  });

  it('setSpeed mirrors engine clamping and rejection', () => {
    usePlaybackStore.getState().computeAndSet(quadratic, { x0: 5 });
    const st = usePlaybackStore.getState();
    st.setSpeed(NaN);
    expect(usePlaybackStore.getState().speed).toBe(1);
    st.setSpeed(0);
    expect(usePlaybackStore.getState().speed).toBe(0.1);
    st.setSpeed(99);
    expect(usePlaybackStore.getState().speed).toBe(8);
  });

  it('reset and step actions mirror the engine', () => {
    const st = usePlaybackStore.getState();
    st.computeAndSet(quadratic, { x0: 5 });
    st.stepForward();
    expect(usePlaybackStore.getState().cursor).toBe(1);
    st.stepBackward();
    expect(usePlaybackStore.getState().cursor).toBe(0);
    st.stepForward();
    st.reset();
    const s = usePlaybackStore.getState();
    expect(s.cursor).toBe(0);
    expect(s.playing).toBe(false);
  });
});
