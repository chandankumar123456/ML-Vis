import { create } from 'zustand';
import { createPlayback, type Playback } from '../engine/playback';
import { computeRun } from '../engine/core';
import type { SimulationDef, Params, SnapshotRun } from '../engine/types';

interface PlaybackState {
  run: SnapshotRun | null;
  playback: Playback | null;
  cursor: number;
  playing: boolean;
  speed: number;
  computeAndSet(sim: SimulationDef, params: Params): void;
  setCursor(i: number): void;
  play(): void;
  pause(): void;
  stepForward(): void;
  stepBackward(): void;
  reset(): void;
  setSpeed(s: number): void;
  tick(): void; // frame tick: advances playback and syncs cursor
}

export const usePlaybackStore = create<PlaybackState>((set, get) => ({
  run: null,
  playback: null,
  cursor: 0,
  playing: false,
  speed: 1,

  computeAndSet: (sim, params) => {
    // dedupe: every ViewHost on a page shares the same sim + params object ref;
    // recomputing per host would multiply synchronous simulation work
    const { run } = get();
    if (run && run.params === params) return;
    const run2 = computeRun(sim, params);
    const playback = createPlayback(run2);
    // mirror playback.cursor so empty-run sentinel (-1) propagates immediately
    set({ run: run2, playback, cursor: playback.cursor, playing: false });
  },

  setCursor: (i) => {
    const { playback } = get();
    if (!playback) return;
    playback.jumpTo(i);
    set({ cursor: playback.cursor });
  },

  play: () => {
    const pb = get().playback;
    if (!pb) return;
    pb.play();
    // mirror engine: play() is a no-op on empty runs — never claim playing
    set({ playing: pb.playing });
  },
  pause: () => { get().playback?.pause(); set({ playing: false }); },
  stepForward: () => {
    const { playback } = get();
    if (!playback) return;
    playback.stepForward();
    set({ cursor: playback.cursor });
  },
  stepBackward: () => {
    const { playback } = get();
    if (!playback) return;
    playback.stepBackward();
    set({ cursor: playback.cursor });
  },
  reset: () => {
    const { playback } = get();
    if (!playback) return;
    playback.reset();
    set({ cursor: playback.cursor, playing: false });
  },
  setSpeed: (s) => {
    const pb = get().playback;
    if (!pb) return;
    pb.setSpeed(s);
    // mirror engine: setSpeed rejects non-finite and clamps to [0.1, 8]
    set({ speed: pb.speed });
  },
  tick: () => {
    const { playback, playing } = get();
    if (!playback || !playing) return;
    playback.tick();
    // sync on cursor AND playing change: auto-stop at run end flips playing with cursor unchanged
    if (playback.cursor !== get().cursor || playback.playing !== get().playing) {
      set({ cursor: playback.cursor, playing: playback.playing });
    }
  },
}));

// Single shared animation loop — exactly ONE tick per animation frame no matter
// how many ViewHosts mount. Per-host loops would advance playback speed ×N
// (each tick() advances by `speed`), breaking the "steps per frame tick" contract.
let rafId: number | null = null;
function loop(): void {
  usePlaybackStore.getState().tick();
  rafId = requestAnimationFrame(loop);
}
export function ensurePlaybackLoop(): void {
  if (rafId === null) rafId = requestAnimationFrame(loop);
}

/** vitest hook: clears singleton-loop bookkeeping between tests */
export function __resetPlaybackLoop(): void {
  rafId = null;
}
