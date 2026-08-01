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
    const run = computeRun(sim, params);
    const playback = createPlayback(run);
    set({ run, playback, cursor: 0, playing: false });
  },

  setCursor: (i) => {
    const { playback } = get();
    if (!playback) return;
    playback.jumpTo(i);
    set({ cursor: playback.cursor });
  },

  play: () => { get().playback?.play(); set({ playing: true }); },
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
    set({ cursor: 0, playing: false });
  },
  setSpeed: (s) => {
    get().playback?.setSpeed(s);
    set({ speed: s });
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
