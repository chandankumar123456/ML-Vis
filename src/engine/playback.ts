import type { SnapshotRun } from './types';

export interface Playback {
  readonly run: SnapshotRun;
  readonly cursor: number;   // integer snapshot index
  readonly playing: boolean;
  readonly speed: number;    // steps per frame tick
  play(): void;
  pause(): void;
  stepForward(): void;
  stepBackward(): void;
  jumpTo(i: number): void;
  reset(): void;
  setSpeed(s: number): void;
  tick(): void;              // called each animation frame; advances by speed while playing
}

export function createPlayback(run: SnapshotRun): Playback {
  let cursorF = 0;               // float accumulator — never exposed raw
  let playing = false;
  let speed = 1;
  const last = () => run.snapshots.length - 1;
  const clamp = (i: number) => Math.max(0, Math.min(last(), i));

  return {
    get run() { return run; },
    get cursor() { return Math.floor(cursorF); },
    get playing() { return playing; },
    get speed() { return speed; },
    play() { playing = true; },
    pause() { playing = false; },
    stepForward() { cursorF = clamp(cursorF + 1); },
    stepBackward() { cursorF = clamp(cursorF - 1); },
    jumpTo(i: number) { cursorF = clamp(i); },
    reset() { cursorF = 0; playing = false; },
    setSpeed(s: number) { speed = Math.max(0.1, Math.min(8, s)); },
    tick() {
      if (!playing) return;
      cursorF = clamp(cursorF + speed);
      if (cursorF >= last()) {
        cursorF = last();
        playing = false;
      }
    },
  };
}
