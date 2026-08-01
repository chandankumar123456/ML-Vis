import { describe, it, expect } from 'vitest';
import { createPlayback } from './playback';
import type { SnapshotRun, SimState } from './types';

const mkRun = (n: number): SnapshotRun => ({
  params: {},
  snapshots: Array.from({ length: n }, (_, i) => ({
    algorithm: { i }, visuals: [], math: [], narration: `s${i}`,
    explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
    highlights: [], metrics: {}, events: [], timeline: [`t${i}`],
  }) as SimState),
  telemetry: { snapshotCount: n, genMs: 0, memBytes: 0 },
});

describe('createPlayback', () => {
  it('steps forward and backward', () => {
    const pb = createPlayback(mkRun(5));
    expect(pb.cursor).toBe(0);
    pb.stepForward();
    expect(pb.cursor).toBe(1);
    pb.stepBackward();
    expect(pb.cursor).toBe(0);
  });
  it('clamps at boundaries', () => {
    const pb = createPlayback(mkRun(3));
    pb.stepBackward();
    expect(pb.cursor).toBe(0);
    pb.jumpTo(99);
    expect(pb.cursor).toBe(2);
  });
  it('play/stop toggles and respects speed', () => {
    const pb = createPlayback(mkRun(10));
    pb.play();
    expect(pb.playing).toBe(true);
    pb.pause();
    expect(pb.playing).toBe(false);
    pb.setSpeed(2);
    expect(pb.speed).toBe(2);
  });
  it('reset returns to step 0', () => {
    const pb = createPlayback(mkRun(10));
    pb.jumpTo(7);
    pb.reset();
    expect(pb.cursor).toBe(0);
  });
  it('tick advances while playing and stops at end', () => {
    const pb = createPlayback(mkRun(3));
    pb.play();
    pb.tick();
    pb.tick();
    expect(pb.cursor).toBe(2);
    pb.tick();
    expect(pb.playing).toBe(false);
    expect(pb.cursor).toBe(2);
  });
  it('tick with fractional speed keeps integer cursor', () => {
    const pb = createPlayback(mkRun(10));
    pb.play();
    pb.setSpeed(0.5);
    pb.tick(); // 0.5 → floor → 0
    expect(pb.cursor).toBe(0);
    pb.tick(); // 1.0 → 1
    expect(pb.cursor).toBe(1);
    pb.tick(); // 1.5 → 1
    expect(pb.cursor).toBe(1);
    pb.tick(); // 2.0 → 2
    expect(pb.cursor).toBe(2);
  });
  it('handles empty runs without breaking the cursor invariant', () => {
    const pb = createPlayback(mkRun(0));
    expect(pb.cursor).toBe(-1); // sentinel: no valid index
    pb.play();
    pb.tick();
    pb.stepForward();
    pb.jumpTo(3);
    expect(pb.cursor).toBe(-1);
    expect(pb.playing).toBe(false);
  });
  it('guards invalid speed input and clamps bounds', () => {
    const pb = createPlayback(mkRun(5));
    pb.setSpeed(NaN);
    expect(pb.speed).toBe(1);
    pb.setSpeed(0.05);
    expect(pb.speed).toBe(0.1);
    pb.setSpeed(99);
    expect(pb.speed).toBe(8);
  });
  it('jumpTo clamps negative indices', () => {
    const pb = createPlayback(mkRun(5));
    pb.jumpTo(-5);
    expect(pb.cursor).toBe(0);
  });
  it('tick advances by speed while playing', () => {
    const pb = createPlayback(mkRun(10));
    pb.play();
    pb.setSpeed(3);
    pb.tick();
    expect(pb.cursor).toBe(3);
    pb.pause();
    pb.tick();
    expect(pb.cursor).toBe(3);
  });
});
