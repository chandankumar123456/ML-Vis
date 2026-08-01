import { describe, it, expect } from 'vitest';
import { computeRun, isFiniteState, timelineStages } from './core';
import type { SimulationDef, Params } from './types';

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

describe('computeRun', () => {
  it('produces deterministic snapshots and terminates', () => {
    const runA = computeRun(quadratic, { x0: 5 }, 1000);
    const runB = computeRun(quadratic, { x0: 5 }, 1000);
    expect(runA.snapshots.length).toBe(runB.snapshots.length);
    expect(runA.snapshots.map(s => s.algorithm.x)).toEqual(runB.snapshots.map(s => s.algorithm.x));
    expect(runA.snapshots.length).toBeGreaterThan(2);
    expect(runA.telemetry.snapshotCount).toBe(runA.snapshots.length);
  });

  it('detects diverging runs and reports failure', () => {
    const diverging: SimulationDef = {
      initialState: (p: Params) => ({
        algorithm: { x: p.x0 as number }, visuals: [], math: [], narration: '',
        explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
        highlights: [], metrics: {}, events: [],
        timeline: ['init'],
      }),
      step: (_p, s) => ({ ...s, algorithm: { x: (s.algorithm.x as number) * 2 } }),
    };
    const run = computeRun(diverging, { x0: 1 }, 100);
    expect(run.telemetry.failedAtStep).toBeDefined();
  });
});

describe('isFiniteState', () => {
  it('flags NaN and Infinity', () => {
    const bad: any = { algorithm: { x: NaN }, metrics: { f: Infinity }, visuals: [], math: [], narration: '', explanation: {}, highlights: [], events: [], timeline: [] };
    expect(isFiniteState(bad)).toBe(false);
    const good: any = { algorithm: { x: 1 }, metrics: { f: 2 }, visuals: [], math: [], narration: '', explanation: {}, highlights: [], events: [], timeline: [] };
    expect(isFiniteState(good)).toBe(true);
  });
});

describe('timelineStages', () => {
  it('dedupes repeated stage labels', () => {
    const run = computeRun(quadratic, { x0: 5 }, 1000);
    const stages = timelineStages(run);
    expect(stages[0].label).toBe('init');
    expect(stages.length).toBeGreaterThan(1);
  });
});
