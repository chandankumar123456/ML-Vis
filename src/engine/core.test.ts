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
    expect(runA.snapshots).toEqual(runB.snapshots); // whole snapshot arrays, all channels
    expect(runA.snapshots.length).toBeGreaterThan(2);
    expect(runA.telemetry.snapshotCount).toBe(runA.snapshots.length);
  });

  it('keeps failedAtStep undefined on clean convergence', () => {
    const run = computeRun(quadratic, { x0: 5 }, 1000);
    expect(run.telemetry.failedAtStep).toBeUndefined();
    expect(run.telemetry.failureReason).toBeUndefined();
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

  it('reports non-finite initial state without running steps', () => {
    const badInit: SimulationDef = {
      initialState: () => ({
        algorithm: { x: NaN }, visuals: [], math: [], narration: '',
        explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
        highlights: [], metrics: {}, events: [], timeline: ['init'],
      }),
      step: () => null,
    };
    const run = computeRun(badInit, {}, 10);
    expect(run.telemetry.failedAtStep).toBe(0);
    expect(run.telemetry.failureReason).toContain('non-finite');
    expect(run.snapshots).toHaveLength(1);
  });

  it('sandboxes thrown step exceptions', () => {
    const throwing: SimulationDef = {
      initialState: () => ({
        algorithm: {}, visuals: [], math: [], narration: '',
        explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
        highlights: [], metrics: {}, events: [], timeline: ['init'],
      }),
      step: () => { throw new Error('boom'); },
    };
    const run = computeRun(throwing, {}, 100);
    expect(run.telemetry.failedAtStep).toBe(1);
    expect(run.telemetry.failureReason).toBe('boom');
  });

  it('prefers non-finite diagnosis over budget at the final step', () => {
    const overflow: SimulationDef = {
      initialState: (p: Params) => ({
        algorithm: { x: p.x0 as number }, visuals: [], math: [], narration: '',
        explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
        highlights: [], metrics: {}, events: [], timeline: ['init'],
      }),
      step: (_p, s) => ({ ...s, algorithm: { x: (s.algorithm.x as number) * 2 } }),
    };
    // x: 1.25e307 → 2.5e307 → 5e307 → 1e308 → Infinity exactly at the last allowed step (i=4, maxSteps=5)
    const run = computeRun(overflow, { x0: 1.25e307 }, 5);
    expect(run.telemetry.failureReason).toContain('non-finite');
    expect(run.telemetry.failedAtStep).toBe(4);
  });

  it('does not let cyclic state escape the sandbox', () => {
    const cyclic: SimulationDef = {
      initialState: () => ({
        algorithm: {}, visuals: [], math: [], narration: '',
        explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
        highlights: [], metrics: {}, events: [], timeline: ['init'],
      }),
      step: (_p, s) => {
        const bad: any = { type: 'point', loop: undefined };
        bad.loop = bad; // circular reference inside a VisualCommand payload
        return { ...s, visuals: [bad] };
      },
    };
    expect(() => computeRun(cyclic, {}, 10)).not.toThrow();
    const run = computeRun(cyclic, {}, 10);
    expect(run.telemetry.memBytes).toBe(0);
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
    // dedupe must collapse the repeated 'step' labels to a single stage
    expect(stages).toEqual([{ label: 'init', step: 0 }, { label: 'step', step: 1 }]);
  });
});
