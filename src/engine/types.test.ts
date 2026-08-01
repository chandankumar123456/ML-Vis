import { describe, it, expect } from 'vitest';
import type { SimState, SimulationDef, Params } from './types';

const sim: SimulationDef = {
  initialState: (params: Params): SimState => ({
    algorithm: { x: params.x as number },
    visuals: [{ type: 'point', id: 'pt', x: 0, y: 0 }],
    math: [{ latex: 'x = 0' }],
    narration: 'Start',
    explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
    highlights: [],
    metrics: {},
    events: [],
    timeline: ['Init'],
  }),
  step: (_p, s) => (s.algorithm.x as number) > 10 ? null : { ...s, algorithm: { x: (s.algorithm.x as number) + 1 } },
};

describe('SimulationDef contract', () => {
  it('produces increasing states then terminates', () => {
    let s = sim.initialState({ x: 0 });
    let steps = 0;
    while (s) {
      steps++;
      const next = sim.step({ x: 0 }, s);
      if (next === null) break;
      s = next;
    }
    expect(steps).toBeGreaterThanOrEqual(11);
  });
});
