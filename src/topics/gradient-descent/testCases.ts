// src/topics/gradient-descent/testCases.ts
import type { TestCase } from '../../engine/types';

export const gdTestCases: TestCase[] = [
  {
    name: 'converges on quadratic at lr=0.1',
    params: { f: 'quadratic', x0: 5, learningRate: 0.1 },
    maxSteps: 500,
    expect: {
      finalMetrics: { x: (v: number) => Math.abs(v) < 1e-3, f: (v: number) => Math.abs(v) < 1e-4 },
      converged: true,
    },
  },
  {
    // |1−2η| = 0.4 < 1 → still converges; this documents the boundary behavior
    name: 'converges (slowly) at lr=0.3 on quadratic',
    params: { f: 'quadratic', x0: 1, learningRate: 0.3 },
    maxSteps: 500,
    expect: { converged: true },
  },
  {
    // |1−2η| = 1 → bounded oscillation, never converges → step budget exceeded
    name: 'oscillates at lr=1.0 on quadratic (never converges)',
    params: { f: 'quadratic', x0: 1, learningRate: 1.0 },
    maxSteps: 100,
    expect: { converged: false },
  },
  {
    name: 'emits converged event for lr=0.05',
    params: { f: 'quadratic', x0: 4, learningRate: 0.05 },
    maxSteps: 500,
    expect: { eventLabels: ['converged'] },
  },
  {
    // |1−2η| = 2 > 1 → |x| doubles each step: 2, 4, 8, ... → overflows double range at step ≈ 1023
    name: 'fails (non-finite) at lr=1.5',
    params: { f: 'quadratic', x0: 2, learningRate: 1.5 },
    maxSteps: 1100,
    expect: { converged: false },
  },
];
