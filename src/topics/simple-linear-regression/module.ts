// src/topics/simple-linear-regression/module.ts
import type { TopicModule, Params, SimState } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { slrTestCases } from './testCases';
import { slrFormulas } from './formulas';
import { slrDerivations } from './derivations';
import { slrMistakes } from './mistakes';
import { slrQuestions } from './questions';
import { mean, solve2x2 } from '../../lib/math/linAlg';

export interface SlrData { xs: number[]; ys: number[]; }

export function generateData(p: Params): SlrData {
  const n = p.n as number;
  const slope = p.slope as number;
  const intercept = p.intercept as number;
  const noise = p.noise as number;
  const rng = mulberry32(p.seed as number ?? 42);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = -5 + (i / Math.max(1, n - 1)) * 10 + (rng() - 0.5) * 0.4;
    xs.push(x);
    ys.push(slope * x + intercept + (rng() - 0.5) * 2 * noise);
  }
  if (p.outlierX !== undefined && p.outlierY !== undefined) {
    xs.push(p.outlierX as number);
    ys.push(p.outlierY as number);
  }
  return { xs, ys };
}

/** Mulberry32 — deterministic PRNG so runs are reproducible. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function fitNormalEquation(_p: Params, data: SlrData): { w: number; b: number } {
  const { xs, ys } = data;
  const n = xs.length;
  // X = [x 1]; solve [Σx² Σx; Σx n] [w; b] = [Σxy; Σy]
  const sxx = xs.reduce((a, x) => a + x * x, 0);
  const sx = xs.reduce((a, x) => a + x, 0);
  const sxy = xs.reduce((a, x, i) => a + x * ys[i], 0);
  const sy = ys.reduce((a, y) => a + y, 0);
  const [w, b] = solve2x2([[sxx, sx], [sx, n]], [sxy, sy]);
  return { w, b };
}

// ONE epoch of full-batch gradient descent: O(n). Call per step; snapshots capture history naturally.
export function gradientStep(p: Params, data: SlrData, w: number, b: number): { w: number; b: number } {
  const { xs, ys } = data;
  const n = xs.length;
  const lr = p.learningRate as number;
  let dw = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const pred = w * xs[i] + b;
    const err = pred - ys[i];
    dw += 2 * err * xs[i];
    db += 2 * err;
  }
  dw /= n; db /= n;
  return { w: w - lr * dw, b: b - lr * db };
}

function mseOf(w: number, b: number, data: SlrData): number {
  const { xs, ys } = data;
  return mean(xs.map((x, i) => (w * x + b - ys[i]) ** 2));
}

export const simulation = {
  initialState: (p: Params): SimState => {
    const data = generateData(p);
    const useNormal = p.useNormalEquation as boolean;
    let fit;
    if (useNormal) {
      fit = { ...fitNormalEquation(p, data), epoch: 0 };
    } else {
      fit = { ...gradientStep(p, data, 0, 0), epoch: 1 }; // one epoch from w=0, b=0
    }
    const mse = mseOf(fit.w, fit.b, data);
    return {
      algorithm: { w: fit.w, b: fit.b, mode: useNormal ? 'normal-equation' : 'gradient-descent', epoch: fit.epoch },
      visuals: [
        ...data.xs.map((x, i) => ({ type: 'point', id: `d${i}`, x, y: data.ys[i], color: '#64748b' })),
        { type: 'line', id: 'fit-line', points: [[-5, fit.w * -5 + fit.b], [5, fit.w * 5 + fit.b]], color: '#3b82f6' },
      ],
      math: [{ latex: `\\hat{y} = ${fit.w.toFixed(3)} x + ${fit.b.toFixed(3)}`, id: 'hypothesis' }],
      narration: useNormal
        ? `Normal equation solved directly: w = ${fit.w.toFixed(3)}, b = ${fit.b.toFixed(3)}`
        : `Gradient descent epoch 1: w = ${fit.w.toFixed(3)}, b = ${fit.b.toFixed(3)}`,
      explanation: {
        changed: [],
        why: useNormal ? 'Closed-form solution from X^T X and X^T y' : 'One epoch of gradient descent on MSE',
        formulaRef: useNormal ? 'normal-equation' : 'mse',
        dependsOn: ['linear-algebra', 'projection'],
        gateConcepts: ['OLS', 'normal equation', 'least squares'],
      },
      highlights: [],
      metrics: { w: fit.w, b: fit.b, mse },
      events: [{ type: 'fit', label: useNormal ? 'normal-equation' : 'gd-epoch', step: 0 }],
      timeline: ['Data', 'Fit', 'Evaluate'],
    };
  },

  step: (p: Params, s: SimState): SimState | null => {
    // Only step for gradient-descent mode: continue epochs
    if (s.algorithm.mode !== 'gradient-descent') return null;
    const data = generateData(p);
    const epochs = p.epochs as number ?? 2000;
    const currentEpoch = (s.algorithm.epoch as number ?? 1) + 1;
    if (currentEpoch > epochs) return null;
    // incremental: one GD epoch from the running (w, b) — O(n) per snapshot, no recomputation
    const fit = gradientStep(p, data, s.algorithm.w as number, s.algorithm.b as number);
    const mse = mseOf(fit.w, fit.b, data);
    return {
      algorithm: { ...s.algorithm, w: fit.w, b: fit.b, epoch: currentEpoch },
      visuals: [
        ...data.xs.map((x, i) => ({ type: 'point', id: `d${i}`, x, y: data.ys[i], color: '#64748b' })),
        { type: 'line', id: 'fit-line', points: [[-5, fit.w * -5 + fit.b], [5, fit.w * 5 + fit.b]], color: '#3b82f6' },
      ],
      math: [{ latex: `\\hat{y} = ${fit.w.toFixed(3)} x + ${fit.b.toFixed(3)}` }],
      narration: `Epoch ${currentEpoch}: MSE = ${mse.toFixed(4)}`,
      explanation: {
        changed: [`w=${fit.w.toFixed(3)}`, `b=${fit.b.toFixed(3)}`],
        why: `Gradient descent step with η = ${p.learningRate}`,
        formulaRef: 'mse',
        dependsOn: ['gradient-descent'],
        gateConcepts: ['SGD', 'MSE'],
      },
      highlights: [],
      metrics: { w: fit.w, b: fit.b, mse },
      events: [...s.events],
      timeline: ['Fit', 'Evaluate'],
    };
  },
};

export const slrModule: TopicModule = {
  id: 'simple-linear-regression',
  title: 'Simple Linear Regression',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 2, mathematical: 3, coding: 2, visualization: 2, gateFrequency: 5 },
    estimatedHours: 6,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['linear-algebra', 'calculus'],
    relatedTopics: ['gradient-descent', 'ridge-regression', 'multiple-linear-regression'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Geometry: Best-Fit Line' },
      { slot: 'primary', component: 'loss-curve', title: 'MSE over Epochs (GD mode)' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'n', label: 'Number of samples', type: 'number', min: 5, max: 100, step: 1, default: 25 },
    { id: 'slope', label: 'True slope', type: 'number', min: -5, max: 5, step: 0.1, default: 2 },
    { id: 'intercept', label: 'True intercept', type: 'number', min: -5, max: 5, step: 0.1, default: 1 },
    { id: 'noise', label: 'Noise σ', type: 'number', min: 0, max: 3, step: 0.05, default: 0.5 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
    { id: 'useNormalEquation', label: 'Use normal equation', type: 'toggle', default: true },
    { id: 'learningRate', label: 'Learning rate η', type: 'number', min: 0.001, max: 0.1, step: 0.001, default: 0.01 },
    { id: 'epochs', label: 'Epochs (GD)', type: 'number', min: 50, max: 5000, step: 50, default: 500 },
  ],
  simulation,
  formulas: slrFormulas,
  derivations: slrDerivations,
  questions: slrQuestions,
  comparisons: [],
  failureDemos: [],
  mistakes: slrMistakes,
  testCases: slrTestCases,
  lossMetricKey: 'mse',
};

export function register() {
  registerTopic(slrModule);
}
