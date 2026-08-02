// src/topics/gradient-descent/module.ts
import type { TopicModule, Params, SimState } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { gdTestCases } from './testCases';
import { gdFormulas } from './formulas';
import { gdDerivations } from './derivations';
import { gdMistakes } from './mistakes';
import { gdQuestions } from './questions';

function gradientOf(f: string, x: number): number {
  switch (f) {
    case 'quadratic': return 2 * x;
    case 'cubic': return 3 * x * x;
    case 'quartic': return 4 * x * x * x;
    default: return 2 * x;
  }
}

function valueOf(f: string, x: number): number {
  switch (f) {
    case 'quadratic': return x * x;
    case 'cubic': return x * x * x;
    case 'quartic': return x * x * x * x;
    default: return x * x;
  }
}

export const simulation = {
  initialState: (p: Params): SimState => {
    const x0 = p.x0 as number;
    const f = p.f as string;
    const g0 = gradientOf(f, x0);
    return {
      algorithm: { x: x0, gradient: gradientOf(f, x0), learningRate: p.learningRate as number, iteration: 0 },
      visuals: [
        { type: 'point', id: 'current', x: x0, y: valueOf(f, x0), color: '#f59e0b' },
      ],
      math: [{ latex: `f(x) = ${f === 'quadratic' ? 'x^2' : f === 'cubic' ? 'x^3' : 'x^4'}`, id: 'f' }],
      narration: Math.abs(g0) < 1e-4
        ? 'Start at x₀. The gradient here is zero — already at a stationary point; the run will report convergence immediately.'
        : g0 > 0
          ? 'Start at x₀. The gradient here is positive, so the function increases to the right — we move left (opposite the gradient).'
          : 'Start at x₀. The gradient here is negative, so the function decreases to the right — we move right (opposite the gradient).',
      explanation: {
        changed: [],
        why: 'Initialization: pick a starting point x₀',
        formulaRef: 'f',
        dependsOn: ['calculus', 'derivative'],
        gateConcepts: ['gradient', 'learning rate', 'convergence'],
      },
      highlights: [{ panel: 'canvas', id: 'current', intensity: 1 }],
      metrics: { x: x0, f: valueOf(f, x0), gradient: gradientOf(f, x0) },
      events: [{ type: 'init', label: 'initialized', step: 0 }],
      timeline: ['Initialization'],
    };
  },

  step: (p: Params, s: SimState): SimState | null => {
    // termination: converged flag set by the previous step → stop cleanly (null = run ends, failedAtStep stays undefined)
    if ((s.algorithm as any).converged) return null;

    const x = s.algorithm.x as number;
    const lr = p.learningRate as number;
    const f = p.f as string;

    // sanity: skip if already diverged (engine would have stopped, but guard anyway)
    if (!Number.isFinite(x)) return null;

    const grad = gradientOf(f, x);
    const xNext = x - lr * grad;
    const iteration = (s.algorithm.iteration as number) + 1;

    // convergence check → emit the converged snapshot, and the NEXT call returns null
    if (Math.abs(grad) < 1e-4) {
      return {
        ...s,
        algorithm: { ...s.algorithm, x, gradient: grad, iteration, converged: true },
        narration: `Converged at x = ${x.toFixed(4)} — gradient is ≈ 0.`,
        events: [...s.events, { type: 'converged', label: 'converged', step: iteration }],
        timeline: [...s.timeline, 'Convergence'],
        visuals: [
          { type: 'point', id: 'current', x, y: valueOf(f, x), color: '#16a34a' },
        ],
      };
    }

    return {
      algorithm: { x: xNext, gradient: gradientOf(f, xNext), learningRate: lr, iteration },
      visuals: [
        { type: 'point', id: 'current', x: xNext, y: valueOf(f, xNext), color: '#f59e0b' },
        { type: 'arrow', id: 'step-arrow', x1: x, y1: valueOf(f, x), x2: xNext, y2: valueOf(f, xNext), color: '#3b82f6' },
      ],
      math: [
        { latex: `x_{${iteration - 1}} = ${x.toFixed(3)}` },
        { latex: `\\nabla f = ${grad.toFixed(3)}` },
        { latex: `x_{${iteration}} = x_{${iteration - 1}} - \\eta \\cdot \\nabla f = ${xNext.toFixed(3)}` },
      ],
      narration: `Step ${iteration}: gradient = ${grad.toFixed(3)} (${grad > 0 ? 'positive → move left' : 'negative → move right'}). x: ${x.toFixed(3)} → ${xNext.toFixed(3)}`,
      explanation: {
        changed: [`x: ${x.toFixed(3)} → ${xNext.toFixed(3)}`],
        why: `Update rule: x ← x − η·∇f with η = ${lr}`,
        formulaRef: 'update',
        dependsOn: ['f', 'grad'],
        gateConcepts: ['update rule', 'learning rate'],
      },
      highlights: [{ panel: 'canvas', id: 'current', intensity: 1 }],
      metrics: { x: xNext, f: valueOf(f, xNext), gradient: gradientOf(f, xNext), iteration },
      events: s.events,
      timeline: [...s.timeline, 'Iteration'],
    };
  },
};

export const gdModule: TopicModule = {
  id: 'gradient-descent',
  title: 'Gradient Descent',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 4, coding: 2, visualization: 2, gateFrequency: 5 },
    estimatedHours: 5,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['calculus', 'simple-linear-regression'],
    relatedTopics: ['simple-linear-regression', 'logistic-regression', 'backpropagation'],
    revision: { quick: '10m', standard: '30m', deep: '1h', mastery: '2h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Geometry: Descent on the Curve' },
      { slot: 'primary', component: 'loss-curve', title: 'Loss over Iterations' },
    ],
    core: [
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: How GD Evolves' },
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer' },
    ],
    advanced: [
      { slot: 'primary', component: 'derivation-player', title: 'Derivation: Update Rule' },
      { slot: 'primary', component: 'question-player', title: 'GATE Questions' },
    ],
  },
  params: [
    { id: 'f', label: 'Function', type: 'select', options: [
      { value: 'quadratic', label: 'x² (convex)' },
      { value: 'cubic', label: 'x³ (non-convex)' },
      { value: 'quartic', label: 'x⁴ (flat minimum)' },
    ], default: 'quadratic' },
    { id: 'x0', label: 'Starting x₀', type: 'number', min: -10, max: 10, step: 0.1, default: 5 },
    { id: 'learningRate', label: 'Learning rate η', type: 'number', min: 0.001, max: 1.0, step: 0.001, default: 0.1 },
  ],
  simulation,
  formulas: gdFormulas,
  derivations: gdDerivations,
  questions: gdQuestions,
  comparisons: [],
  failureDemos: [],
  mistakes: gdMistakes,
  testCases: gdTestCases,

  validateParams: (p) => {
    const issues: string[] = [];
    const lr = p.learningRate as number;
    if (lr <= 0) issues.push('Learning rate must be positive');
    if (lr > 1) issues.push('Learning rate ≥ 1 will oscillate or diverge for this objective');
    return issues;
  },
};

export function register() {
  registerTopic(gdModule);
}
