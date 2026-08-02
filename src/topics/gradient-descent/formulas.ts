// src/topics/gradient-descent/formulas.ts
import type { Formula } from '../../engine/types';

export const gdFormulas: Formula[] = [
  {
    id: 'f',
    latex: 'f(x) = x^2',
    symbols: [
      { symbol: 'x', meaning: 'parameter value', dimensions: 'scalar' },
      { symbol: 'f', meaning: 'objective (cost) value', dimensions: 'scalar' },
    ],
    assumptions: ['f is differentiable', 'f has a minimum'],
    failureCases: ['Non-differentiable points (abs value at 0)', 'Plateaus stall progress'],
    connections: ['Update rule', 'Gradient'],
    whyWorks: 'A convex bowl gives GD a single attractor.',
  },
  {
    id: 'grad',
    latex: "\\frac{df}{dx} = 2x",
    symbols: [
      { symbol: 'x', meaning: 'parameter', dimensions: 'scalar' },
    ],
    assumptions: ['f differentiable at x'],
    failureCases: ['Cusp points have no gradient'],
    derivesFrom: ['f'],
    connections: ['Update rule'],
    whyWorks: 'Power rule of differentiation.',
  },
  {
    id: 'update',
    latex: 'x_{t+1} = x_t - \\eta \\cdot \\nabla f(x_t)',
    symbols: [
      { symbol: 'x_t', meaning: 'parameter at step t', dimensions: 'scalar' },
      { symbol: '\\eta', meaning: 'learning rate', dimensions: 'scalar' },
      { symbol: '\\nabla f', meaning: 'gradient at x_t', dimensions: 'scalar' },
    ],
    assumptions: ['η small enough for descent'],
    failureCases: ['η too large → divergence', 'η too small → slow'],
    derivesFrom: ['grad'],
    connections: ['f'],
    whyWorks: 'Move opposite the gradient to reduce f locally.',
  },
];
