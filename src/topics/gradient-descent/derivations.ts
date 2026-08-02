// src/topics/gradient-descent/derivations.ts
import type { Derivation } from '../../engine/types';

export const gdDerivations: Derivation[] = [
  {
    id: 'gd-update-derivation',
    title: 'Deriving the Gradient Descent Update Rule',
    steps: [
      {
        latex: 'f(x) = x^2',
        justification: 'Start from the objective.',
      },
      {
        latex: "f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h} = 2x",
        justification: 'Definition of derivative; power rule for polynomials.',
      },
      {
        latex: '\\nabla f(x_t) = 2 x_t',
        justification: 'Gradient is the derivative for a scalar objective.',
      },
      {
        latex: 'x_{t+1} = x_t - \\eta \\nabla f(x_t)',
        justification: 'Move opposite the gradient by step η.',
      },
      {
        latex: 'x_{t+1} = x_t - 2 \\eta x_t = x_t (1 - 2\\eta)',
        justification: 'Substitute the gradient of f.',
      },
      {
        latex: '|1 - 2\\eta| < 1 \\iff 0 < \\eta < 1',
        justification: 'Convergence condition for this quadratic.',
      },
    ],
    derivedFrom: ['update'],
  },
];
