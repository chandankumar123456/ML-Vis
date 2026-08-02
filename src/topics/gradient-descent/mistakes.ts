// src/topics/gradient-descent/mistakes.ts
import type { Mistake } from '../../engine/types';

export const gdMistakes: Mistake[] = [
  {
    id: 'gd-sign',
    pattern: 'Using the wrong sign in the update (x + η·∇f instead of x − η·∇f)',
    example: 'x_{t+1} = x_t + \\eta \\nabla f(x_t)',
    whyWrong: 'The plus sign climbs the function instead of descending. GATE trap: questions on direction of update.',
    gateTrap: true,
  },
  {
    id: 'gd-lr',
    pattern: 'Thinking larger learning rate always converges faster',
    whyWrong: 'Beyond a critical η the update overshoots and diverges (oscillation → blow-up).',
    gateTrap: true,
  },
  {
    id: 'gd-derivative',
    pattern: 'Computing ∂f/∂x but forgetting the chain rule in composed functions',
    example: 'f(x) = (g(x))^2 \\Rightarrow f\' = 2g \\cdot g\'',
    whyWrong: 'Missing the inner derivative g′ is the most common numerical GATE error.',
    gateTrap: true,
  },
];
