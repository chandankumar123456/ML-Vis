// src/topics/simple-linear-regression/mistakes.ts
import type { Mistake } from '../../engine/types';

export const slrMistakes: Mistake[] = [
  {
    id: 'slr-transpose',
    pattern: 'Forgetting the transpose in X^T X or X^T y',
    example: 'X^T X \\neq X X',
    whyWrong: 'Dimension mismatch produces wrong matrix — classic GATE numerical trap.',
    gateTrap: true,
  },
  {
    id: 'slr-invertibility',
    pattern: 'Assuming (X^T X) is always invertible',
    whyWrong: 'Perfect multicollinearity (or n < d) makes it singular; GATE asks what happens then.',
    gateTrap: true,
  },
  {
    id: 'slr-outlier',
    pattern: 'Believing OLS is robust to outliers',
    whyWrong: 'Squared error magnifies outlier influence — one extreme point can rotate the line.',
    gateTrap: false,
  },
];
