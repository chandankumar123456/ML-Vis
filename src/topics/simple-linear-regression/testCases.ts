// src/topics/simple-linear-regression/testCases.ts
import type { TestCase } from '../../engine/types';

export const slrTestCases: TestCase[] = [
  {
    name: 'least squares fit on clean linear data',
    params: { n: 20, slope: 2, intercept: 1, noise: 0.0, useNormalEquation: true },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        // NOTE: simulation metrics are { w, b, mse } — testCases must reference those exact keys
        w: (v: number) => Math.abs(v - 2) < 0.05,
        b: (v: number) => Math.abs(v - 1) < 0.05,
        mse: (v: number) => v < 0.01,
      },
    },
  },
  {
    name: 'normal equation equals gradient descent optimum (noise=0)',
    params: { n: 30, slope: 1.5, intercept: -0.5, noise: 0.0, useNormalEquation: false, learningRate: 0.01, epochs: 2000 },
    maxSteps: 2000,
    expect: {
      finalMetrics: {
        w: (v: number) => Math.abs(v - 1.5) < 0.05,
        b: (v: number) => Math.abs(v - (-0.5)) < 0.05,
      },
    },
  },
  {
    name: 'outlier shifts the OLS line (robustness demo)',
    params: { n: 20, slope: 2, intercept: 1, noise: 0.0, outlierX: 15, outlierY: -10, useNormalEquation: true },
    maxSteps: 3,
    expect: {
      finalMetrics: { w: (v: number) => v < 2.05 }, // pulled down by outlier
    },
  },
];
