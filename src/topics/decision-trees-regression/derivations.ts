// src/topics/decision-trees-regression/derivations.ts
// Line-by-line derivations for the regression-tree math, each step justified
// in plain English.
import type { Derivation } from '../../engine/types';

export const dtrDerivations: Derivation[] = [
  {
    id: 'sse-variance-decomposition',
    title: 'SSE as variance: SSE(S) = n_S·Var(S) and the split gain',
    steps: [
      {
        latex: 'SSE(S) = \\sum_{i \\in S} (y_i - \\bar{y}_S)^2',
        justification: 'Definition of the squared error of the region constant ȳ_S.',
      },
      {
        latex: '\\sum_{i \\in S} (y_i - \\bar{y}_S)^2 = \\sum_{i \\in S} y_i^2 - n_S \\bar{y}_S^2',
        justification: 'Expand the square and use Σ(y_i − ȳ_S) = 0 (the mean cancels the linear term): Σ(y_i² − 2y_iȳ_S + ȳ_S²) = Σy_i² − 2n_Sȳ_S² + n_Sȳ_S².',
      },
      {
        latex: '\\operatorname{Var}(S) = \\frac{1}{n_S} \\sum_{i \\in S} y_i^2 - \\bar{y}_S^2 = \\frac{SSE(S)}{n_S}',
        justification: 'The empirical variance is mean-square minus mean-squared; substituting the previous line shows SSE(S) = n_S·Var(S).',
      },
      {
        latex: '\\Delta SSE = n_S\\operatorname{Var}(S) - n_L\\operatorname{Var}(S_L) - n_R\\operatorname{Var}(S_R)',
        justification: 'Multiply the variance identity by each node size and substitute into ΔSSE = SSE(S) − SSE(S_L) − SSE(S_R). So the CART criterion is literally a variance reduction — the regression-tree analog of information gain.',
      },
    ],
    derivedFrom: ['sse-node', 'sse-reduction'],
  },
  {
    id: 'split-candidates',
    title: 'Why midpoints suffice: the partition changes only at data values',
    steps: [
      {
        latex: 'x_{(1)} < x_{(2)} < \\cdots < x_{(n_S)} \\quad (\\text{distinct sorted values})',
        justification: 'Sort the distinct values of the split feature inside the node.',
      },
      {
        latex: 'L(t) = \\{i \\in S : x_i < t\\}, \\quad R(t) = \\{i \\in S : x_i \\ge t\\}',
        justification: 'A threshold t induces a left/right partition of the samples.',
      },
      {
        latex: 't_1, t_2 \\in (x_{(k)}, x_{(k+1)}) \\Rightarrow L(t_1) = L(t_2)',
        justification: 'No sample value lies strictly between x_(k) and x_(k+1), so the comparison x_i < t cannot change while t stays in that open interval.',
      },
      {
        latex: 'SSE \\text{ depends only on the partition } \\Rightarrow \\text{ evaluate } t_k = \\frac{x_{(k)} + x_{(k+1)}}{2}',
        justification: 'SSE(L) + SSE(R) is a function of the sample sets, not of t itself — so one representative threshold per interval (the midpoint) covers every possible split. Only n_S − 1 candidates need testing.',
      },
    ],
    derivedFrom: ['split-candidates', 'sse-reduction'],
  },
  {
    id: 'leaf-mean',
    title: 'The leaf value that minimizes SSE is the mean',
    steps: [
      {
        latex: '\\min_c \\; \\sum_{i \\in S} (y_i - c)^2',
        justification: 'For a fixed region, the best constant prediction c solves this minimization.',
      },
      {
        latex: '\\frac{d}{dc} \\sum_{i \\in S} (y_i - c)^2 = -2 \\sum_{i \\in S} (y_i - c)',
        justification: 'Differentiate term by term (the squared loss is quadratic in c, so a single critical point).',
      },
      {
        latex: '-2 \\sum_{i \\in S} (y_i - c) = 0 \\Rightarrow \\sum_{i \\in S} y_i = n_S c',
        justification: 'Set the derivative to zero and solve: the sum of residuals must vanish.',
      },
      {
        latex: 'c^* = \\frac{1}{n_S} \\sum_{i \\in S} y_i = \\bar{y}_S',
        justification: 'The minimizer is exactly the sample mean — this is why regression-tree leaves store the mean of their samples, not a fitted linear model.',
      },
    ],
    derivedFrom: ['sse-node', 'leaf-mean'],
  },
  {
    id: 'gini-vs-entropy',
    title: 'Gini vs entropy: two impurity measures with the same ranking',
    steps: [
      {
        latex: 'G(p) = 1 - \\sum_k p_k^2, \\qquad H(p) = -\\sum_k p_k \\log_2 p_k',
        justification: 'Both measure how mixed a classification node is; both are 0 for a pure node and maximal for the uniform distribution.',
      },
      {
        latex: 'G\\left(\\tfrac12, \\tfrac12\\right) = 1 - \\tfrac14 - \\tfrac14 = 0.5, \\qquad H\\left(\\tfrac12, \\tfrac12\\right) = 1',
        justification: 'For the maximally mixed binary node, Gini = 0.5 and entropy = 1 — different scales, same ordering of candidate splits.',
      },
      {
        latex: '\\text{gini: } O(K) \\text{ arithmetic; entropy: } O(K) \\text{ with } K \\log_2 K \\text{ logarithms}',
        justification: 'Gini needs no logarithms, so it is cheaper per node — a real advantage when many candidate splits are scored (e.g. a continuous feature with n_S − 1 midpoints).',
      },
    ],
    derivedFrom: ['gini-impurity'],
  },
];