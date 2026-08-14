// src/topics/decision-trees-regression/formulas.ts
// Regression-tree math (CART, SSE objective) with the Gini impurity formula
// taught as the CLASSIFICATION analog (the plan's two-lens design — the
// simulation itself is pure regression).
import type { Formula } from '../../engine/types';

export const dtrFormulas: Formula[] = [
  {
    id: 'gini-impurity',
    latex: 'G = 1 - \\sum_{k} p_k^2',
    symbols: [
      { symbol: 'p_k', meaning: 'fraction of class-k samples in the node', dimensions: '0..1, Σ p_k = 1' },
      { symbol: 'G', meaning: 'Gini impurity — 0 for a pure node, maximized by a uniform class mix', dimensions: '0..0.5 (binary)' },
    ],
    assumptions: ['Classification node (class counts, not regression targets)', 'p_k estimated as relative frequencies'],
    failureCases: [
      'Applying Gini to regression targets: y is continuous, so "class fractions" are undefined — regression trees minimize SSE, not Gini',
      'Gini ignores class ORDER/severity: two errors of different classes are weighted equally, so it is wrong when misclassification costs differ',
    ],
    derivesFrom: ['sse-node'],
    connections: ['Entropy H = −Σp log₂p (same ranking, Gini is log-free and cheaper)'],
    whyWorks: "A pure node has p_k = 1 for one class, so Σp_k² = 1 and G = 0. The most mixed binary node (p = 0.5/0.5) gives G = 1 − 0.25 − 0.25 = 0.5 — G measures how often a random sample would be mislabeled if labeled by the node's majority.",
  },
  {
    id: 'sse-node',
    latex: 'SSE(S) = \\sum_{i \\in S} (y_i - \\hat{y}_S)^2,\\quad \\hat{y}_S = \\frac{1}{n_S}\\sum_{i \\in S} y_i',
    symbols: [
      { symbol: 'S', meaning: 'samples in one node region', dimensions: 'index set' },
      { symbol: 'y_i', meaning: 'regression target of sample i', dimensions: 'output units' },
      { symbol: '\\hat{y}_S', meaning: 'node constant = mean of y in S (the leaf prediction)', dimensions: 'output units' },
      { symbol: 'n_S', meaning: 'number of samples in S', dimensions: 'count' },
    ],
    assumptions: ['Squared-error loss', 'Leaves predict a single constant (not a linear model)'],
    failureCases: [
      'SSE is squared: a single outlier far from the others inflates a node\'s SSE and drags its mean (leaf value) toward it',
      'A node with one sample always has SSE = 0 — so min-leaf constraints exist to stop the tree from memorizing single points',
    ],
    connections: ['Leaf mean (derives from this formula)', 'SSE reduction', 'Variance decomposition'],
    whyWorks: 'For a fixed region S, the mean ȳ_S minimizes Σ(y_i − c)² over all constants c (derivation: leaf-mean). SSE(S) is therefore the smallest error a constant fit can achieve on S — the honest "cost" of labeling the region with one number.',
  },
  {
    id: 'sse-reduction',
    latex: '\\Delta SSE = SSE(S) - SSE(S_L) - SSE(S_R)',
    symbols: [
      { symbol: 'SSE(S)', meaning: 'SSE of the parent node before splitting', dimensions: 'squared units' },
      { symbol: 'SSE(S_L), SSE(S_R)', meaning: 'SSE of the left/right child after the split', dimensions: 'squared units' },
      { symbol: '\\Delta SSE', meaning: 'gain of the split — CART picks the split with the largest Δ', dimensions: 'squared units' },
    ],
    assumptions: ['Binary splits (CART)', 'Greedy: only the immediate gain matters at each node'],
    failureCases: [
      'Greedy non-optimality: a split with a smaller immediate gain can unlock a much better follow-up split — CART never looks ahead, so it can miss the globally best tree',
      'ΔSSE = 0 when y is constant in S (or the split cannot separate different values) — the node must stop',
    ],
    derivesFrom: ['sse-node'],
    connections: ['Variance decomposition: ΔSSE = n_S·Var(S) − n_L·Var(S_L) − n_R·Var(S_R)'],
    whyWorks: 'Every split replaces one constant region with two, so the fit can only get closer to the training points: the total error cannot rise, and the gain measures exactly how much the two new constants buy.',
  },
  {
    id: 'split-candidates',
    latex: 't \\in \\left\\{ \\frac{x_{(i)} + x_{(i+1)}}{2} \\right\\}',
    symbols: [
      { symbol: 'x_{(i)}', meaning: 'sorted distinct values of the split feature in the node', dimensions: 'feature units' },
      { symbol: 't', meaning: 'split threshold — left: x < t, right: x ≥ t', dimensions: 'feature units' },
    ],
    assumptions: ['Continuous numeric feature', 'Threshold placed between two consecutive distinct observed values'],
    failureCases: [
      'Categorical features: midpoints are meaningless — the ordering is arbitrary (CART then needs a different encoding, e.g. binary indicator splits)',
      'Duplicate x values collapse candidates: only the distinct sorted values matter, so n points may yield far fewer than n−1 midpoints',
    ],
    derivesFrom: ['sse-reduction'],
    connections: ['Why midpoints suffice: the partition changes only when t crosses a data value'],
    whyWorks: 'Between two consecutive x values the partition of the samples is IDENTICAL for every threshold in the open interval, and SSE depends only on the partition — so the midpoint is a canonical representative of the whole interval (derivation: split-candidates).',
  },
  {
    id: 'leaf-mean',
    latex: '\\hat{y}(x) = \\frac{1}{n_S} \\sum_{i \\in S} y_i \\quad (x \\in \\text{region } S)',
    symbols: [
      { symbol: 'S', meaning: 'leaf region containing x', dimensions: 'index set' },
      { symbol: '\\hat{y}(x)', meaning: 'prediction at x — a PIECEWISE-CONSTANT (step) function of x', dimensions: 'output units' },
    ],
    assumptions: ['Squared-error loss', 'No extrapolation model beyond the leaves'],
    failureCases: [
      'Extrapolation: outside the observed x range the prediction is still the constant of the nearest leaf — the flat tail is a guess, not a trend (a linear model would slope away)',
      'The mean is not robust: one extreme y in a leaf shifts the constant for the whole region',
    ],
    derivesFrom: ['sse-node'],
    connections: ['kNN (local constant fits)', 'Bias–variance: depth controls step count'],
    whyWorks: 'The prediction is constant on each region, so the fitted function is a step function whose jump points are the split thresholds — depth adds steps, and too many steps overfit the noise.',
  },
  {
    id: 'train-test-error',
    latex: '\\text{MSE}_{split} = \\frac{1}{n_{split}}\\sum_{i \\in split} (y_i - \\hat{y}(x_i))^2',
    symbols: [
      { symbol: 'split', meaning: 'training subset (first 70%) or held-out test subset', dimensions: 'index set' },
      { symbol: '\\hat{y}(x_i)', meaning: 'tree prediction of sample i', dimensions: 'output units' },
    ],
    assumptions: ['Independent, identically distributed train/test draws'],
    failureCases: [
      'Reporting only train error: it falls monotonically with depth (the tree memorizes), hiding the U-shaped test error',
      'Test split too small: the test error wobbles run to run (a 30% test split on n = 30 is only 9 points)',
    ],
    derivesFrom: ['leaf-mean'],
    connections: ['Overfitting / bias–variance (deep trees = high variance)'],
    whyWorks: 'Train error measures fit, test error measures generalization. A deep regression tree drives train error toward 0 while test error bottoms out then rises — the two curves diverging IS the overfitting signature.',
  },
];