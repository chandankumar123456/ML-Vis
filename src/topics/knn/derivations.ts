// src/topics/knn/derivations.ts
import type { Derivation } from '../../engine/types';

export const knnDerivations: Derivation[] = [
  {
    id: 'knn-majority-vote',
    title: 'The Majority-Vote Rule as a Local Posterior Estimate',
    steps: [
      {
        latex: 'P(y = c \\mid q) \\;\\approx\\; \\frac{1}{|N_k(q)|} \\sum_{i \\in N_k(q)} [y_i = c]',
        justification: 'Estimate the class posterior at the query by the empirical class fraction inside its neighborhood — the natural local estimator for a non-parametric (lazy) learner.',
      },
      {
        latex: '\\hat{y}(q) = \\underset{c}{\\arg\\max}\\, P(y = c \\mid q) \\;\\Rightarrow\\; \\hat{y}(q) = \\underset{c}{\\arg\\max} \\sum_{i \\in N_k(q)} [y_i = c]',
        justification: 'Plug the estimate into the Bayes decision rule (pick the class with highest posterior). The 1/|N_k| factor is constant across classes, so the argmax reduces to a raw majority vote.',
      },
      {
        latex: 'k = 1: \\; \\hat{y}(q) = y_{i^*} \\quad \\text{where } i^* = \\arg\\min_i d(p_i, q)',
        justification: 'Special case: a single nearest neighbor means the vote is just that neighbor\'s label — the highest-variance (most overfit) end of the bias–variance dial.',
      },
      {
        latex: '\\text{tie? } \\; \\Rightarrow \\; \\underset{c\\,\\in\\,\\text{tied}}{\\arg\\min}\\, \\min_{i \\in N_k(q),\\, y_i = c} d(p_i, q) \\; \\Rightarrow \\; \\text{lower class index}',
        justification: 'Deterministic tie-break: among classes tied on votes, the class whose closest neighbor (smallest distance within the k-set) wins; on equal distance, the lower class index. This makes the rule total (no random outcomes) and reproducible.',
      },
    ],
    derivedFrom: ['knn-majority-vote'],
  },
  {
    id: 'knn-voronoi-boundary',
    title: 'The k-NN Decision Boundary: Nearest-Neighbor Cells',
    steps: [
      {
        latex: 'V_i = \\{ q : d(p_i, q) \\le d(p_j, q) \\; \\forall j \\}',
        justification: 'For k=1, the query space partitions into Voronoi cells: the set of points closer to p_i than to any other stored point. Each cell inherits p_i\'s label.',
      },
      {
        latex: '\\text{cell boundary}(p_i, p_j) = \\left\\{ q : d(p_i, q) = d(p_j, q) \\right\\}',
        justification: 'Cell boundaries are the perpendicular bisector between two points under L2 (a line in 2D), or the piecewise axis-aligned bisector under L1 — the metric literally shapes the boundary geometry.',
      },
      {
        latex: 'k = 1 \\Rightarrow \\text{region count} \\approx n \\quad\\text{vs}\\quad k = 15 \\Rightarrow \\text{region count } \\downarrow',
        justification: 'With k=1 every stored point gets its own region (overfit, many boundary segments). Larger k averages votes over a neighborhood, merging cells: empirically the region count drops from 51 (k=1) to 35 (k=15) on the default simulation seed.',
      },
    ],
    derivedFrom: ['knn-majority-vote'],
  },
  {
    id: 'knn-loo-unbiased',
    title: 'Leave-One-Out Error: Why Train Error is Misleading for k-NN',
    steps: [
      {
        latex: 'E_{\\text{train}}(k=1) = \\frac{1}{n}\\sum_i [\\hat{y}_{+i}(p_i) \\neq y_i] = 0',
        justification: 'With k=1 the nearest neighbor of p_i (itself, at distance 0) is included in the vote, so every training point is predicted as its own class — train error is exactly 0 by construction.',
      },
      {
        latex: 'E_{\\text{LOO}}(k=1) = \\frac{1}{n}\\sum_i [\\hat{y}_{-i}(p_i) \\neq y_i] \\neq 0 \\text{ in general}',
        justification: 'Excluding the point itself removes the free pass: p_i is now classified by its nearest OTHER point, which can disagree — giving an honest, non-trivial error at k=1 (≈ 0.42 on the default seed).',
      },
      {
        latex: '\\mathbb{E}[E_{\\text{LOO}}] \\approx \\text{true generalization error} \\quad (\\text{nearly unbiased})',
        justification: 'Each fold predicts one held-out point with n−1 training points — the closest achievable unbiased estimator for small n, and the honest error-vs-k curve in this simulation.',
      },
    ],
    derivedFrom: ['knn-loo-error'],
  },
];
