// src/topics/naive-bayes/failures.ts
import type { FailureDemo } from '../../engine/types';

export const nbFailureDemos: FailureDemo[] = [
  {
    id: 'nb-f1',
    title: 'Correlated features: the naive model double-counts evidence',
    scenario: 'correlated-features',
    params: { nClasses: 2, nPerClass: 25, correlation: 0.9, smoothing: 0.1, seed: 42, queryX1: 2.4, queryX2: 2 },
    narration: 'At ρ = 0.9 the two features share the same latent signal. The naive model treats them as independent, so at the query point (2.4, 2) it reports a strong posterior for class 1 while the true generative model assigns that point to class 0 — the naive boundary stays vertical while the truth tilts.',
    whyItBreaks: 'P(x₁,x₂|C) = P(x₁|C)·P(x₂|C) counts the shared evidence twice. When ρ is high, the product overstates the evidence of the class whose blob the point is near, producing overconfident and often wrong posteriors.',
  },
  {
    id: 'nb-f2',
    title: 'Rare events: an unseen feature value zeros the posterior (no smoothing)',
    scenario: 'rare-events',
    params: { nClasses: 2, nPerClass: 4, correlation: 0.5, smoothing: 0, seed: 42, discrete: true, queryX1: 0, queryX2: 3 },
    narration: 'In discrete mode the query (0, 3) mixes the value ranges of the two classes: x₂ = 3 was never seen in class 0 and x₁ = 0 never in class 1. With α = 0 both class likelihoods are exactly 0 and the posterior is degenerate — the model cannot classify the point at all.',
    whyItBreaks: 'A product likelihood has no "credit" for unseen values: count = 0 forces P(x|C) = 0. Laplace smoothing (count+α)/(n+αV) keeps every value strictly positive, giving the unseen value ε = α/(n+αV) instead of 0.',
  },
  {
    id: 'nb-f3',
    title: 'Continuous features mishandled as categorical: binning throws away the distance structure',
    scenario: 'continuous-mishandled',
    params: { nClasses: 2, nPerClass: 4, correlation: 0.5, smoothing: 1, seed: 42, discrete: true, queryX1: 2.4, queryX2: 2 },
    narration: 'Discrete mode bins the continuous feature space into coarse cells {0,1,2,3} with class 0 on {0,1} and class 1 on {2,3}. A value like x₁ = 2.4 is snapped to the cell x₁ = 2, so the model cannot tell 2.4 apart from 2.9 — all distance information inside a cell is destroyed.',
    whyItBreaks: 'Categorical NB compares exact value matches, so it treats near-identical values as different categories and far-apart values as the same one if they share a bin. Gaussian NB keeps the actual distances via (x−μ)²/σ² and stays smooth.',
  },
];
