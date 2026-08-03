// src/topics/naive-bayes/mistakes.ts
import type { Mistake } from '../../engine/types';

export const nbMistakes: Mistake[] = [
  {
    id: 'nb-m1',
    pattern: 'Ignoring the prior — treating P(C) as uniform when the training set is imbalanced',
    example: 'P(C_0 \\mid x) \\propto P(x \\mid C_0)\\ \\text{(prior dropped!)}, \\quad n_{C_0} = 90, \\ n_{C_1} = 10',
    whyWrong: 'Bayes theorem multiplies by the prior. With equal likelihoods, class 0 (90% of data) should win 9:1; dropping the prior forces a coin flip at equal likelihoods and biases the boundary toward the frequent class.',
    gateTrap: true,
    relatedConcept: 'prior',
  },
  {
    id: 'nb-m2',
    pattern: 'Multiplying raw unsmoothed zero probabilities (unseen value kills the class)',
    example: 'P(x \\mid C_0) = P(x_1\\mid C_0) \\cdot \\underbrace{0}_{\\text{unseen } x_2} = 0',
    whyWrong: 'The naive likelihood is a product — a single zero factor annihilates the class entirely, even if every other feature strongly supports it. Laplace smoothing (count+α)/(n+αV) keeps every value strictly positive.',
    gateTrap: true,
    relatedConcept: 'laplace-smoothing',
  },
  {
    id: 'nb-m3',
    pattern: 'Assuming independence holds when features are correlated (double counting evidence)',
    example: 'x_1 \\approx x_2 \\text{ (shared latent)} \\Rightarrow P(x_1, x_2\\mid C) \\neq P(x_1\\mid C)P(x_2\\mid C)',
    whyWrong: 'Two correlated features carry the same underlying signal; the naive product counts it twice, inflating the posterior of the winning class. The simulator shows the full-covariance posterior diverging from the naive one as ρ grows.',
    gateTrap: true,
    relatedConcept: 'independence',
  },
  {
    id: 'nb-m4',
    pattern: 'No log-space computation → underflow on products of many small likelihoods',
    example: '\\prod_{j=1}^{400} 0.01 = 10^{-800} \\to 0 \\text{ in double precision; } \\ \\sum_{j=1}^{400}\\log 0.01 = -1842 \\text{ stays finite}',
    whyWrong: 'A product of hundreds of probabilities collapses to 0 in floating point, making every class look equally impossible. Working in log space converts the product to a finite sum and preserves the argmax (log is monotone).',
    gateTrap: false,
    relatedConcept: 'log-posterior',
  },
];
