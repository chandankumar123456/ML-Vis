// src/topics/naive-bayes/derivations.ts
import type { Derivation } from '../../engine/types';

export const nbDerivations: Derivation[] = [
  {
    id: 'nb-posterior',
    title: 'The Naive Bayes posterior from Bayes theorem + independence',
    steps: [
      {
        latex: 'P(C \\mid x) = \\frac{P(x \\mid C)\\, P(C)}{P(x)}',
        justification: 'Bayes theorem — rewrite the target posterior in terms of estimable quantities.',
      },
      {
        latex: 'P(x \\mid C) = P(x_1, x_2, \\dots, x_d \\mid C)',
        justification: 'P(x|C) is the joint class-conditional likelihood of all d features.',
      },
      {
        latex: 'P(x_1, x_2, \\dots, x_d \\mid C) = P(x_1\\mid C)\\,P(x_2\\mid C, x_1)\\, \\cdots\\, P(x_d \\mid C, x_1, \\dots, x_{d-1})',
        justification: 'Product rule of probability — in general each feature needs all earlier features as conditioning context.',
      },
      {
        latex: 'P(x_j \\mid C, x_1, \\dots, x_{j-1}) \\;\\approx\\; P(x_j \\mid C) \\quad \\forall j',
        justification: 'THE naive assumption: features are conditionally independent given the class. This is the only approximation in the whole derivation.',
      },
      {
        latex: 'P(x \\mid C) = \\prod_{j=1}^{d} P(x_j \\mid C)',
        justification: 'Substituting the assumption collapses the chain product into a product of univariate factors — the "naive" likelihood.',
      },
      {
        latex: 'P(C \\mid x) = \\frac{P(C)\\, \\prod_{j=1}^{d} P(x_j \\mid C)}{P(x)} \\;\\propto\\; P(C)\\, \\prod_{j=1}^{d} P(x_j \\mid C)',
        justification: 'Plugging back into Bayes: P(x) is constant across classes, so for argmax prediction it is dropped.',
      },
    ],
    derivedFrom: ['bayes'],
  },
  {
    id: 'nb-logspace',
    title: 'Why naive Bayes is computed in log space (underflow)',
    steps: [
      {
        latex: 'P(x \\mid C) = \\prod_{j=1}^{d} P(x_j \\mid C)',
        justification: 'The naive likelihood is a product of d probabilities, each typically well below 1.',
      },
      {
        latex: '\\text{double precision underflows below } \\approx 10^{-308}',
        justification: 'A product of many small probabilities reaches 0 in floating point long before the true value is mathematically 0.',
      },
      {
        latex: '\\log\\Big(\\prod_j P(x_j\\mid C)\\Big) = \\sum_j \\log P(x_j\\mid C)',
        justification: 'The log turns the underflowing product into a sum of moderate negative numbers — perfectly representable.',
      },
      {
        latex: '\\arg\\max_C P(C\\mid x) = \\arg\\max_C \\Big[\\log P(C) + \\sum_j \\log P(x_j\\mid C)\\Big]',
        justification: 'log is strictly increasing, so the argmax of the posterior is unchanged by working in log space.',
      },
      {
        latex: '\\log P(x) = m + \\log\\sum_C \\exp\\big(\\ell_C - m\\big), \\qquad m = \\max_C \\ell_C',
        justification: 'Normalization is recovered with the logsumexp trick (subtract the max to avoid overflow), giving true probabilities.',
      },
    ],
    derivedFrom: ['bayes', 'log-posterior'],
  },
];
