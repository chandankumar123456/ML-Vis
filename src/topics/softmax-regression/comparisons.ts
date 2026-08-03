// src/topics/softmax-regression/comparisons.ts
import type { Comparison } from '../../engine/types';

export const softmaxComparisons: Comparison[] = [
  {
    id: 'softmax-vs-logistic',
    title: 'Softmax vs Logistic Regression (K = 2 equivalence)',
    topics: ['softmax-regression', 'logistic-regression'],
    axes: [
      {
        axis: 'Output layer',
        entries: [
          { topic: 'softmax-regression', value: 'Softmax over K logits — K probabilities summing to 1' },
          { topic: 'logistic-regression', value: 'Single sigmoid σ(z) = 1/(1 + e^{−z}) — one probability p' },
        ],
      },
      {
        axis: 'Class count',
        entries: [
          { topic: 'softmax-regression', value: 'Any K ≥ 2; K = 2 reduces to logistic' },
          { topic: 'logistic-regression', value: 'Exactly 2 (binary); ŷ₁ = σ(z₁ − z₀) with softmax' },
        ],
      },
      {
        axis: 'Parameters',
        entries: [
          { topic: 'softmax-regression', value: 'W (K×d) + b (K,) — per-class weights' },
          { topic: 'logistic-regression', value: 'w (d) + b (1) — one decision boundary' },
        ],
      },
      {
        axis: 'Loss',
        entries: [
          { topic: 'softmax-regression', value: 'Categorical CE: −Σ log ŷ_{yᵢ}' },
          { topic: 'logistic-regression', value: 'Binary CE: −Σ [y log p + (1−y) log(1−p)]' },
        ],
      },
    ],
    notes: [
      'Mathematically the same family: the 2-class softmax IS σ(z₁ − z₀); logistic regression is the K = 2 special case of softmax regression.',
      'Both are linear classifiers in logit space — softmax just keeps K score functions and normalizes.',
    ],
  },
  {
    id: 'softmax-vs-ovr',
    title: 'Softmax vs One-vs-Rest (OvR)',
    topics: ['softmax-regression', 'logistic-regression'],
    axes: [
      {
        axis: 'Model count',
        entries: [
          { topic: 'softmax-regression', value: 'ONE model with K score functions sharing features' },
          { topic: 'logistic-regression', value: 'K separate binary models (one per class vs the rest)' },
        ],
      },
      {
        axis: 'Loss coupling',
        entries: [
          { topic: 'softmax-regression', value: 'Single CE over all classes — classes compete through the normalizer' },
          { topic: 'logistic-regression', value: 'K independent binary CE losses — no coupling' },
        ],
      },
      {
        axis: 'Probabilities',
        entries: [
          { topic: 'softmax-regression', value: 'By construction Σ ŷ = 1 (proper distribution)' },
          { topic: 'logistic-regression', value: 'K sigmoid scores need calibration/normalization to sum to 1' },
        ],
      },
      {
        axis: 'Tie handling',
        entries: [
          { topic: 'softmax-regression', value: 'Clean argmax over K scores; ties resolved by index' },
          { topic: 'logistic-regression', value: 'Ambiguous when two OvR classifiers both say "positive"' },
        ],
      },
    ],
    notes: [
      'OvR is a reduction trick (any binary learner becomes multiclass); softmax is a direct multiclass model.',
      'Softmax is usually preferred when classes are mutually exclusive; OvR shines when binary experts are easier to tune or classes overlap.',
    ],
  },
  {
    id: 'softmax-vs-nn-output',
    title: 'Softmax as the Neural Network Output Layer',
    topics: ['softmax-regression', 'neural-networks'],
    axes: [
      {
        axis: 'Role',
        entries: [
          { topic: 'softmax-regression', value: 'The WHOLE model: linear logits z_k = w_k·x + b_k' },
          { topic: 'neural-networks', value: 'The LAST layer only: logits z = W·h(x) + b over hidden features h' },
        ],
      },
      {
        axis: 'Representation',
        entries: [
          { topic: 'softmax-regression', value: 'No hidden layers — features are raw x' },
          { topic: 'neural-networks', value: 'Hidden layers learn the feature map; softmax classifies on top' },
        ],
      },
      {
        axis: 'Gradient',
        entries: [
          { topic: 'softmax-regression', value: '∂L/∂w_k = Σ(ŷ_k − 1{y=k})x — closed form per epoch' },
          { topic: 'neural-networks', value: 'Same top-layer gradient, then backprop chains it through hidden layers' },
        ],
      },
      {
        axis: 'Decision regions',
        entries: [
          { topic: 'softmax-regression', value: 'Always linear (hyperplanes)' },
          { topic: 'neural-networks', value: 'Non-linear once hidden layers are non-linear' },
        ],
      },
    ],
    notes: [
      'Softmax + categorical CE is the standard NN multiclass output; softmax regression is that output layer with NO hidden layers.',
      'This topic isolates the output-layer math (the ŷ − 1{y=k} gradient) that backpropagation reuses — the clean link to neural networks.',
    ],
  },
];
