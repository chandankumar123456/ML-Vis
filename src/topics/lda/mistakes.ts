// src/topics/lda/mistakes.ts
import type { Mistake } from '../../engine/types';

export const ldaMistakes: Mistake[] = [
  {
    id: 'lda-pca-instead',
    pattern: 'Using PCA (unsupervised variance maximization) when class LABELS are available',
    example: 'w_{PCA} = \\text{top eigenvector of } \\Sigma \\quad \\text{(no labels used)} \\quad \\text{vs} \\quad w_{LDA} = S_W^{-1}(\\mu_1-\\mu_2)',
    whyWrong:
      'PCA maximizes total variance wᵀΣw — it has never seen the labels, so it happily projects along the most spread-out axis even when that axis confuses the two classes. ' +
      'LDA maximizes between/within separation using the labels. In the simulation (tilted shared covariance) the two axes are visually distinct: PCA follows the ellipse\'s long direction, LDA cuts along the classes\' separation after whitening. Labels available → prefer LDA (or a supervised model).',
    gateTrap: true,
    relatedConcept: 'lda-between-scatter',
  },
  {
    id: 'lda-forgot-inverse',
    pattern: 'Using the raw mean difference μ₁ − μ₂ as the projection direction (forgetting S_W⁻¹)',
    example: 'w_\\text{wrong} = \\mu_1 - \\mu_2, \\qquad w_\\text{correct} = S_W^{-1}(\\mu_1 - \\mu_2)',
    whyWrong:
      'The mean difference ignores the within-class geometry: a direction that slices diagonally through a wide, elongated class projects the class onto a large spread (big denominator s₀²+s₁² → small J). ' +
      'S_W⁻¹ whitens the within-class scatter first, re-aiming the axis along the classes\' tight directions. The toy-set tests verify that the S_W⁻¹-whitened direction (11/5, −2/5) beats the raw (4, 1) mean gap on the Fisher criterion.',
    gateTrap: true,
    relatedConcept: 'lda-solution',
  },
  {
    id: 'lda-non-gaussian',
    pattern: 'Assuming LDA handles arbitrary (non-Gaussian, multimodal, heteroscedastic) class distributions',
    example: '\\text{bimodal class } C_1 = \\text{two blobs } \\Rightarrow \\text{one }\mu_1 \\text{ and one threshold mangle both}',
    whyWrong:
      'LDA is a generative linear classifier: it assumes each class is ONE Gaussian and all classes SHARE a covariance. ' +
      'A bimodal class collapses to a single mean sitting between its modes (misclassifying the middle), and unequal covariances make the single shared Σ wrong (QDA territory). ' +
      'The failure demos demonstrate both with crafted data; the assumption is the boundary between LDA and more flexible models.',
    gateTrap: true,
    relatedConcept: 'lda-gaussian-assumption',
  },
  {
    id: 'lda-swapped-scatter',
    pattern: 'Confusing within-class and between-class scatter in the Fisher ratio (or in S_W/S_B definitions)',
    example: 'J_\\text{wrong} = \\frac{w^T S_W w}{w^T S_B w} \\quad\\text{(maximizing compactness instead of separation)}',
    whyWrong:
      'J = wᵀS_Bw / wᵀS_Ww: BETWEEN in the numerator (spread the means apart), WITHIN in the denominator (keep each class tight). ' +
      'S_W lives on the class-INTERNAL deviations (xᵢ − μ_c); S_B on the BETWEEN-means difference (μ₁ − μ₂). ' +
      'Flipping them turns LDA into "find the most compact, least separating direction" — a complete inversion of the objective, and a classic GATE distractor in T/F questions.',
    gateTrap: true,
    relatedConcept: 'lda-within-scatter',
  },
  {
    id: 'lda-perfect-separation',
    pattern: 'Believing LDA achieves zero error when trained on overlapping Gaussians',
    example: '\\text{Bayes error } > 0 \\Rightarrow \\text{no linear threshold reaches } 0 \\text{ train error on overlapping classes}',
    whyWrong:
      'LDA is a single linear threshold — if the projected class densities overlap, the best threshold still misclassifies the overlap region. ' +
      'The sweep\'s training error metric is exactly this honesty: on the default overlap-heavy configuration the optimal axis\'s error is visibly > 0. ' +
      'Claiming "LDA perfectly separates" only holds for truly separable draws; otherwise it is the irreducible Bayes error of the linear model.',
    gateTrap: false,
    relatedConcept: 'lda-threshold',
  },
];