// src/topics/pca/derivations.ts
// Measured anchors (n 40, corr 0.7, rotDeg 30, noise 0.15, seed 42 — the default,
// ALL verified by running the module):
//   Σ = [[0.3701, 0.3791],[0.3791, 1.8463]], λ₁ = 1.9379736879527472,
//   λ₂ = 0.2784478979012177, total variance = 2.216421585853965,
//   PC1 = (0.2350, 0.9720) at θ₁ = 76.41°,
//   R(0°) = Σ₁₁ = 0.3701 (the x-variance), R(θ₁) = λ₁ = 1.9380 (the sweep peak),
//   reconErrK1 = 0.2784 = λ₂, reconErrK2 = 0.
import type { Derivation } from '../../engine/types';

export const pcaDerivations: Derivation[] = [
  {
    id: 'pca-variance-along-w',
    title: 'The Variance Along a Direction w is wᵀΣw',
    steps: [
      {
        latex: 'z_i = w \\cdot (x_i - \\mu) \\quad \\text{(the projection of point } i \\text{ onto the unit direction } w)',
        justification: 'Project each centered point onto the candidate unit direction w. The scalar zᵢ is the point\'s 1-D coordinate along the axis — the quantity the sweep animates for every direction.',
      },
      {
        latex: '\\bar{z} = \\frac{1}{n}\\sum_i z_i = w \\cdot \\Big(\\frac{1}{n}\\sum_i (x_i - \\mu)\\Big) = w \\cdot 0 = 0',
        justification: 'Because the data is centered, the projected values are centered too: the mean of the projections is the projection of the mean, which is zero. This is why PCA centers FIRST — it makes the projected coordinates zero-mean, so their variance has no bias term.',
      },
      {
        latex: '\\operatorname{Var}(z) = \\frac{1}{n}\\sum_i z_i^2 = \\frac{1}{n}\\sum_i \\big(w \\cdot (x_i - \\mu)\\big)^2',
        justification: 'With mean zero, the variance is the average squared projection. Now expand the square: each term is (w·(xᵢ−μ))², a quadratic form in w.',
      },
      {
        latex: '\\operatorname{Var}(z) = w^T \\Big(\\frac{1}{n}\\sum_i (x_i - \\mu)(x_i - \\mu)^T\\Big) w = w^T \\Sigma w',
        justification: 'The sum of outer products (1/n)Σᵢ (xᵢ−μ)(xᵢ−μ)ᵀ IS the covariance matrix Σ (formula pca-covariance). So the variance along w is exactly the Rayleigh quotient wᵀΣw. Measured: along the x-axis (w = (1,0)) this is R = Σ₁₁ = 0.3701; along PC1 it reaches its maximum λ₁ = 1.9380.',
      },
    ],
    derivedFrom: ['pca-covariance'],
  },
  {
    id: 'pca-maximize-lagrange',
    title: 'Maximizing wᵀΣw with ‖w‖ = 1 via Lagrange → Σw = λw',
    steps: [
      {
        latex: '\\max_w \\; w^T \\Sigma w \\quad \\text{s.t.} \\quad w^T w = 1',
        justification: 'The goal: find the unit direction maximizing projected variance. The constraint ‖w‖ = 1 fixes the scale — otherwise multiplying w by c multiplies wᵀΣw by c² and the problem is unbounded.',
      },
      {
        latex: 'L(w, \\lambda) = w^T \\Sigma w - \\lambda (w^T w - 1)',
        justification: 'Attach a Lagrange multiplier λ to the unit-norm constraint. At the optimum the constraint is active (‖w‖ = 1), and λ will turn out to BE the achieved variance.',
      },
      {
        latex: '\\nabla_w L = 2\\Sigma w - 2\\lambda w = 0 \\;\\Rightarrow\\; \\Sigma w = \\lambda w',
        justification: 'Stationarity: the gradient of wᵀΣw is 2Σw (Σ is symmetric) and the gradient of wᵀw is 2w. Setting the gradient to zero gives the eigen-equation — the stationary directions of the Rayleigh quotient ARE the eigenvectors of Σ.',
      },
      {
        latex: 'w^T \\Sigma w = \\lambda w^T w = \\lambda \\quad \\text{(at the optimum, with } \\|w\\| = 1)',
        justification: 'Left-multiply Σw = λw by wᵀ: wᵀΣw = λwᵀw = λ. The Lagrange multiplier equals the projected variance along the eigenvector. So the LARGEST eigenvalue λ₁ is the maximum possible variance, achieved exactly at its eigenvector v₁ = PC1 — measured 1.9380 on the default run, the peak of the sweep curve.',
      },
      {
        latex: '\\theta = \\tfrac{1}{2}\\operatorname{atan2}(2b, a - c), \\quad v_1 = (\\cos\\theta, \\sin\\theta), \\quad \\lambda_{1,2} = \\frac{a + c \\pm \\sqrt{(a-c)^2 + 4b^2}}{2}',
        justification: 'The exact 2×2 closed form: writing wᵀΣw in polar form shows the maximum sits at θ = ½·atan2(2b, a−c), and plugging that direction into the quadratic form yields the two eigenvalues. No iteration is needed — this topic solves PCA exactly, mirroring the adjugate closed form of lda.',
      },
    ],
    derivedFrom: ['pca-eigenequation', 'pca-rayleigh'],
  },
  {
    id: 'pca-reconstruction-error',
    title: 'Reconstruction Error = Sum of the Dropped Eigenvalues',
    steps: [
      {
        latex: 'x_i - \\mu = z_{i1} v_1 + z_{i2} v_2, \\quad z_{ik} = v_k \\cdot (x_i - \\mu)',
        justification: 'The two PCs are an orthonormal basis of the plane, so every centered point decomposes exactly into its two scores: the projection onto PC1 plus the projection onto PC2.',
      },
      {
        latex: '\\hat{x}_i^{(k)} = \\mu + \\sum_{j \\le k} z_{ij} v_j \\;\\Rightarrow\\; x_i - \\hat{x}_i^{(1)} = z_{i2} v_2',
        justification: 'Reconstructing with k = 1 PC keeps only the first term; the residual of each point is exactly its PC2 component — a vector along v₂ whose length is |zᵢ₂|.',
      },
      {
        latex: '\\text{error}_1 = \\frac{1}{n}\\sum_i \\|x_i - \\hat{x}_i^{(1)}\\|^2 = \\frac{1}{n}\\sum_i z_{i2}^2 \\;\\|v_2\\|^2 = \\frac{1}{n}\\sum_i z_{i2}^2',
        justification: 'The squared residual of point i is zᵢ₂² (v₂ is unit length). Summing over points gives the average squared score on PC2 — which is, by the first derivation, exactly the variance of the projections onto v₂.',
      },
      {
        latex: '\\frac{1}{n}\\sum_i z_{i2}^2 = v_2^T \\Sigma v_2 = \\lambda_2 \\;\\Rightarrow\\; \\text{error}_1 = \\lambda_2, \\quad \\text{error}_2 = 0',
        justification: 'The variance along v₂ is the Rayleigh quotient v₂ᵀΣv₂, which equals the eigenvalue λ₂ (derivation 2). So the reconstruction error with k = 1 PC is exactly the dropped eigenvalue; with k = 2 the residual is zero. Measured on the default run: error₁ = 0.2784 = λ₂ to 1e-12, error₂ = 0 — the identity the tests assert.',
      },
    ],
    derivedFrom: ['pca-projection', 'pca-reconstruction'],
  },
];