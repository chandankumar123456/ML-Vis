// src/topics/mle/formulas.ts
// Measured anchors (seed 42 throughout — see testCases.test.ts):
//   coin pTrue 0.7: p̂ = 0.707 at n=1000 (k = 707); score(p̂) = 1.14e−13;
//     nll@n=1000 = 0.6048 below H(0.7) = 0.6108643020548935.
//   gaussian μ=1, σ=1.5: n=10 → σ̂² = 0.850428 (÷n) vs σ̂²ᵤ = 0.944920 (÷(n−1)),
//     ratio exactly n/(n−1) = 10/9; n=1000 → gap 0.00222 (ratio 1000/999);
//     μ̂ = 0.758533 at n=100, → 0.929392 at n=1000.
//   linear slope 1.5, intercept −0.5, noise 0.8: at n=100 θ̂ = (−0.467261, 1.590870),
//     rss = 71.052727, σ̂² = RSS/n = 0.710527, σ̂²ᵤ = RSS/(n−2) = 0.725028;
//     XᵀX = [[100, 10.494344],[10.494344, 265.756110]].
import type { Formula } from '../../engine/types';

export const mleFormulas: Formula[] = [
  {
    id: 'mle-likelihood',
    latex: 'L(\\theta) = \\prod_{i=1}^{n} p(x_i \\mid \\theta)',
    symbols: [
      { symbol: 'L(\\theta)', meaning: 'the LIKELIHOOD — probability of the observed sample as a function of the parameter θ', dimensions: 'probability (product of n densities)' },
      { symbol: 'p(x_i \\mid \\theta)', meaning: 'probability (or density) of the i-th observation under θ', dimensions: 'per-observation probability/density' },
      { symbol: '\\theta', meaning: 'the unknown parameter(s): p (coin), (μ, σ²) (gaussian), (β₀, β₁, σ²) (linear)', dimensions: 'parameter units' },
    ],
    assumptions: ['Observations are independent and identically distributed (i.i.d.)', 'The parametric family p(x|θ) is specified correctly'],
    failureCases: ['Underflow: for n = 1000 the product of 1000 factors below 1 underflows — measured L(0.707) = 2.1465e−263 and L(0.9) = 0 in double precision, while ℓ stays finite (−604.8)', 'L is a product, so a single zero-probability factor makes the whole likelihood 0'],
    derivesFrom: [],
    derivationIds: ['mle-likelihood-derivation'],
    connections: ['Log-likelihood', 'Bayes rule (posterior ∝ likelihood × prior)', 'Maximum likelihood estimation'],
    whyWorks: 'L(θ) answers: how likely was THIS sample under each candidate θ? Maximizing L over θ picks the parameter that makes the observed data most probable. It is a function of θ (not a probability distribution over θ — the total area under L as a function of θ need not be 1). On the seeded coin run the maximizer is p̂ = 0.707, the empirical frequency of 707 heads in 1000 flips.',
  },
  {
    id: 'mle-loglik',
    latex: '\\ell(\\theta) = \\ln L(\\theta) = \\sum_{i=1}^{n} \\ln p(x_i \\mid \\theta)',
    symbols: [
      { symbol: '\\ell(\\theta)', meaning: 'the LOG-LIKELIHOOD — a sum instead of a product', dimensions: 'log-probability' },
      { symbol: '\\ln L(\\theta)', meaning: 'natural log of the likelihood; ln is strictly increasing, so argmax ℓ = argmax L', dimensions: 'monotone transform' },
    ],
    assumptions: ['Every factor p(xᵢ|θ) is strictly positive (log of zero is −∞)', 'ln is strictly monotone — it never moves the maximizer'],
    failureCases: ['Working with L directly loses underflow: at n=1000, L(0.9) = 0 exactly (product of 0.9 and 0.1 powers), while ℓ(0.9) = −604.8 remains perfectly representable', 'ℓ can be very negative (large n); comparing ℓ values across DIFFERENT n is meaningless (each sample contributes a log-probability term)'],
    derivesFrom: ['mle-likelihood'],
    derivationIds: ['mle-likelihood-derivation'],
    connections: ['Score function', 'Cross-entropy loss', 'Likelihood'],
    whyWorks: 'The log turns the product into a sum — numerically stable and analytically easier (derivatives of sums are sums of derivatives). Because ln is strictly increasing, the SAME θ maximizes L and ℓ: the measured grid at n=10 shows both L(p) and ℓ(p) peaking at p = 0.8 = k/n. The average −ℓ(θ̂)/n is the nllPerSample loss curve this topic plots (lossMetricKey), which descends toward the true entropy as n grows: 0.5004 → 0.6048 for the coin, with H(0.7) = 0.610864 the limit.',
  },
  {
    id: 'mle-score',
    latex: 'S(\\theta) = \\frac{\\partial \\ell}{\\partial \\theta} = \\sum_{i=1}^{n} \\frac{\\partial}{\\partial \\theta} \\ln p(x_i \\mid \\theta)',
    symbols: [
      { symbol: 'S(\\theta)', meaning: 'the SCORE function — the derivative of the log-likelihood', dimensions: 'per parameter unit' },
      { symbol: '\\frac{\\partial \\ell}{\\partial \\theta}', meaning: 'gradient of ℓ with respect to the parameter vector', dimensions: 'per parameter unit' },
    ],
    assumptions: ['ℓ is differentiable in θ (true for Bernoulli, Gaussian, linear families)', 'The maximizer is interior, not on a boundary'],
    failureCases: ['The score can vanish at a LOCAL maximum of a non-concave ℓ — for a mixture ½N(μ,1)+½N(−μ,1) the seeded grid shows local maxima at μ ≈ ±2 (ℓ = −8.412) around a local minimum at μ = 0 (ℓ = −13.676): S = 0 there is NOT the global MLE', 'At the Bernoulli boundary (k = 0 or k = n) the derivative is undefined without clamping'],
    derivesFrom: ['mle-loglik'],
    connections: ['MLE stationarity S(θ̂) = 0', 'Fisher information (variance of the score)', 'Gradient descent'],
    whyWorks: 'The MLE is found by solving S(θ) = 0 — "set the score to zero". For the Bernoulli family S(p) = k/p − (n−k)/(1−p), measured to vanish at p̂ = 0.707 (1.14e−13 on the seeded stream). For linear regression the score is the matrix form S(θ) = Xᵀ(y − Xθ), measured at [8.4e−15, 8.5e−14] at the fitted θ̂ — the stationarity condition IS the normal equation.',
  },
  {
    id: 'mle-bernoulli',
    latex: 'S(p) = \\frac{k}{p} - \\frac{n-k}{1-p} = 0 \\;\\Rightarrow\\; \\hat p = \\frac{k}{n}',
    symbols: [
      { symbol: 'k', meaning: 'number of heads (successes) in n flips', dimensions: 'count' },
      { symbol: '\\hat p = k/n', meaning: 'the Bernoulli MLE — the empirical frequency of heads', dimensions: 'probability' },
    ],
    assumptions: ['i.i.d. Bernoulli(p) flips', '0 < p < 1 (interior maximizer)'],
    failureCases: ['At k = 0 or k = n the MLE is the boundary value 0 or 1 — the log-likelihood is −∞ there and the score must be clamped (the module clamps p to [1e−12, 1−1e−12])', 'Small n: p̂ = k/n is unbiased but high-variance — at n=10 the seeded draw gives p̂ = 0.8, error 0.1'],
    derivesFrom: ['mle-score', 'mle-loglik'],
    derivationIds: ['mle-bernoulli-derivation'],
    connections: ['Cross-entropy loss', 'Consistency', 'Maximum likelihood estimation'],
    whyWorks: 'Solving k/p = (n−k)/(1−p) gives k(1−p) = (n−k)p → k = np → p̂ = k/n. On the seeded stream k = 707 at n=1000 gives p̂ = 0.707 against the true p = 0.7. Consistency is visible in the sweep: |p̂ − p| = 0.1 at n=10, 0.04 at n=100, 0.007 at n=1000 — the estimate tightens toward p as n grows.',
  },
  {
    id: 'mle-gaussian-mean',
    latex: '\\frac{\\partial \\ell}{\\partial \\mu} = \\frac{1}{\\sigma^2}\\sum_{i=1}^{n}(x_i - \\mu) = 0 \\;\\Rightarrow\\; \\hat\\mu = \\bar x = \\frac{1}{n}\\sum_{i=1}^{n} x_i',
    symbols: [
      { symbol: '\\hat\\mu', meaning: 'the Gaussian MLE of the mean — the sample mean', dimensions: 'same units as x' },
      { symbol: '\\bar x', meaning: 'arithmetic mean of the sample', dimensions: 'same units as x' },
    ],
    assumptions: ['i.i.d. Gaussian samples with finite σ²', 'σ² is fixed while solving for μ (the two solve simultaneously; μ̂ does not depend on σ²)'],
    failureCases: ['The sample mean is sensitive to outliers (one huge draw drags μ̂ arbitrarily far)', 'μ̂ ≈ μ only in expectation — the seeded n=10 draw gives μ̂ = 0.060 against μ = 1 (error 0.940); at n=1000 it is 0.929 (error 0.071)'],
    derivesFrom: ['mle-loglik'],
    derivationIds: ['mle-gaussian-derivation'],
    connections: ['Sample mean', 'Central limit theorem', 'Consistency'],
    whyWorks: 'The mean solves the score equation exactly: Σ(xᵢ − μ) = 0 ⟺ μ = Σxᵢ/n. The measured n=100 run gives μ̂ = 0.758533 (true μ = 1), and the module\'s distribution-view draws the fitted N(μ̂, σ̂²) curve beside the true N(1, 2.25) — the fitted curve slides onto the true one as n grows.',
  },
  {
    id: 'mle-gaussian-var',
    latex: '\\hat\\sigma^2 = \\frac{1}{n}\\sum_{i=1}^{n} (x_i - \\hat\\mu)^2 \\;\\;\\text{— the MLE divides by } n, \\text{ NOT } n-1',
    symbols: [
      { symbol: '\\hat\\sigma^2', meaning: 'the Gaussian MLE of the variance — the BIASED (÷n) estimator', dimensions: 'squared units of x' },
      { symbol: '\\frac{1}{n}\\sum (x_i - \\hat\\mu)^2', meaning: 'mean squared deviation from the fitted mean', dimensions: 'squared units of x' },
    ],
    assumptions: ['i.i.d. Gaussian samples', 'μ is unknown and replaced by its MLE μ̂ (this is what makes the ÷n estimator biased)'],
    failureCases: ['The ÷n estimator is BIASED (E[σ̂²] = (n−1)/n·σ²): measured at n=10, σ̂² = 0.850428 while the unbiased ÷(n−1) estimate is σ̂²ᵤ = 0.944920 — a 10% gap that shrinks as 1/n (0.01 at n=100, 0.001 at n=1000)', 'Plugging the ÷(n−1) sample variance into a Gaussian-MLE exam answer is THE classic trap — the MLE is ÷n by the stationarity condition'],
    derivesFrom: ['mle-gaussian-mean', 'mle-score'],
    derivationIds: ['mle-gaussian-derivation'],
    connections: ['Sample variance', 'Bias–variance tradeoff', 'Chi-squared distribution (sampling distribution)'],
    whyWorks: 'Setting ∂ℓ/∂σ² = −n/(2σ²) + Σ(xᵢ−μ̂)²/(2σ⁴) = 0 forces σ̂² = Σ(xᵢ−μ̂)²/n — the MLE divides by n because it estimates the spread AROUND THE FITTED MEAN (μ̂ is chosen to minimize exactly those deviations, so the residual squares are systematically small). The unbiased ÷(n−1) estimator is reported alongside in the module (metrics sigmaUnbSq, and the σ̂²ᵤ = RSS/(n−2) linear analogue) so the gap σ̂²ᵤ − σ̂² = 0.02341 at n=100 can be watched shrink to 0.00222 at n=1000.',
  },
  {
    id: 'mle-score-matrix',
    latex: 'S(\\theta) = X^T (y - X\\theta) = 0 \\;\\Rightarrow\\; \\text{the MLE is the least-squares solution}',
    symbols: [
      { symbol: 'X', meaning: 'n×2 design matrix [1 x] — a column of ones and the predictor', dimensions: 'n×2' },
      { symbol: 'X^T (y - X\\theta)', meaning: 'the matrix score: each column of X dotted with the residual vector', dimensions: '2×1' },
      { symbol: 'y - X\\theta', meaning: 'the residual vector at parameter θ', dimensions: 'n×1' },
    ],
    assumptions: ['Gaussian noise with constant variance (homoscedastic)', 'X has full column rank — det(XᵀX) > 0, otherwise θ is not identifiable'],
    failureCases: ['A degenerate design (all x equal) gives det(XᵀX) = n·Σx² − (Σx)² = 0: the score never pins down θ — measured, any θ with β₀ + β₁ = 5 gives the same ℓ = −6.9445 on the flat likelihood', 'Non-Gaussian noise: the score is then NOT the least-squares normal equation'],
    derivesFrom: ['mle-score'],
    derivationIds: ['mle-linear-derivation'],
    connections: ['Normal equation', 'OLS', 'Linear regression'],
    whyWorks: 'For Gaussian noise, ∂ℓ/∂θ = Xᵀ(y − Xθ)/σ². Setting it to zero gives the normal equation XᵀXθ = Xᵀy — the MLE and OLS coincide exactly. On the seeded n=100 design the fitted θ̂ = (−0.467261, 1.590870) is the unique θ where the measured score is zero (8.4e−15, 8.5e−14), and perturbing θ by 0.1 increases RSS from 71.053 to 74.920 and lowers ℓ.',
  },
  {
    id: 'mle-ols',
    latex: '\\hat\\theta = (X^T X)^{-1} X^T y \\;\\;\\text{— the normal equation, solved exactly}',
    symbols: [
      { symbol: 'X^T X', meaning: 'the Gram matrix (2×2): [[n, Σx], [Σx, Σx²]]', dimensions: '2×2' },
      { symbol: '(X^T X)^{-1} X^T y', meaning: 'the closed-form OLS/MLE parameter vector', dimensions: '2×1' },
    ],
    assumptions: ['XᵀX is invertible (det > 0 — the design must have spread in x)', 'Gaussian errors (so OLS = MLE)'],
    failureCases: ['Near-singular XᵀX (x nearly constant) inverts to huge coefficients — the flat-likelihood failure in the limit det = 0', 'Numerically, inverting XᵀX is less stable than solving the system directly — the module uses the 2×2 adjugate form for exactness'],
    derivesFrom: ['mle-score-matrix'],
    derivationIds: ['mle-linear-derivation'],
    connections: ['OLS', 'Normal equation', 'Ridge regression (adds λI to XᵀX to fix singularity)'],
    whyWorks: 'With X = [1 x], XᵀX = [[n, Σx],[Σx, Σx²]] and det = n·Σx² − (Σx)². The module evaluates the adjugate inverse exactly: measured at n=100, XᵀX = [[100, 10.494344],[10.494344, 265.756110]] and Xᵀy = [−30.030970, 417.879909], giving θ̂ = (−0.467261, 1.590870). The MLE noise variance is σ̂² = RSS/n = 0.710527 (÷n), with the unbiased RSS/(n−2) = 0.725028 alongside — the linear analogue of the Gaussian ÷n bias.',
  },
  {
    id: 'mle-consistency',
    latex: '\\hat\\theta_n \\xrightarrow{p} \\theta \\;\\; \\text{as } n \\to \\infty \\;\\; (\\text{and } \\sqrt{n}(\\hat\\theta_n - \\theta) \\xrightarrow{d} \\mathcal{N}(0, I^{-1}))',
    symbols: [
      { symbol: '\\hat\\theta_n', meaning: 'the MLE from n samples', dimensions: 'parameter units' },
      { symbol: 'I', meaning: 'the Fisher information matrix — the asymptotic variance of the MLE', dimensions: 'per parameter unit²' },
    ],
    assumptions: ['Identifiability (distinct θ give distinct distributions)', 'Regularity: the true parameter is in the interior, the score is differentiable, etc.'],
    failureCases: ['Non-identifiable parameters violate the assumptions: the flat-likelihood design (det XᵀX = 0) has infinitely many MLEs and no convergence', 'The non-convex mixture likelihood converges to ONE of the two symmetric local modes (μ ≈ ±2), not to a unique truth'],
    derivesFrom: ['mle-score'],
    connections: ['Law of large numbers', 'Central limit theorem', 'Fisher information'],
    whyWorks: 'The MLE is consistent: it converges in probability to the true θ and is asymptotically normal with variance 1/I (efficiency). The seeded sweep demonstrates the convergence: p̂ error 0.1 → 0.04 → 0.007 over n = 10 → 100 → 1000; gaussian μ̂ error 0.940 → 0.071; linear slope error 0.095 → 0.012. The nllPerSample curve descends toward the true entropy H(0.7) = 0.610864 as n grows.',
  },
  {
    id: 'mle-invariance',
    latex: '\\hat\\theta \\text{ MLE } \\;\\Rightarrow\\; g(\\hat\\theta) \\text{ is the MLE of } g(\\theta) \\;\\text{ for any monotone } g',
    symbols: [
      { symbol: 'g', meaning: 'a monotone (more generally any) function of the parameter', dimensions: 'function' },
      { symbol: 'g(\\hat\\theta)', meaning: 'the MLE of the transformed parameter', dimensions: 'transformed units' },
    ],
    assumptions: ['g is injective enough to identify the transformed parameter', 'The MLE exists and is unique'],
    failureCases: ['For NON-monotone g the invariance can fail or give misleading labels (the mixture likelihood has two symmetric modes — g(θ̂) hides the ambiguity)', 'Invariance applies to the estimate, not to its expectation: E[g(θ̂)] ≠ g(E[θ̂]) in general'],
    derivesFrom: ['mle-likelihood'],
    connections: ['MLE properties', 'Biased estimators', 'Asymptotic normality'],
    whyWorks: 'If p̂ maximizes ℓ(p), then any monotone transform of p̂ maximizes the reparametrized ℓ — the MLE does not care about parameterization. The module leans on this: σ̂ (the MLE of the standard deviation) is √σ̂², and the fitted-curve label shows N(μ̂, σ̂²) directly. It is also why maximizing ℓ and L agree: ln is the monotone transform connecting them.',
  },
  {
    id: 'mle-cross-entropy',
    latex: '\\mathrm{CE}(\\hat\\theta) = -\\frac{1}{n}\\ell(\\hat\\theta) = -\\frac{1}{n}\\sum_{i=1}^{n} \\ln p(x_i \\mid \\hat\\theta)',
    symbols: [
      { symbol: '\\mathrm{CE}', meaning: 'the average negative log-likelihood — the empirical cross-entropy of the fitted model', dimensions: 'nats per sample' },
      { symbol: '-\\frac{1}{n}\\ell', meaning: 'negative log-likelihood normalized per sample', dimensions: 'nats per sample' },
    ],
    assumptions: ['The fitted model is used as a predictive distribution over the sample', 'Same n when comparing CE values'],
    failureCases: ['CE is a per-sample quantity: comparing CE across different n is invalid (each ℓ term is a log-density)', 'Minimizing CE on TRAINING data overfits — consistency is about the population limit, not the finite-sample fit'],
    derivesFrom: ['mle-loglik', 'mle-consistency'],
    connections: ['Cross-entropy loss', 'KL divergence', 'Logistic regression'],
    whyWorks: 'Maximizing ℓ is exactly minimizing average negative log-likelihood — the same objective deep-learning frameworks call cross-entropy. The loss-curve view plots nllPerSample = −ℓ(θ̂)/n: on the coin it descends 0.5004 → 0.6048 toward the true entropy H(p) = 0.610864, the theoretical floor. This is the "CE loss = negative log-likelihood" link the plan calls out.',
  },
];