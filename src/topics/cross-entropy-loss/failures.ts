// src/topics/cross-entropy-loss/failures.ts
import type { FailureDemo } from '../../engine/types';

export const ceFailureDemos: FailureDemo[] = [
  {
    id: 'ce-fail-log-zero',
    title: 'log(0) → −∞: a degenerate predicted distribution blows CE up to ∞',
    scenario: 'log-zero',
    // q0 = 1 makes q = [1, 0]: q₁ = 0 while the truth p₁ = 0.1 > 0 → the term
    // 0.1·log(0) = −∞ → CE = +∞. The sandbox reports a clean non-finite failure.
    // This param deliberately probes OUTSIDE the validated domain (validateParams
    // requires q0 ∈ (0,1)) to show WHY the (0,1) contract exists.
    params: { facet: 'cross-entropy', p0: 0.9, q0: 1 },
    narration: 'With q₀ = 1 the predicted distribution is q = [1, 0]: the model is CERTAIN class 1 never happens — but the truth puts p₁ = 0.1 on class 1. The term p₁·log q₁ = 0.1·log(0) = −∞, so CE = +∞ and the run terminates as a clean non-finite failure. This is the naive-implementation failure: log(0) is −∞, and cross-entropy is genuinely infinite when the predicted distribution assigns zero probability where the truth has mass.',
    whyItBreaks: 'CE(p,q) = −Σ pᵢ log qᵢ contains log qᵢ. If qᵢ = 0 for a class with pᵢ > 0, the term is −∞ and CE is truly +∞ — no clamp is mathematically honest here. Real code avoids this by clamping q (add ε), which is why validateParams keeps q strictly inside (0,1) and the sliders live in [0.05, 0.95] (full support).',
  },
  {
    id: 'ce-fail-confident-wrong',
    title: 'Confident-wrong predictions: the −log penalty explodes',
    scenario: 'confident-wrong',
    params: { facet: 'cross-entropy', p0: 0.9, q0: 0.05 },
    narration: 'The truth p = [0.9, 0.1] strongly favors class 0, but the model assigns it only q₀ = 0.05: the class-0 penalty −log(0.05) ≈ 3.0 dominates, giving CE = 0.9·ln(1/0.05) + 0.1·ln(1/0.95) ≈ 2.70 — more than 8× the entropy floor H(p) ≈ 0.33. Scrub the CE facet toward smaller q₀ and watch the red penalty curve and the CE marker both climb: overconfidence in the wrong class is penalized without bound.',
    whyItBreaks: 'The penalty for the class the truth favors is −log q, which → ∞ as q → 0. Unlike MSE (bounded by 1 for probabilities), CE has an unbounded penalty — that is by design (it is the NLL), and it is what makes confident-wrong classifiers pay heavily. The practical failure is numerical: for q below ~1e−16 the penalty saturates at double-precision ∞.',
  },
  {
    id: 'ce-fail-zero-log-zero',
    title: '0·log 0 is undefined — the ε-convention in action',
    scenario: 'zero-log-zero',
    params: { facet: 'cross-entropy', p0: 1, q0: 0.05 },
    narration: 'With p₀ = 1 the truth is degenerate: p = [1, 0]. The class-1 term p₁·log q₁ = 0·log(0.95) contributes exactly 0 by the 0·log 0 = 0 convention (limit x·ln x → 0 as x → 0⁺), so the run stays finite: CE = 1·ln(1/0.05) + 0 = 3.00. The convention rescues zero-WEIGHT terms. It does NOT rescue qᵢ = 0 with pᵢ > 0 — that is the log(0) failure of the previous demo. Implementations must evaluate 0·log 0 as 0, never as NaN.',
    whyItBreaks: 'Naive code calls Math.log(0) = −∞ before multiplying by pᵢ = 0, producing 0·(−∞) = NaN and poisoning the whole loss. The standard information-theory fix is the limit convention 0·log 0 = 0, applied only to the weight side: x·log(y) with x = 0 is 0 regardless of y. The failure here is implementation-level — floating-point arithmetic does not know the limit.',
  },
];
