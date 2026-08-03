// src/topics/cross-entropy-loss/mistakes.ts
import type { Mistake } from '../../engine/types';

export const ceMistakes: Mistake[] = [
  {
    id: 'dropping-the-log',
    pattern: 'Dropping the log: writing the CE loss as −Σ pᵢ·qᵢ instead of −Σ pᵢ·log qᵢ',
    example: '-\\sum_i p_i q_i \\;\\text{(wrong)} \\quad\\text{vs}\\quad -\\sum_i p_i \\log q_i \\;\\text{(correct)}',
    whyWrong: 'Without the log the "penalty" is bounded (≤ 1) and the gradient no longer matches the likelihood: minimizing −Σ p·q is NOT maximizing likelihood. The log is what turns the likelihood product into a sum (NLL = CE) and what makes confident-wrong predictions (q → 0) explode to ∞.',
    gateTrap: true,
    relatedConcept: 'cross-entropy',
  },
  {
    id: 'log-base',
    pattern: 'Mixing log base 2 and base e (bits vs nats — the units mistake)',
    example: 'H_\\text{bits}(p) = H_\\text{nats}(p) / \\ln 2, \\qquad 1\\ \\text{nat} \\approx 1.44\\ \\text{bits}',
    whyWrong: 'Base 2 measures information in BITS, base e in NATS — they differ by the constant 1/ln 2 ≈ 1.44. The optimizer argmax is unaffected (a constant factor scales the gradient), but reported values and NAT comparisons are off by ~44%. The MLE/CE identity requires natural log.',
    gateTrap: true,
    relatedConcept: 'entropy',
  },
  {
    id: 'h-vs-ce',
    pattern: 'Confusing H(p) with CE(p,q) — entropy of one distribution vs cross-entropy of two',
    example: 'CE(p,q) = H(p) \\;\\text{only when } q = p; \\qquad \\text{generally } CE(p,q) = H(p) + KL(p\\|q)',
    whyWrong: 'H(p) = −Σ pᵢ·log pᵢ uses the SAME distribution in the weight and the log; CE(p,q) weights by p but logs q. For p ≠ q they differ by KL(p‖q). GATE traps: "the loss is the entropy of the predictions" — it is the cross-entropy of true vs predicted.',
    gateTrap: true,
    relatedConcept: 'ce-h-kl',
  },
  {
    id: 'ce-is-symmetric',
    pattern: 'Assuming CE or KL is symmetric (like a distance metric)',
    example: 'CE(p,q) \\ne CE(q,p), \\qquad KL(p\\|q) \\ne KL(q\\|p) \\quad \\text{in general}',
    whyWrong: 'CE(p,q) weights the log qᵢ terms by pᵢ; swapping p and q changes both weights and logs, so the value changes (asymmetric check in the test cases: CE([0.8,0.2],[0.3,0.7]) = 1.035 vs CE([0.3,0.7],[0.8,0.2]) = 1.194). KL also violates symmetry and the triangle inequality — it is a divergence, not a metric.',
    gateTrap: true,
    relatedConcept: 'kl-divergence',
  },
  {
    id: 'zero-log-zero',
    pattern: 'Mishandling 0·log 0 (treating it as −∞ instead of the limit 0)',
    example: '0 \\cdot \\log 0 = 0 \\;\\text{by convention} \\quad\\leftarrow\\quad \\lim_{x \\to 0^+} x \\log x = 0',
    whyWrong: 'The term pᵢ·log pᵢ (or pᵢ·log qᵢ) with pᵢ = 0 is genuinely 0 by the limit — implementations that call log(0) = −∞ and multiply get NaN. BUT the same is NOT true when qᵢ = 0 and pᵢ > 0: that term is −∞ (log 0) and CE is truly infinite. The 0·log 0 = 0 convention only rescues zero-weight terms.',
    gateTrap: false,
    relatedConcept: 'cross-entropy',
  },
];
