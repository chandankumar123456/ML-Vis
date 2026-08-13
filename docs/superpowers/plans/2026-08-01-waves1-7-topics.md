# GATE ML Visualizer — Waves 1-7: Topic Module Builds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill all remaining syllabus topics (29 modules) to the same depth as the Wave 0 reference topics, using the proven `TopicModule` contract. Each wave is a batch of parallel topic builds; every topic gets the complete ecosystem (geometry, math, optimization, matrix, simulation, playground, formulas, derivations, GATE questions, comparison, failure, mistakes, timeline, testCases).

**Architecture:** Each topic is a self-registering module in `src/topics/<id>/module.ts` implementing the exact contract from `src/engine/types.ts` (see Wave 0 plan). Topics reuse registry components (`scatter-plot`, `loss-curve`, `matrix-animator`, `formula-explorer`, `timeline-view`, `mistake-view`, `question-player`, `decision-boundary`, etc.). New registry components needed by a topic are built in the shared `src/visualizers/` area (with their own tests) before or alongside the topic.

**Tech Stack:** Same as Wave 0. All topics must pass: `npm run lint && npm run test && npm run build`.

**Spec:** `docs/superpowers/specs/2026-08-01-gate-ml-visualizer-design.md`
**Wave 0 plan:** `docs/superpowers/plans/2026-08-01-wave0-platform.md`

---

## The Topic Build Pattern (used by EVERY task in this plan)

Every topic folder follows this exact structure — read Wave 0 Task 16/17 (`gradient-descent`, `simple-linear-regression`) as the reference implementation before starting:

```
src/topics/<topic-id>/
  module.ts          — TopicModule: metadata, layers, views, params, simulation,
                       formulas, derivations, questions, comparisons, failureDemos,
                       mistakes, testCases, validateParams, register()
  formulas.ts        — Formula[] (each: symbols, assumptions, derivesFrom, failureCases,
                       whyWorks, connections)
  derivations.ts     — Derivation[] (line-by-line animated steps with justifications)
  questions.ts       — Question[] (≥5 per topic: mix of gate-mcq, conceptual-mcq, nat,
                       matrix, visual; trapExplanations required on MCQ)
  mistakes.ts        — Mistake[] (≥3, gateTrap flags)
  comparisons.ts     — Comparison[] (vs similar algorithms, when ≥2 comparable topics exist)
  failures.ts        — FailureDemo[] (≥2 failure scenarios with params + whyItBreaks)
  testCases.ts       — TestCase[] (≥3: convergence, edge cases, numeric expectations)
  testCases.test.ts  — copy the pattern from Wave 0 (computeRun + expect checks)
  register()         — export function register() { registerTopic(module); }
```

### Quality bar (every topic, no exceptions)

1. **Simulation** must be pure + deterministic (same params → same snapshots). No `Math.random()` without a seeded PRNG (use `mulberry32` from `simple-linear-regression/module.ts`).
2. **testCases** must be mathematically correct — verified by `npx vitest run src/test/runTestCases.test.ts` (the centralized runner picks up any registered topic automatically).
3. **Narration** in every snapshot must explain what the algorithm is thinking ("The slope is negative. Move right."), not just report numbers.
4. **Questions**: all numerical answers must be verified by hand-calculating (put the calculation in `explanation`). Every wrong MCQ option needs a `trapExplanation`.
5. **Derivations**: every step must have a `justification` in plain English.
6. **Metadata** must be filled honestly (weightage/difficulty/priority from GATE DA syllabus norms).
7. **validateParams** must guard degenerate parameter combinations.
8. **Views** must come from the existing registry; if a new visualizer is needed, it goes in `src/visualizers/` with tests, registered via `registerView`, and only then referenced by the topic.

### New registry components this plan adds (build first, in dependency order)

| Component | When | Used by |
|---|---|---|
| `decision-boundary` | Wave 2 ✅ SHIPPED (`733376e`, hardened `e138f2b`) | logistic-regression, softmax-regression, svm-*, perceptron, knn, naive-bayes |
| `eigenviewer` | Wave 4 | pca, pca-svd, lda |
| `tree-builder` | Wave 5 | decision-trees, decision-trees-regression |
| `cluster-animator` | Wave 5 | kmeans, hierarchical-clustering |
| `dendrogram` | Wave 5 | hierarchical-clustering |
| `nn-inspector` | Wave 6 | neural-networks, backpropagation |
| `activation-view` | Wave 6 | neural-networks, backpropagation, logistic-regression |
| `distribution-view` | Wave 4 | naive-bayes, mle, lda |
| `confusion-explorer` | Wave 7 | classification-metrics, roc-auc |
| `roc-viewer` | Wave 7 | roc-auc |
| `curve-comparator` | Wave 7 | bias-variance, overfitting-underfitting, cross-validation |
| `derivation-player` | Wave 1 (small) ✅ SHIPPED (`ab985b5`) | all topics with derivations |
| `explain-step` panel | Wave 1 (small) ✅ SHIPPED (`00d0b04`) | all topics (renders snapshot.explanation) |

Each new component follows the ViewProps contract: `{ run?, snapshot?, params, subscribe?, compact?, topic? }` (the `topic?` field was added in `ab985b5` for topic-context lookups like formulaRef/lossMetricKey).

---

## Wave 1 — Regression cluster (4 topics) ✅ COMPLETE

All four topics + both small registry items shipped (`64f381b`/`77cb641`, `cad9487`, `0887bcb`, `122554e`, `00d0b04`, `e19793e`, `5121b62`); suite 146/146, lint, build, e2e 3/3. Platform gains: `TopicModule.lossMetricKey2` + LossCurve dual-series/bar mode (`5121b62`). New topics self-register via `import.meta.glob` — no main.tsx edits. See per-task drift/SHIPPED notes below.

### Task 1: multiple-linear-regression

**Files:** `src/topics/multiple-linear-regression/{module,formulas,derivations,questions,mistakes,comparisons,failures,testCases}.ts`

- [ ] **Step 1: Write testCases first**

TestCases (≥3):
1. `normal equation on 2 features` — synthetic data y = 3x₁ − 2x₂ + 1 + tiny noise → fitted coefficients within 0.05 of truth.
2. `GD converges to same optimum` — GD with small lr + 2000 epochs ≈ normal equation result.
3. `fails when features collinear` — x₂ = 2x₁ → XᵀX singular → validateParams flags it OR telemetry fails cleanly (test the validateParams path).
4. `predicts correctly on test point` — final model's prediction within 0.1 of true.

Math content: design matrix X (n×d), normal equation θ = (XᵀX)⁻¹Xᵀy, MSE in matrix form, cost surface for 2 weights (3D-perspective via loss-landscape later; use loss-curve + matrix-animator now). Derivation: normal equation from gradient = 0 (matrix calculus). Formula explorer: hypothesis, MSE, normal equation, R².

Simulation: `useNormalEquation` toggle + `nFeatures` (1..3) slider, `noise`, `seed`; steps = epochs for GD mode, single shot for normal equation. Visuals: `matrix-animator` commands for X, XᵀX, Xᵀy, θ (show dimension compatibility: (d×n)(n×d) etc.), scatter for 1-feature case, residual lines.

Questions (≥5): mix of NAT (compute θ given X,y — the slr-003 pattern extended to 2 features), conceptual (why XᵀX must be invertible), matrix (dimension matching), visual (fit quality vs noise), trap (forgetting bias column).

Mistakes: omitting bias column; inverting wrong matrix; dimension mismatch in XᵀX; assuming n>d unnecessary.

Comparisons: vs simple-linear-regression, vs ridge (when d large), vs polynomial (nonlinear).

Failures: multicollinearity (rank-deficient X), n < d (underdetermined — infinite solutions).

- [ ] **Step 2: Run testCases red → green**

Run: `npx vitest run src/topics/multiple-linear-regression`
Expected: FAIL (module missing) → PASS after writing module.

- [ ] **Step 3: Write module + content files, register**

Include `validateParams`: if collinear → return ['Features are collinear — XᵀX is singular'].

- [ ] **Step 4: Full verification**

Run: `npm run lint && npm run test && npm run build`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/topics/multiple-linear-regression
git commit -m "feat: multiple-linear-regression topic (full ecosystem)"
```

> **Plan drift (Task 1):** the `matrix-animator` X → XᵀX → Xᵀy → θ story is
> emitted in normal-equation mode only. Rationale: GD mode runs up to 2000
> snapshots and per-step matrix payloads (X alone is n×(d+1) cells) would blow
> up memory/scrub latency; X, XᵀX, Xᵀy are constant across epochs anyway, so
> only θ changes. NE mode is the default and single-shot, so the full matrix
> story is always one click away. Platform note (deferred): MatrixAnimator
> could render a "rows×cols" caption under each matrix id.
>
> SHIPPED: `64f381b` (feat, 9 files, 896 lines) + `77cb641` (review nits:
> GD-vs-NE test now uses noise 0.1 on a shared seed — non-degenerate target,
> both methods fit identical data so agreement is exact; matrix-mode comment
> made self-contained). Reviews: spec-compliance APPROVE WITH NITS (13/13
> PASS; drift note above; validateParams path exercised explicitly in
> testCases.test.ts); quality APPROVE (zero Critical/Medium; toStandard/
> fromStandard round-trip algebra hand-verified correct — w̃ⱼ=wⱼσⱼ,
> b̃=b+Σwⱼμⱼ; MLR-001 NAT answer w₁=2 hand-verified via XᵀX·[2,1,1]=Xᵀy;
> contraction 0.99/epoch → machine precision at 2000 epochs; correlation
> threshold 0.9999 unreachable by chance in 100k MC trials). Gates after
> nits: 90/90 vitest, lint clean, build ✓, e2e 3/3.
>
> Implementation notes: (1) topic self-registers via `import.meta.glob` —
> no main.tsx edit needed. (2) GD is feature-standardized (toStandard/
> fromStandard) with bias-last design matrix, matching SLR conventions.
> (3) `collinear` test param is intentionally not in the UI schema.
> (4) dev-history: the round-trip once used Σw̃ⱼμⱼ (bias leaked ~0.024/epoch
> → GD diverged to b≈−57.5) — the correct pair is Σwⱼμⱼ, now guarded by the
> green GD test.

### Task 2: polynomial-regression

**Files:** `src/topics/polynomial-regression/{...}.ts`

TestCases:
1. `degree 1 on quadratic data underfits` — high MSE, line ≈ linear.
2. `degree 2 recovers quadratic` — coefficients ≈ truth (w₂ within 0.05).
3. `degree 15 overfits` — train MSE tiny, but coefficient magnitudes explode (|w| > 10) and/or test MSE high (use held-out split).
4. `bias-variance demo` — train vs test error curve as degree slider moves (error curves computed from 5 seeds averaged).

Math content: basis expansion φ(x) = [1, x, x², ..., x^d], design matrix with polynomial features, normal equation on transformed features, train/test split, bias-variance decomposition intro.

Simulation: `degree` slider (1..15), `noise`, `nTrain`, `nTest`, `seed`, `fitOn` (train/all). Visuals: scatter of train/test points (different colors), polynomial curve animated as degree changes, train vs test loss bars. Matrix view: the Vandermonde design matrix Φ — animate its growth with degree.

Questions: conceptual (why high-degree overfits), NAT (predict y for given x with fitted degree-2 polynomial), visual (identify underfit/overfit curve), trap (degree ↑ always ↓ training error but ↑ test error).

Mistakes: thinking more features always better; fitting on full data without validation; ignoring feature scaling for high degree.

Comparisons: vs simple-linear-regression, vs ridge (ridge fixes polynomial overfitting), vs lasso.

Failures: degree 30 → numerical instability of normal equation (huge condition number); data at extremes → Runge's phenomenon.

> **Plan drift (Task 2):** (1) features are modeled on a NORMALIZED basis
> u = x/x_max (x_max = 3) — degree-d in u ⇔ degree-d in x (linear
> reparameterization, model class unchanged), but raw powers of x∈[−3,3]
> reach 3¹⁵≈1.4×10⁷ and make ΦᵀΦ unusable; the conditioning story is real
> (normalized κ ≈ 4.3e12 at d=15 vs raw 9.4e16). (2) `matInverse` pivot
> threshold is 1e-14 (MLR uses 1e-12) — empirically tuned: d=15 solves 50/50
> seeds at 1e-14 while d=30 fails cleanly (19/20 seeds; the shipped demo uses
> the deterministic failing seed 42). (3) Bias-FIRST φ=[1,u,u²,…] matches the
> plan's own expansion (MLR/SLR bias-last is for feature matrices). (4) Runge
> demo is least-squares-with-noise (regression-domain analogue, honestly
> labeled "Runge-type"). (5) "train vs test loss bars" delivered via the
> LossCurve dual-series + bar-mode platform extension (commit `5121b62` —
> lossMetricKey2 'testMse'; single-shot topics render grouped bars).
>
> SHIPPED: `cad9487` (feat, 9 files, 1019 lines) + `e19793e` (nits: honest
> κ wording in the degree-30 failure demo). Reviews: spec APPROVE WITH NITS
> 13/13 (5-seed U-curve verified: train ↓ strictly monotone across
> [1,2,3,5,8,12,15], test U with min at d=2 measured 0.183/0.094/0.110/1.20/
> 16482; NATs poly-001=6, poly-002=1.625 hand-verified; condition numbers
> empirically re-verified); quality APPROVE zero issues (basis consistency
> x↔u verified everywhere; no train/test leakage; d=15 explosion 28,068 real).
> TDD caught a real bug: `mseOf('train')` fell through to ALL data — fixed
> with explicit `subsetOf`. Gates: 146/146 full suite, lint, build, e2e 3/3.

### Task 3: ridge-regression

**Files:** `src/topics/ridge-regression/{...}.ts`

TestCases:
1. `closed form with λ=0 equals OLS` — ridge(0) ≈ normal equation.
2. `increasing λ shrinks coefficients` — ‖θ(λ₂)‖ < ‖θ(λ₁)‖ for λ₂ > λ₁.
3. `ridge fixes collinearity` — collinear features: OLS explodes (‖θ‖ huge), ridge(λ=1) stable finite coefficients.
4. `bias-variance: λ sweet spot` — as λ grows: train error ↑, test error ↓ then ↑ (compute from averaged seeds).

Math content: J = MSE + λ‖θ‖₂², closed form θ = (XᵀX + λI)⁻¹Xᵀy, ridge path (coefficient shrinkage curve vs λ), bias-variance tradeoff, equivalence to constrained optimization ‖θ‖₂ ≤ t.

Simulation: `lambda` log-slider (0..10), `collinear` toggle, `nFeatures`; animate the L2 constraint circle + unconstrained optimum (geometry: ellipse touching circle). Matrix: (XᵀX + λI) — animate adding λ to diagonal. Loss curve: train/test vs λ.

Questions: NAT (compute ridge solution for 2×2 given λ — the classic GATE numeric), conceptual (why ridge doesn't zero coefficients), visual (shrinkage path), trap (λ=0 vs OLS; ridge vs lasso sparsity).

Mistakes: confusing ridge with lasso (sparsity); forgetting +λI makes invertible; thinking λ shrinks only large coefficients equally (it shrinks all, proportionally).

Comparisons: vs lasso (geometry: circle vs diamond), vs OLS (bias-variance), vs polynomial (ridge as fix for overfitting).

Failures: λ→∞ → θ→0 underfit; unscaled features → uneven shrinkage.

> **Plan drift (Task 3):** (1) the run is a λ-SWEEP — one snapshot per λ on
> [0, 0.5, …, slider], last snapshot exactly the slider λ — not an
> epoch-based simulation (ridge is closed-form; scrubbing/playing IS the
> shrinkage-path animation; loss-curve plots train/test vs λ with ≥2 points).
> (2) TestCase 4's "test error ↓ then ↑" is PHYSICALLY UNREACHABLE under the
> plan's constraints (d≤3, λ≤10, well-conditioned i.i.d. data): design
> eigenvalues μ≈125 with σ²d/n≈0.45 → variance reduction is negligible while
> bias² grows, so test error is monotone non-decreasing; the honest test
> asserts train ↑, large-λ underfit (STRONG signal at λ=10, per review
> NIT-1), moderate-λ ≈ free. The dip regime is collinear/high-d — covered by
> the near-collinear case. (3) `collinearJitter` test-only param (1e-5) so
> the OLS-explodes assertion is finite (‖θ‖≈2389 vs ridge 1.05, ratio >100);
> exact-collinear stays the clean-failure path. (4) penalty applies to ALL of
> θ including bias (plan's literal J = MSE + λ‖θ‖₂²; non-standard vs Hastie,
> documented in formulas). (5) `collinear` is test-only, not a UI toggle
> (mirrors MLR). Loss-curve train/test via `lossMetricKey:'testMse'` +
> `lossMetricKey2:'trainMse'` (platform extension `5121b62`).
>
> SHIPPED: `0887bcb` (feat, 9 files, 1077 lines) + `e19793e` (T4 λ=10 tail
> assertion — the earlier λ=5-only net ≈ +0.005 sat at the MC noise floor).
> Reviews: spec APPROVE WITH NITS 13/13; quality APPROVE zero issues
> (NATs ridge-001 w=4/3, ridge-008 b=1 hand-verified via (XᵀX+I)⁻¹; jitter
> margin actually 10⁷× not 670×; "λ=0 bit-identical to MLR" verified — shared
> TRUE_W/TRUE_B/PRNG). λ-sweep ≤21 snapshots, matrices ≤4×4. Gates: 146/146,
> lint, build, e2e 3/3.

### Task 4: lasso-regression

**Files:** `src/topics/lasso-regression/{...}.ts`

TestCases:
1. `lasso with λ=0 equals OLS` (verify via coordinate descent at tiny λ ≈ OLS on well-conditioned data).
2. `lasso zeroes small coefficients` — coefficient < threshold becomes exactly 0 for sufficient λ.
3. `ridge never zeroes, lasso zeroes` — same data, ridge keeps |w| > 0, lasso hits exactly 0.
4. `coordinate descent converges` — monotone decrease of objective (track objective per sweep).

Math content: J = MSE + λ‖θ‖₁, soft-thresholding operator S(z, λ) = sign(z)·max(|z|−λ, 0), coordinate descent update, subgradient at 0 (why lasso zeroes), comparison of L1 vs L2 constraint geometry.

Simulation: `lambda` slider, `nFeatures` (3-6), coordinate descent steps animated: one coefficient updated per step, soft-threshold visualized as shrink-toward-zero; feature selection count as metric. Visuals: coefficient paths (each feature a line vs λ), diamond constraint + ellipse (geometry view).

Questions: NAT (apply soft-threshold: given z and λ compute new coefficient), conceptual (why lasso selects features, ridge doesn't), visual (coefficient path plot reading), trap (lasso ≠ always better; correlated features → arbitrary selection).

Mistakes: thinking lasso shrinks uniformly (it soft-thresholds); confusing λ direction; thinking ridge also produces exact zeros.

Comparisons: vs ridge (the diamond/circle contrast — the classic GATE figure), vs OLS, vs polynomial.

Failures: correlated features → lasso picks arbitrarily; λ too large → all zero (underfit).

> **Plan drift (Task 4):** (1) step model = ONE COORDINATE UPDATE per step
> (plan's explicit suggestion); sweep = d+1 steps, convergence checked only
> at sweep boundaries. (2) coefficient-path/geometry "views" substituted via
> available channels — loss-curve plots the CD OBJECTIVE (`lossMetricKey:
> 'objective'`), per-step soft-threshold narration with exact-zero events, a
> visual question reading a path plot, geometry via formulas/derivations/
> comparisons (no path/geometry registry view exists; quality-bar mandates
> registry-only views). (3) ridge comparison is self-contained inside
> testCases.test.ts (local ridge closed form — the ridge topic was being
> written in parallel). (4) λ slider min 0 / step 0.25 / default 0.5
> (ON-STEP); max 10 makes the all-zero failure UI-reachable (data corr ≈ 8.7).
> (5) case-level fit assertion uses r2 > 0.9 (mse < 0.5 was unachievable:
> measured MSE ≈ 1.28 at λ=0.5 from honest shrinkage bias); the plan's
> monotone-decrease-per-sweep is tested on the trajectory. (6) matrices
> emitted only on the converged snapshot (MLR precedent).
>
> SHIPPED: `122554e` (feat, 9 files, 1100 lines) + `e19793e` (sweeps fallback
> 300 → 200 aligned with the schema default). Reviews: spec APPROVE WITH NITS
> 13/13 (exact-zero verified runtime w4=0; UI-reachable all-zero at λ=10;
> last snapshot converged for all test cases); quality APPROVE zero issues —
> CD verified in depth: bias update excludes current bias (the oscillator-bug
> class `b̃←ȳ−b̃` documented + avoided), soft-threshold returns literal IEEE-754
> zero (exact zeros provable), monotonicity guaranteed by exact coordinate
> minimizers, toStandard/fromStandard matches MLR's corrected pair; NATs
> lasso-001 S(3.2,1)=2.2, lasso-002 S(0.8,1)=0 hand-verified. Gates: 146/146,
> lint, build, e2e 3/3.

---

## Wave 2 — Classification cluster (5 topics) ✅ COMPLETE

All five topics + decision-boundary shipped (`733376e`, `0c358b0`, `d729d9d`, `3aad4b0`, `645cfa4`, `d1492bc`); review cycle closed (12 reviews, fixes `e138f2b`/`880eb5e`/`d5c69a7`, closure re-reviews all green); suite 273/273 (28 files), lint, build, e2e 3/3. Platform gains: classifier registry contract (`registerClassifier`/`getClassifier`, viewRegistry.ts); ScatterPlot `onDragPoint` + `circle` VisualCommand; `src/visualizers/bounds.ts` (shared boundsOfVisuals incl. circles + `safeClassify`). Note: the softmax task was agent-cancelled mid-flight, but all 9 files landed, passed 19/19 tests, and were reviewed to APPROVE — no work was lost. See per-task drift/SHIPPED notes below.

### Task 5: New registry component — decision-boundary (+ derivation-player, explain-step)

**Files:** `src/visualizers/DecisionBoundary.tsx`, `src/visualizers/DerivationPlayer.tsx`, `src/visualizers/ExplainStep.tsx`, tests for each.

- DecisionBoundary: renders 2D grid of class regions (per-pixel color from a `classify(x,y)` callback — efficient via offscreen 50×50 grid + imageData scaling), decision line, margin lines, support vectors (highlighted), animated per snapshot. Props: `{ snapshot?, params, classify?: (x: number, y: number) => number }` — classify comes from the topic's module via a registry lookup on `params.modelId` OR the topic passes a global classify function via `subscribe`-style context. Simplest contract: topics register a `classifier` alongside views:

```ts
// registry addition: viewRegistry.setClassifier(viewId, fn) — optional, per topic
export function registerClassifier(id: string, fn: (x: number, y: number, params: Params) => number): void
export function getClassifier(id: string): ((x: number, y: number, params: Params) => number) | undefined
```

Update `src/registry/viewRegistry.ts` (add classifier map + tests). DecisionBoundary consumes `getClassifier(topicId)` when rendered.

- DerivationPlayer: takes `Derivation[]` + step-by-step UI (next/prev per step, Latex rendering, justification panel, progress bar). Registered as `derivation-player`.
- ExplainStep: renders `snapshot.explanation` (changed[], why, formulaRef, dependsOn, gateConcepts) as an expandable panel. Registered as `explain-step`.

- [ ] **Steps:** TDD each component (test render + classifier registration), then:

Run: `npm run lint && npm run test && npm run build`
Commit: `feat: decision-boundary, derivation-player, explain-step components`

> **Plan drift (Task 5):** derivation-player + explain-step were already shipped in Wave 1 (registry table: `ab985b5`, `00d0b04`); this wave shipped only decision-boundary. The classifier contract landed exactly as planned — `registerClassifier(id, fn: (x, y, params) => number)` / `getClassifier(id)` in viewRegistry.ts (Map-backed; unknown id → undefined, never throws; re-register overwrites; return = CLASS INDEX 0/1, not probability). DecisionBoundary: 50×50 grid of cell-center samples → ImageData → drawImage upscale (`data-decision-grid="50"`, colors via `--cat1/--cat2` CSS vars), merges `{...params, ...snapshot.algorithm}` ONCE per snapshot so classifiers read current-step weights, bounds follow ScatterPlot's convention with [−7,7]² fallback. Review-mandated hardening (`e138f2b`): `safeClassify` wraps every classifier call site (throwing/undefined/NaN classifiers degrade to class 0 — no canvas crash), `imageSmoothingEnabled=false` bracket for crisp upscale, grid memoized per (snapshot, classifier, effParams) so scrub/re-render reuses 2500 classifications, bounds include circle extents (knn rings), canvas aria-label; pure helpers extracted to `src/visualizers/bounds.ts` (+9 tests).
> SHIPPED: `733376e` (feat, 6 files, 518 insertions) → reviews: spec 11/11 APPROVE WITH NITS; quality REQUEST CHANGES (Critical: uncaught classifier exceptions crash the render; Medium: blurry upscale, per-call params rebuild breaking identity-keyed caches, knn per-call point regeneration; Low: circle-less boundsOfVisuals, missing aria-label) → fixes `e138f2b` → quality re-review: ALL 7 findings CLOSED, no regressions. Hardening also required knn-side points memoization → `880eb5e`.

### Task 6: logistic-regression

**Files:** `src/topics/logistic-regression/{...}.ts`

TestCases:
1. `decision boundary separates linearly separable data` — trained model classifies all train points correctly (accuracy = 1 on separable synthetic).
2. `sigmoid maps to (0,1)` — σ(0) = 0.5, σ(10) ≈ 1, σ(−10) ≈ 0.
3. `cross-entropy loss decreases over epochs` — monotone non-increasing on well-conditioned data.
4. `probability calibration` — predicted p ≈ empirical fraction for points near boundary (weak check: p > 0.5 ⟺ prediction).

Math content: sigmoid σ(z) = 1/(1+e⁻ᶻ), log-odds z = w·x + b, likelihood, cross-entropy loss L = −Σ[y·log(ŷ) + (1−y)·log(1−ŷ)], gradient ∂L/∂w = Σ(ŷ−y)x, decision boundary (linear where p = 0.5), MLE connection.

Simulation: 2D synthetic data (two Gaussian clusters or separable grid), `lr`, `epochs`, `init`, `margin` (cluster separation). Step = one GD epoch; animate: sigmoid curve with current weights (activation-view), decision boundary moving (decision-boundary), loss curve, probability heat coloring of points (blue→red intensity = p). Matrix view: w as vector, x as vector, dot product animation.

Derivations: (1) CE loss from MLE; (2) ∂L/∂w via chain rule (the famous (ŷ−y)x result).

Questions (≥6): NAT (compute σ(z) for given z), numeric (given w, x, predict class + confidence), conceptual (why not MSE for classification), matrix (gradient as vector sum), visual (boundary position vs margin), trap (sigmoid saturation).

Mistakes: using MSE loss (non-convex for sigmoid); forgetting the (ŷ−y) factor in gradient; predicting class by p ≥ 0.5 only (threshold is a choice); assuming boundary always separable.

Comparisons: vs perceptron (loss: CE vs hinge-ish zero), vs SVM (margin maximizer), vs naive bayes (generative vs discriminative), vs linear regression (why not use it for classification).

Failures: class imbalance → boundary pushed to majority; non-linear data → linear boundary fails; saturated sigmoid → vanishing gradient.

> **Plan drift (Task 6):** (1) step model = one GD epoch over a fixed 2D dataset (two Gaussian clusters, configurable margin, seeded); the classifier contract feeds decision-boundary — each snapshot writes current-step weights into `snapshot.algorithm` (`w1/w2/b` in ORIGINAL space via toStandard/fromStandard, bias unpenalized) and the registered `logistic-regression` classifier reads them (memoized `train()` fallback keyed by params when algorithm weights are absent). (2) activation-view is Wave 6 — NOT registered; the sigmoid response is shown via scatter-plot probability heat-coloring + matrix-animator log-odds story (z = w·x + b, σ(z)) instead (registry-only views rule). (3) loss = stable softplus cross-entropy `y·softplus(−z)+(1−y)·softplus(z)` — differentiates to `(1/n)Σ(ŷ−y)x` (finite-diff verified), no log(0) at saturation; CE monotone non-increasing at lr 0.3 (300 epochs). (4) `lr` slider max 1.0; validateParams warns on lr > 1. (5) lossMetricKey 'ce'.
> SHIPPED: `0c358b0` (feat, 9 files, 1077 lines) → reviews: spec 14/14 APPROVE WITH NITS (3 cosmetic); quality APPROVE (gradient identity verified, boundary in original space, σ(10)≈1−4.5e-5, monotone CE, jitter margin 10⁷×). Tests 17/17.

### Task 7: cross-entropy-loss (+ MLE — same module, two facets)

**Files:** `src/topics/cross-entropy-loss/{...}.ts`

TestCases:
1. `CE ≥ 0 and CE = 0 when p = q` (perfect prediction).
2. `CE(p, q) asymmetric` — CE(p,q) ≠ CE(q,p) for p ≠ q (information theory).
3. `CE relates to KL + H` — CE(p,q) = H(p) + KL(p‖q) numeric check.
4. `MLE maximizes likelihood ⟺ minimizes CE` — numeric equivalence on small dataset.

Math content: information H(p) = −Σp·log p, cross-entropy, KL divergence, MLE derivation (product → log → sum), Bernoulli/Binomial likelihood, negative log likelihood = CE for classification. Relation chain: Probability → MLE → CE → Logistic Regression (hidden-gate-link).

Simulation: distributions p, q adjustable (sliders per class probability); animate the log curve, the -log penalty for confident wrong predictions; show H, KL, CE numerically; for MLE: dataset of coin flips, likelihood curve over θ, log-likelihood, MLE at argmax.

Questions: conceptual (why log-likelihood), NAT (compute CE for given p,q), matrix (softmax + CE gradient), visual (loss surface for two classes), trap (CE vs squared error).

Mistakes: dropping the log; using log base 2 vs e (bits vs nats — units); confusing H(p) with CE(p,q).

Comparisons: vs MSE, vs hinge loss (SVM), vs 0-1 loss (non-differentiable).

Failures: p_i = 0 with q_i = 0 → 0·log 0 undefined (use convention/ε); log(0) → −∞ in naive implementation.

> **Plan drift (Task 7):** (1) `facet` param `('cross-entropy'|'mle')` — one module, two simulations; K=2 only (Bernoulli; binomial coefficients omitted — θ-independent, so the NLL=CE identity holds; no multinomial). (2) cross-entropy facet: p₀/q₀ sliders [0.05, 0.95] step 0.05; q₀-sweep ≤19 snapshots, last = slider; metrics cePQ/hP/klPQ; lossMetricKey 'cePQ' + lossMetricKey2 'hP' (dual-series loss-curve: CE vs entropy H(p)). (3) MLE facet: 21-point θ grid 0.02→0.98; likelihood/log-likelihood/MLE curves; MLE ⟺ min-CE verified (hand-check h=12, n=20, θ=0.6 → NLL 0.6020). (4) xlogy convention: `0·log0 = 0` (x≤0 guard); x>0, y=0 → unclamped −∞ (documented failure demo). (5) deterministic grids — no seed param. NATs verified live: CE(p‖q)=1.194, KL=0.223, θ̂=0.7, KL=0.092.
> SHIPPED: `d729d9d` (feat, 9 files, 13 tests) → reviews: spec 14/14 APPROVE WITH NITS (3 informational); quality APPROVE (all 4 NATs, NLL=CE derivation hand-checked, xlogy audit).

### Task 8: softmax-regression (multiclass logistic)

**Files:** `src/topics/softmax-regression/{...}.ts`

TestCases:
1. `softmax outputs sum to 1` — Σ softmax(z) = 1 for random z.
2. `softmax is invariant to constant shift` — softmax(z) = softmax(z + c).
3. `3-class classification converges` — accuracy → 1 on separable 3-cluster data.
4. `categorical CE gradient` — gradient formula numeric check (∂L/∂w_k = Σ(ŷ_k − 1{y=k})x).

Math content: softmax(z)_k = e^z_k / Σ e^z_j, categorical CE loss, log-sum-exp stability, gradient with indicator, relation to sigmoid (K=2 case), one-hot encoding.

Simulation: 3 Gaussian clusters, `lr`, `epochs`; animate: probability vector per point (distribution-view), decision regions (decision-boundary with 3 colors), loss curve, weight matrix W (matrix-animator: rows = classes), per-epoch misclassification count metric.

Questions: NAT (compute softmax for given z vector — the classic), conceptual (why exp: monotone + positivity), matrix (W·x + b dimensions), visual (decision regions), trap (softmax ≠ stable without max-subtraction).

Mistakes: applying softmax without max-shift numerically; confusing sigmoid (2-class) vs softmax; forgetting sum over classes in gradient.

Comparisons: vs logistic (K=2 equivalence), vs one-vs-rest, vs neural network output layer.

Failures: class imbalance; logits huge → exp overflow (use log-sum-exp); correlated classes.

> **Plan drift (Task 8):** (1) parameterization W(K×d) + per-class bias b(K,) (bias-last, rows = classes — matrix-animator shows W animating per epoch); algorithm keys `w11..w32, b1..b3` written per step in ORIGINAL space; classifier = argmax over σ(Wx+b) → class index, with memoized `trainFinal` fallback. (2) K=3 fixed, 3 Gaussian clusters, lr/epochs; per-epoch misclassification count + accuracy metrics (lossMetricKey 'ce' + lossMetricKey2 'accuracy'). (3) stable max-shift softmax (log-sum-exp style); gradient `(1/n)Σ(ŷ_k − 1{y=k})x` finite-diff verified (h=1e-5, tol 1e-6); K=2 ⟺ σ(z₁−z₀) algebra verified. (4) distribution-view is Wave 4 — NOT registered; per-point probability mass shown via scatter-plot coloring + the loss/accuracy curves. NATs verified: softmax-001 ŷ₃ = e³/(e¹+e²+e³) = 0.66524; softmax-002 ŷ₂ = e¹/(1+e¹+1) = 0.57612.
> SHIPPED: `3aad4b0` (feat, 9 files, 19 tests) → reviews: spec 15/15 APPROVE WITH NITS (2 conventions); quality APPROVE (gradient, FD, shift-invariance tol 1e-10, both NATs). Task was agent-cancelled mid-flight but all files landed and were verified — no work lost.

### Task 9: knn

**Files:** `src/topics/knn/{...}.ts`

TestCases:
1. `k=1 reproduces training labels` — 1-NN classifies train set perfectly (zero train error).
2. `k increases smooths boundary` — decision boundary region count decreases with k (empirical check).
3. `majority vote with tie-breaking` — deterministic tie-break (nearest of tied).
4. `distance metric changes regions` — L1 vs L2 produce different classifications on a test point (assert at least one differs on crafted data).

Math content: distance metrics (Euclidean, Manhattan, Minkowski), majority vote, k choice, curse of dimensionality (volume concentration), time complexity O(n·d) per query, no training phase (lazy learner).

Simulation: `k` slider (1..20), `metric` select, points draggable (click to add, drag to move — needs pointer interaction on ScatterPlot: extend ScatterPlot with optional `onDragPoint` prop in Wave 2), decision regions (decision-boundary via nearest-neighbor classifier), distance rings around a query point (knn-animator component or decision-boundary + circles), error vs k curve.

Questions: NAT (compute distance + majority class for given k), conceptual (why k=1 overfits), visual (boundary with k=1 vs k=15), trap (tie-breaking; scaling matters — feature normalization), indirect (curse of dimensionality link).

Mistakes: no feature scaling (dominating feature); odd k only for 2 classes (tie-break rules for 3+); thinking KNN has a training phase.

Comparisons: vs naive bayes (lazy vs eager... actually NB has params; both simple), vs decision tree (boundary shape), vs SVM (boundary smoothness).

Failures: high dimensions (curse), class imbalance (majority swamps), noisy features, large n (slow inference).

> **Plan drift (Task 9):** (1) step model = k-SWEEP mirroring ridge's λ-sweep — one snapshot per k on [1..params.k], last = slider k; scrubbing IS the boundary-smoothing + ring-expansion animation (lazy learner, no epochs). (2) metrics: lossMetricKey 'error' = honest LEAVE-ONE-OUT error (k=1 is NOT zero: 0.417 on default seed), lossMetricKey2 'trainError' = self-classification (k=1 → exactly 0 — the "k=1 overfits" signature). MEASURED LOO curve is NON-monotone: bottoms 0.208 at k∈{7,9,15,16,18}, oscillates [0.21, 0.42], creeps back up at k=19–20; curves coincide with train error at k∈{15,16,18}; region count 51→35 (k=1 vs 15) is a TREND, not strict monotone — all narrative claims reworded to match the measured curve (fixes `880eb5e`, `d5c69a7`). (3) ScatterPlot extended with optional `onDragPoint` (additive; 14px pickup; screenToWorld) + `circle` VisualCommand (one distance ring per neighbor; `type` is `string` — pre-existing non-discriminated design, structurally valid). (4) `points` JSON param (validated: even count, equal class counts, in [−5,5]², ≥2 points, k ≤ size); scaling deliberately NOT applied — z-scoring lives in mistakes. (5) classifier path memoizes the point set (`getPoints`, key = seed|nPerClass|margin or points-JSON, bounded 64) — DecisionBoundary's 2500 calls/snapshot hit one cached array.
> SHIPPED: `645cfa4` (feat, 10 files incl. ScatterPlot) + fixes `880eb5e` (points memo, narrative) + `d5c69a7` (residual narrative: honest trend claims, metric-diff changed list) → reviews: spec 13/15 APPROVE WITH NITS — 2 FAILs: item 12 test-coverage claim (report said "7 tests in one it"; code has 6 its / 7 assertions — coverage note added), item 14 narrative monotonicity (claims contradicted the measured curve) → re-review: item 12 PASS, item 14 empirical claims 100% reproduced + residuals fixed; quality APPROVE (tie-break nearest-of-tied, L1/L2 flip, LOO honest 0.417, onDragPoint additive, circles world→pixel, region counts). Tests 15/15.

### Task 10: naive-bayes

**Files:** `src/topics/naive-bayes/{...}.ts`

TestCases:
1. `posterior normalization` — Σ posterior over classes = 1.
2. `independence assumption changes results` — compute posterior with and without independence on correlated features → different (demonstrate the assumption's effect).
3. `zero-probability handling` — with Laplace smoothing, unseen feature value gets ε not 0; without smoothing, posterior = 0.
4. `Gaussian NB fits and predicts` — synthetic Gaussian per class → posterior argmax matches cluster.

Math content: Bayes theorem, prior P(C), likelihood P(x|C) = Π P(x_i|C) (naive assumption), posterior, Laplace smoothing, Gaussian NB (per-class mean/variance), log-space computation for stability.

Simulation: `nClasses` (2-3), `smoothing` (ε or 0), features with adjustable correlation (to show independence violation), distribution-view showing per-feature class likelihoods, animation of posterior calculation (probability flow), decision regions via NB classifier.

Questions: NAT (compute posterior given table of likelihoods — the classic GATE table problem), conceptual (why naive is naive), trap (zero prob without smoothing; forgetting prior), matrix (n/a — but feature×class table), visual (boundary vs true generative).

Mistakes: ignoring prior; multiplying raw (unsmoothed) zero probabilities; assuming independence holds; log-space not used → underflow.

Comparisons: vs logistic (generative vs discriminative), vs knn (parametric vs instance), vs decision trees.

Failures: correlated features (double counting evidence); rare events (zero counts); continuous features mishandled as categorical.

> **Plan drift (Task 10):** (1) dual model: Gaussian NB (primary, 2 continuous features) + `discrete` toggle (categorical NB over a crafted 8-row table, V=4); smoothing = variance floor σ̃²=σ̂²+α for Gaussian / Laplace α for discrete (V counted correctly in the denominator). (2) correlation slider [0, 0.95] step 0.05 via shared latent `x1 = μ1 + √ρ·u + √(1−ρ)·w1`, `x2 = μ2 + √ρ·u + √(1−ρ)·w2` — Corr = ρ EXACTLY (construction verified); correlation sweep refits per snapshot; "without independence" contrast = full-covariance joint Gaussian (posterior flips 0.872 → 0.003 at ρ=0.9 — direction verified; Σ⁻¹ via adjugate, ΣΣ⁻¹=I checked). (3) distribution-view is Wave 4 — NOT registered; per-feature likelihoods shown via formula-explorer + scatter coloring instead. (4) log-space posterior via logsumexp (no underflow); fit memo cache bounded 64, keyed nClasses/nPerClass/correlation/smoothing/seed. NATs verified: nb-001 = 14/15, nb-005 = 5/6; naive-vs-joint contrast correct.
> SHIPPED: `d1492bc` (feat, 9 files, 13 tests) → reviews: spec APPROVE WITH NITS (13/13 items, 7 test cases); quality APPROVE (Gaussian MLE variance /n, variance floor on both features, logsumexp stability, ρ-exactness, contrast direction, Laplace V).

---

## Wave 3 — Geometry cluster (4 topics) ✅ COMPLETE

All four topics shipped (`430f1bd`, `d450c5e`, `293acad`, `58534ba`); review cycle closed (12 reviews — spec ×4, quality ×4, closure re-reviews ×4; fixes `cfb3f8f`/`e9dd50c`/`a1b82b9`/`5ea33d8` + hardening `97ed465`); suite 384/384 (32 files), lint, build, e2e 3/3. No new registry components — all four topics reuse the existing registered views (incl. decision-boundary + classifier registry). See per-task drift/SHIPPED notes below.

### Task 11: svm-hard-margin

**Files:** `src/topics/svm-hard-margin/{...}.ts`

TestCases:
1. `max margin solution on separable data` — weight vector perpendicular to boundary; margin = 2/‖w‖ correct for toy set (compute expected).
2. `support vectors are the closest points` — exactly 2-3 points on margin (distance = 1/‖w‖).
3. `scaling invariance` — scaling data by 2 doubles ‖w‖ but same boundary (margin/‖w‖ consistent).
4. `classifier correctness` — all points classified correctly with margin ≥ γ.

Math content: hyperplane w·x + b = 0, functional vs geometric margin, margin = 2/‖w‖, constrained optimization (min ½‖w‖² s.t. yᵢ(w·xᵢ+b) ≥ 1), Lagrangian, dual formulation, KKT (support vectors: αᵢ > 0), convexity.

Simulation: two clusters, `margin` separation slider, animate: candidate hyperplanes (showing non-max margins rejected), margin band expanding, support vectors highlighting as optimization proceeds (via simple iterative margin-maximization for 2D — exact QP is overkill; use a geometric solver: max over pairs of boundary points), loss curve of ½‖w‖² per iteration. Decision-boundary component shows final.

Derivations: (1) margin = 2/‖w‖; (2) primal to dual via Lagrangian (show steps); (3) KKT complementary slackness → support vectors.

Questions: NAT (compute margin given w,b; classify point), conceptual (why maximize margin — generalization), trap (support vectors vs all points), matrix (w as normal vector), visual (which points are SVs).

Mistakes: confusing functional/geometric margin; thinking all points matter (only SVs); forgetting b in margin; sign errors in constraint yᵢ(w·xᵢ+b) ≥ 1.

Comparisons: vs perceptron (any separator vs max margin), vs logistic (probabilistic vs geometric), vs LDA.

Failures: non-separable data (hard margin fails → soft margin); outliers (single outlier forces tiny margin).

> **Plan drift (Task 11):** (1) EXACT 2D geometric solver — candidate-direction enumeration (2 SVs → ŵ ∥ cross-class segment; 3 SVs → ŵ ⟂ same-class segment); per direction gap = min₁(ŵ·x) − max₀(ŵ·x), w = 2ŵ/gap so margin = gap; global optimum = largest-gap candidate (exact, deterministic; all-pairs superset of the plan's "max over pairs of boundary points"). (2) run = top-40 candidate sweep (SWEEP_CAP) weakest→strongest margin, lossMetricKey ½‖w‖² non-increasing 1.240917→1.227584, final snapshot = exact optimum (asserted to 9 decimals). (3) bounded deterministic seed search for separability (plan unspecified) — telemetry failure with "separable" in the reason when 200 seeds fail; tight/noisy in-range draws may advance past the requested seed (documented; failures.ts). (4) `scale` is test-only, no slider. (5) plan's "scaling by 2 doubles ‖w‖" framing corrected: scaling data UP by 2 HALVES ‖w‖ (features numerically larger → smaller weights suffice); the doubling direction is delivered via scale 0.5; margin·‖w‖ = 2 invariant exact in every case. Measured anchors: default w=(1.519692, 0.381713), b=−0.230105, ‖w‖=1.566898, margin=1.276408, SVs=[9,21]; seed 7: SVs=[0,14,22], margin 1.364655; boundary 14.1° off vertical.
> SHIPPED: `430f1bd` (feat, 9 files, 22 tests) → reviews: spec APPROVED-WITH-NITS (3 comment-level nits: seed-search header claim, "MINIMUM effective constraints" superlative, "weakest feasible separator" vs top-40 slice) → fixes `cfb3f8f` → closure: all 3 CLOSED; quality APPROVE (classifier tie-break aligned with sibling + orientedGap guard → `97ed465`).

### Task 12: svm-soft-margin

**Files:** `src/topics/svm-soft-margin/{...}.ts`

TestCases:
1. `C→∞ approximates hard margin` — on separable data, soft with C=1000 ≈ hard margin solution.
2. `C small allows misclassification` — with outlier, small C ignores it (margin large, 1-2 points inside margin); large C hugs it.
3. `hinge loss zero for confident correct` — max(0, 1 − y·f(x)) = 0 when y·f(x) ≥ 1.
4. `slack variables` — ξᵢ ≥ 0 and ξᵢ = 0 for non-violated points.

Math content: slack ξᵢ, objective ½‖w‖² + CΣξᵢ, hinge loss, C as regularization (bias-variance), subgradient, dual with box constraint 0 ≤ αᵢ ≤ C, relation to logistic loss (hinge vs CE).

Simulation: same 2D data + outlier toggle, `C` log-slider (0.01..1000), animate: margin shrink/grow with C, violated points (inside margin) shown with slack lines, hinge loss per point, loss curve. Decision-boundary + failure-view (outlier scenario).

Questions: NAT (compute hinge loss for given y·f), conceptual (C ↑ → variance ↑), trap (C=0 → all margin violation allowed), visual (margin vs C), indirect (hinge vs logistic loss compare).

Mistakes: thinking C ↑ always better; confusing slack with hinge loss; forgetting ξᵢ ≥ 0 constraint; sign conventions.

Comparisons: vs hard margin (limit), vs logistic regression (loss shape), vs ridge (C ↔ λ duality).

Failures: C→∞ with outliers (overfit noise); C→0 (underfit); label noise.

> **Plan drift (Task 12):** (1) deterministic geometric solver instead of QP — orientation grid + exact hinge-offset minimizer + golden-section scale/θ; C→∞ reproduces an independent exhaustive hard-margin reference to rel. diff 8.6e-6 / angle 0.017°. (2) plan's "C log-slider" delivered as linear slider + log-spaced C_GRID sweep in the run (one snapshot per grid value ≤ slider C; final snapshot = slider C exactly; lossMetricKey 'objective' + lossMetricKey2 'hingeLoss' dual series — hingeLoss ≡ slackSum by construction, alias documented). (3) measured behavior corrected the textbook story: a deep outlier is ABSORBED by slack even at C=1000 (boundary unchanged, C·Σξ = 99.99% of the objective — box constraint αᵢ ≤ C caps single-point influence); boundary distortion happens at SMALL C (label-noise demo moved to C=1, ‖Δw‖≈0.14); C=0.01: 19/20 points pay slack, 5 misclassified, margin 6.80 vs 2.21 (underfit end). (4) metric `freeSupportCount` (free SVs ξ=0 only); bounded SVs (ξ>0 ⇒ αᵢ=C) named explicitly in narration, consistent with the KKT story. All failure demos narrate measured values.
> SHIPPED: `d450c5e` (feat, 9 files, 16 tests) → reviews: spec CHANGES-REQUIRED (4 narration FAILs + supportCount semantics + 4 nits — every number independently re-measured) → fixes `e9dd50c` (+ formulas.ts:16 alignment) → closure: ALL CLOSED, 16/16; quality APPROVE (tie-break `>=`→`>` aligned with hard-margin sibling → `97ed465`).

### Task 13: perceptron

**Files:** `src/topics/perceptron/{...}.ts`

TestCases:
1. `converges on linearly separable data` — zero train error within bounded iterations (Perceptron Convergence Theorem — show iterations < some bound).
2. `oscillates on non-separable data` — never converges (weight vector keeps changing; detect via cycle length ≤ 2·‖w‖² bound or snapshot count cap).
3. `single update rule correct` — w ← w + η·y·x when misclassified (verify by hand example).
4. `weight norm growth` — ‖w‖ grows by at most bounded amounts (convergence theorem flavor).

Math content: linear decision rule ŷ = sign(w·x + b), update rule w ← w + η·yᵢ·xᵢ (on mistake), convergence theorem (R·‖w*‖/γ² iterations), margin γ, learning rate η (irrelevant for convergence in classic form but keep slider), bias update.

Simulation: separable toggle, `η`, `init`, animate: decision boundary moving (decision-boundary), mistake counter, weights vector (matrix-animator/arrow), iteration count vs bound; non-separable mode: show oscillation. Timeline: Initialize → Mistake → Update → Repeat → Converge.

Derivations: convergence theorem sketch (distance to target plane decreases).

Questions: NAT (apply update rule to given w, x, y), conceptual (perceptron vs GD — no loss function; updates only on mistakes), trap (non-separable → infinite loop; η doesn't affect convergence), visual (boundary rotation per update).

Mistakes: thinking perceptron minimizes a loss; using sigmoid; applying update on correct predictions; forgetting bias.

Comparisons: vs logistic (loss vs no loss), vs SVM (margin vs any separator), vs SGD (per-example).

Failures: non-separable data (oscillation); η huge (numerical blowup); imbalanced classes (majority boundary).

> **Plan drift (Task 13):** (1) Novikoff bound stated as (R/γ)² with GEOMETRIC γ — measured (2.244588/0.047207)² = 2260.769 (4 updates ≈ 565× under the bound; the plan's (R·‖w*‖/γ²) notation was the ‖w*‖-scaled variant, NOT Novikoff's theorem — corrected everywhere incl. a rewritten proof with the ‖w_k‖² ≤ k·η²R² norm recursion). (2) convergence check = clean-rotation scan (stricter than epoch-boundary; fixed a partial-sweep bug that masked accuracy 0.975 as converged); converged re-emission snapshot → 6 snapshots on the default; timeline 'Converge' only on the converged snapshot. (3) non-separable: overlapping offset clouds (class 0 at −margin, class 1 at origin — narration honest), exact-cycle detection + OSCILLATION_CAP=180 → 181 snapshots, failedAtStep 181, honest reason (cap message: "likely not separable, or needs more than the cap"). (4) loss-curve shows mistakesPerEpoch — "perceptron has NO loss function" (layer title). (5) η-invariance measured: η=1 vs 0.5 → identical 4 updates, weights scale exactly ×2. (6) required coverage adds: imbalanced-classes failure demo (nClass1 36:6 — measured: does NOT converge, minority accuracy 0.167, boundary x-intercept past the minority cluster, 89/180 updates on 3 minority points), vs-SGD comparison, sigmoid-misuse + update-on-correct gateTrap mistakes, random-init test (2 updates, accuracy 1).
> SHIPPED: `293acad` (feat, 9 files, 24 tests) → reviews: spec CHANGES-REQUIRED (8 findings: derivation validity, bound formula, oscillation honesty, 3 coverage gaps, 3 wording, 1 test gap) → fixes `a1b82b9` (→ 25 tests; (R/γ)² re-anchored 2260.769) → closure: ALL CLOSED (empirically re-verified, full suite green); quality APPROVE (nClass1 guard + internal interface export cleanup → `97ed465`).

### Task 14: lda (Fisher's LDA — classification + dimensionality reduction)

**Files:** `src/topics/lda/{...}.ts`

TestCases:
1. `projection direction known for 2 classes` — compute w = S_w⁻¹(μ₁−μ₂) on toy data, verify numerically.
2. `LDA projection maximizes class separation` — Fisher criterion J = (μ₁−μ₂)²/(s₁²+s₂²) higher than random directions.
3. `2-class LDA = decision rule` — threshold classification correctness on Gaussian classes.
4. `reduces to 1-D` — 2D data → 1D projection; variance within classes minimal.

Math content: between-class scatter S_B, within-class scatter S_W, Fisher criterion J(w) = wᵀS_Bw / wᵀS_Ww, solution w = S_W⁻¹(μ₁−μ₂), Gaussian assumption (shared covariance), threshold decision, LDA as dimensionality reduction (top k eigenvectors of S_W⁻¹S_B), relation to PCA (unsupervised vs supervised).

Simulation: 2 Gaussian clusters, sliders for means/covariance/angle, animate: candidate projection lines (rotate), J(w) value per angle (curve), optimal w highlighted, data projected onto line (points collapse to 1D with class histograms), decision threshold. Eigenviewer for the DR view (project to 1D, reconstruct).

Questions: NAT (compute w given μ, S_w — small numeric), conceptual (LDA vs PCA: labels), matrix (S_B/S_W eigen problem), visual (projection quality), trap (LDA assumes shared covariance/Gaussianity).

Mistakes: using PCA when labels available; forgetting S_W⁻¹; assuming LDA handles non-Gaussian; confusing within/between scatter.

Comparisons: vs PCA (supervised vs unsupervised — frequently-confused edge), vs logistic (generative vs discriminative), vs SVM.

Failures: singular S_W (few samples per class); non-Gaussian multimodal classes; outliers in covariance estimation.

> **Plan drift (Task 14):** (1) EXACT analytic solution w = S_W⁻¹(μ₁−μ₀) via adjugate (2×2) — no training iterations; run = 36×5° direction sweep (J(θ) curve, per-angle projection metrics) + final snapshot = closed-form optimum evaluated EXACTLY (J = 3.869000715 strictly beats every grid angle, max 3.861638793). (2) S_W convention = normalized per-class covariances (C₀+C₁) making J(ŵ) collapse EXACTLY to the plan's test-2 formula (μ̄₁−μ̄₂)²/(s₀²+s₁²); eigen-link λ = dᵀS_W⁻¹d asserted to 1e-9. (3) eigenviewer + distribution-view are Wave 4 — substituted with scatter projection + matrix-animator + metrics (registry-only rule). (4) unequal-priors Bayes threshold taught with the corrected formula τ = ŵᵀ(μ₀+μ₁)/2 − ln(P(C₁)/P(C₀))/‖w‖ (sign + scale fixed in review; simulation uses the equal-priors midpoint). (5) singular S_W → telemetry failure "singular" (failures.ts); lossMetricKey 'jFisher' higher-better (layer title says so); "variance within classes minimal" honestly qualified (Fisher maximizes the RATIO — within-variance compresses below the mean-diff axis but is not the absolute minimum).
> SHIPPED: `58534ba` (feat, 9 files, 29 tests) → reviews: spec CHANGES-REQUIRED (1 math finding — Bayes threshold sign+scale, verified τ=−4.394 on the diag(4,1)/π₁=0.9 counterexample; 3 narration nits) → fixes `5ea33d8` → closure: ALL CLOSED, 29/29; quality APPROVE (minors informational — invSW reuse, near-singular test — non-blocking).

---

## Wave 4 — Dim-reduction cluster (3 topics)

### Task 15: New registry component — eigenviewer (+ distribution-view)

**Files:** `src/visualizers/Eigenviewer.tsx`, `src/visualizers/DistributionView.tsx`, tests.

- Eigenviewer: animated rotation + projection. Props: `{ snapshot?, params }` — VisualCommands: `{type:'axis', id, angle, color}`, `{type:'projection', point, onto, residual}`. Renders: 2D data cloud, candidate axis (rotatable via slider), projection lines from points to axis, projected points on axis (1D strip), variance explained bars, reconstruction (project back + error lines). Zoom/pan reuse CanvasStage.
- DistributionView: plots a distribution (PDF) per class with adjustable params — for NB (per-feature Gaussians), MLE (likelihood curve), LDA (class densities). Props: `{ distributions: {label, mean, variance, color}[], xRange, yRange }`.

TDD both, then commit: `feat: eigenviewer and distribution-view components`

### Task 16: pca

**Files:** `src/topics/pca/{...}.ts`

TestCases:
1. `first PC maximizes variance` — variance along PC1 ≥ variance along any other direction (numerical check).
2. `PCs are orthogonal` — v₁·v₂ = 0 (within 1e-6).
3. `eigenvalues = explained variance` — λ_k = variance of data projected on v_k (numerical).
4. `centering matters` — PCA without centering gives different (wrong) PCs; centered gives correct.
5. `reconstruction` — projecting to k PCs then back: reconstruction error = Σ_{j>k} λ_j (verify).

Math content: covariance matrix Σ = (1/n)XᵀX (centered), eigen-decomposition Σv = λv, variance maximization derivation (Rayleigh quotient), projection P = X·v_k, explained variance ratio, reconstruction, k selection (elbow), relation to SVD (X = UΣVᵀ; PCs = right singular vectors).

Derivations: (1) variance along w = wᵀΣw; (2) maximize wᵀΣw s.t. ‖w‖=1 via Lagrange → Σw = λw; (3) reconstruction error = sum of dropped eigenvalues.

Simulation: 2D correlated data (slider: correlation, rotation angle, noise), animate: axis rotation (eigenviewer), variance curve vs angle (peak at PC1), projection animation, reconstruction with k=1 vs k=2 (error lines), explained variance bars, scree plot. Matrix view: covariance matrix with eigenvalues highlighted (matrix-animator), eigen-decomposition steps.

Questions: NAT (given covariance matrix, find eigenvalue — 2×2 characteristic polynomial), conceptual (why eigenvectors), matrix (Σ = (1/n)XᵀX derivation), visual (which direction is PC1), trap (uncentered PCA; eigenvalue ordering), indirect (SVD link).

Mistakes: forgetting centering; confusing eigenvectors with PCs (directions vs projections); sorting eigenvalues wrong; thinking PCA is supervised.

Comparisons: vs LDA (labels vs no labels — frequently-confused), vs feature selection (selection vs extraction), vs SVD (equivalent computation).

Failures: high noise (PCs unstable); outliers (variance dominated); scaling (features with different units dominate).

### Task 17: pca-svd

**Files:** `src/topics/pca-svd/{...}.ts`

TestCases:
1. `SVD gives same PCs as eigen-decomposition` — right singular vectors V = eigenvectors of XᵀX (numeric).
2. `singular values relate to eigenvalues` — σ_k² = λ_k of covariance.
3. `low-rank approximation` — rank-k approximation error = σ_{k+1} (Eckart-Young).
4. `economy SVD shapes` — U (n×k), Σ (k×k), Vᵀ (k×d) dimensions.

Math content: X = UΣVᵀ, left/right singular vectors, singular values, relation XᵀX = VΣ²Vᵀ, Eckart-Young theorem, low-rank approximation (image compression demo possible), pseudoinverse via SVD.

Simulation: same 2D data + matrix view of full SVD (U, Σ, Vᵀ as animated matrices with dims), rank slider (1..min(n,d)), reconstruction with error heatmap, singular value bars, relation animation (XᵀX eigen-decomposition vs SVD side-by-side).

Questions: NAT (given SVD matrices, find rank-k approx error), conceptual (why SVD is numerically stable), matrix (dimension compatibility UΣVᵀ), visual (compression quality vs rank), trap (U vs V confusion), indirect (pseudoinverse for OLS).

Mistakes: confusing left/right singular vectors; forgetting Σ is diagonal sorted; thinking SVD needs square matrices; sign ambiguity of singular vectors.

Comparisons: vs eigen-decomposition (symmetric vs general), vs PCA (same math), vs normal equation (pseudo-inverse stability).

Failures: NaN for missing data (SVD needs complete matrices — note alternatives); numerical sign flips.

### Task 18: mle (Maximum Likelihood Estimation — REQUIRED topic; spec §5 topic #12)

**Files:** `src/topics/mle/{...}.ts`

TestCases:
1. `MLE recovers true Bernoulli parameter` — MLE of p on clean Bernoulli samples ≈ empirical frequency.
2. `MLE recovers Gaussian mean/variance` — MLE μ ≈ sample mean, σ² ≈ biased (÷n) sample variance, not ÷(n−1).
3. `log-likelihood maximization` — likelihood < log-likelihood monotonicity: argmax identical, verify via numeric gradient.
4. `MLE = OLS for Gaussian noise` — linear model with Gaussian noise: MLE solution equals normal equation (numeric).
5. `MLE consistency demo` — estimate improves with sample count n (10 → 100 → 1000, noise seeded).

Math content: likelihood L(θ), log-likelihood ℓ(θ), score function = derivative, MLE invariance property, asymptotic normality (mention), connection: CE loss = negative log-likelihood, OLS = MLE under Gaussian noise. Derivation: MLE for Bernoulli (coin flip), Gaussian (μ, σ²), linear regression (via score = 0).

Simulation: dataset family select (coin flips, Gaussian samples, linear regression), `n` slider (seeded), distribution curve animated as n grows, MLE estimate marker vs true parameter, log-likelihood surface over (μ, σ²) as 2D contour + gradient arrows (self-contained via CanvasStage path — do NOT depend on loss-landscape, which lands in Task 27/Wave 6; swap to 3D surface later when available), bias demo (σ² ÷n vs ÷(n−1) gap shrinks with n).

Questions: NAT (derive MLE of λ for exponential), conceptual (why log-likelihood), trap (biased variance in MLE), visual (likelihood vs log-likelihood), matrix (score = Xᵀ(y−Xθ) for regression), indirect (MLE of Bernoulli from counts).

Mistakes: confusing likelihood with probability; forgetting MLE variance is biased (÷n); maximizing likelihood instead of log (underflow); plugging sample variance into Gaussian MLE without bias discussion.

Comparisons: vs MAP (prior term), vs CE (same math different lens), vs OLS (MLE instance).

Failures: underflow with many samples (use log), non-identifiable parameters (flat likelihood), non-convex likelihood (local maxima).

Commit: `feat: mle topic module with likelihood surface simulation`

---

## Wave 5 — Trees & clustering cluster (4 topics)

### Task 19: New registry components — tree-builder, cluster-animator, dendrogram

**Files:** `src/visualizers/TreeBuilder.tsx`, `src/visualizers/ClusterAnimator.tsx`, `src/visualizers/Dendrogram.tsx`, tests.

- TreeBuilder: renders decision tree (nodes, splits with thresholds, leaf classes), grows step-by-step from VisualCommands `{type:'node', id, x, y, label, splitInfo}`; entropy/gini bars per node; highlight path for a sample.
- ClusterAnimator: scatter with centroids (colored), assignment lines, loss per iteration text, animated convergence; VisualCommands `{type:'centroid'}`, `{type:'assignment'}`.
- Dendrogram: hierarchical tree with merge heights, distance axis, click a node → shows merged clusters; VisualCommands `{type:'merge', id, height, children}`.

TDD each, commit: `feat: tree, cluster, dendrogram visualizers`

### Task 20: decision-trees (classification, entropy)

**Files:** `src/topics/decision-trees/{...}.ts`

TestCases:
1. `pure split gives zero entropy` — entropy of {a,a,a,a} = 0.
2. `maximum entropy at uniform` — entropy of 4 classes uniform = log₂4 = 2.
3. `information gain chooses correct feature` — on XOR-ish toy, IG picks the right first split.
4. `tree classifies training data perfectly` — fully grown tree, zero train error (overfit demo).
5. `pruning reduces overfit` — test error improves after pruning with min-leaf constraint.

Math content: entropy H = −Σp log₂p, conditional entropy, information gain IG = H(parent) − Σ(n_k/n)H(child), Gini (mention, full in regression task), split selection, stopping criteria (depth, min samples, impurity threshold), pruning (pre/post), overfitting.

Simulation: 2D synthetic (2-4 classes), animate tree growth split-by-split (tree-builder): at each step show current node's entropy bar, candidate splits with IG values, chosen split highlighted, data partition updated (decision-boundary), depth slider + minLeaf slider; show train/test accuracy vs depth.

Questions: NAT (compute entropy/IG for given distribution — classic), conceptual (why entropy vs Gini), visual (which split chosen), trap (log base; weighted child entropy), indirect (IG with continuous features).

Mistakes: forgetting weighted average of children; log base 2; thinking more depth always better; using IG with non-binary splits incorrectly.

Comparisons: vs SVM (boundary shape), vs KNN (interpretability), vs random forest (ensemble — mention as extension).

Failures: high variance (deep trees), class imbalance (entropy biased), correlated features (unstable splits), tiny data (spurious splits).

### Task 21: decision-trees-regression (Gini, CART, regression trees)

**Files:** `src/topics/decision-trees-regression/{...}.ts`

TestCases:
1. `gini impurity` — pure node gini = 0; uniform 2-class gini = 0.5.
2. `regression tree splits at midpoint optimum` — on 1D data, best split minimizes SSE (verify by exhaustive check on toy).
3. `leaf prediction = mean of samples` — regression leaf value = mean(y in node).
4. `CART handles continuous features` — split threshold chosen among midpoints; test correctness.

Math content: Gini impurity 1 − Σp², SSE reduction for regression, CART algorithm (binary splits, greedy), split candidates (sorted midpoints), leaf = mean, cost-complexity pruning mention, comparison entropy vs gini (similar, gini cheaper).

Simulation: 1D regression data (noisy curve), animate: candidate split lines with SSE, chosen split, tree growth (tree-builder), fitted step-function (scatter + step curve), depth/leaf sliders, train vs test error vs depth.

Questions: NAT (compute gini/SSE for given split), conceptual (gini vs entropy), visual (step function fit), trap (regression tree predicts mean — not linear), matrix (n/a), indirect (CART vs ID3 differences).

Mistakes: using classification metrics for regression trees; thinking leaves store linear models (they store constants); greedy non-optimality.

Comparisons: vs linear regression (piecewise constant), vs kNN (adaptive resolution), vs SVM regression.

Failures: extrapolation (constant outside range), high variance, jagged fits with deep trees.

### Task 22: kmeans

**Files:** `src/topics/kmeans/{...}.ts`

TestCases:
1. `k-means converges monotonically` — loss (SSE) non-increasing per iteration.
2. `recovers known clusters` — 3 well-separated Gaussian blobs → correct assignments (ARI ≈ 1 or exact match).
3. `initialization matters` — bad seeds → suboptimal local minimum (two different seeds → different loss; demo).
4. `empty cluster handling` — with k > natural clusters, some centroids get no points → reinitialize strategy works.
5. `k=1 trivial` — single centroid = data mean.

Math content: SSE objective J = ΣΣ‖x − μ_k‖², assignment step (argmin distance), update step (mean), convergence proof sketch (two monotonically decreasing steps), initialization (Forgy, k-means++), k selection (elbow, silhouette), Lloyd's algorithm, time complexity O(n·k·d·iter).

Simulation: n points, `k` slider, `init` select (random, k-means++), `seed`, animate: assignment (colors per centroid), centroids moving (cluster-animator), J value per iteration (loss-curve), step counter, final SSE; elbow plot (J vs k) as separate view (curve-comparator).

Questions: NAT (assign point to centroid / compute new centroid), conceptual (why k-means can be suboptimal), visual (elbow reading), trap (k-means assumes spherical clusters; scaling), indirect (EM relation — hard vs soft assignment).

Mistakes: assuming k known; not scaling features; thinking k-means finds global optimum; distance in raw feature space with mixed units.

Comparisons: vs hierarchical (flat vs nested), vs GMM (hard vs soft assignment), vs DBSCAN (k not needed).

Failures: non-spherical clusters; outliers; high dimensions; k wrong; different scales.

### Task 23: hierarchical-clustering

**Files:** `src/topics/hierarchical-clustering/{...}.ts`

TestCases:
1. `single linkage merges nearest pair` — verify merge order on 4 points by hand.
2. `linkage affects structure` — single vs complete produce different dendrograms on same data (assert different merge heights).
3. `dendrogram is a tree` — n points → n−1 merges.
4. `cut at height k` — cutting dendrogram at height h yields k clusters consistent with threshold.

Math content: agglomerative (bottom-up) vs divisive (top-down), linkage criteria (single, complete, average, Ward), distance matrix update (Lance-Williams mention), dendrogram interpretation, cophenetic distance, time complexity O(n³)/O(n² log n) with heaps.

Simulation: points, `linkage` select, animate merges one at a time (dendrogram grows, points merge colors), distance matrix heatmap (matrix-animator with color cells), cut-height slider → clusters highlighted, cophenetic correlation display.

Questions: NAT (merge order for 4 points given distances), conceptual (single linkage chaining), visual (dendrogram reading), trap (Ward uses variance not distance directly), indirect (vs k-means).

Mistakes: single vs complete chaining confusion; reading dendrogram height wrong; thinking merges are reversible.

Comparisons: vs k-means (no k needed, nested structure), vs DBSCAN (arbitrary shapes).

Failures: chaining with single linkage; O(n³) on large data; noisy outliers create singleton clusters.

---

## Wave 6 — Deep learning cluster (3 topics)

### Task 24: New registry components — nn-inspector, activation-view

**Files:** `src/visualizers/NNInspector.tsx`, `src/visualizers/ActivationView.tsx`, tests.

- NNInspector: layered network graph (input → hidden → output), neuron activations (colored intensity), weights on edges (width/thickness + tooltip), biases, gradients (red/green arrows during backprop), dead neurons (gray), vanish/explode indicators. Props: `{ snapshot?, params }`; VisualCommands `{type:'neuron', id, layer, index, activation, dead}`, `{type:'edge', from, to, weight, gradient}`.
- ActivationView: plots activation functions (sigmoid, tanh, ReLU, LeakyReLU, ELU) with derivative curves, slider for input z, shows σ(z) and σ′(z); overlay of gradients to demonstrate vanishing (sigmoid saturates, ReLU dead). Registered for reuse.

TDD both, commit: `feat: nn-inspector and activation-view`

### Task 25: neural-networks (FFN)

**Files:** `src/topics/neural-networks/{...}.ts`

TestCases:
1. `forward pass correct for tiny net` — hand-computed 2-2-1 net with fixed weights → verify output.
2. `universal approximation flavor` — 1 hidden layer with enough units fits XOR (accuracy → 1 with tanh).
3. `activation matters` — ReLU vs sigmoid convergence speed on deeper net (sigmoid slower — check epoch counts).
4. `weight init scale matters` — init too large → gradients explode → loss NaN.

Math content: neuron computation z = w·a + b, activation functions (sigmoid, tanh, ReLU, leaky ReLU, softmax output), layer stacking, vectorized forward pass (matrix form: a^(l) = σ(W^(l)a^(l−1) + b^(l))), universal approximation theorem intuition, capacity/overfitting, weight initialization.

Simulation: `architecture` config (2-4-2, 3-5-3...), `activation` select, `lr`, `epochs`, `initScale`; animate forward pass per sample (nn-inspector: activations lighting up), loss curve, decision boundary of the net (decision-boundary with net classifier), per-epoch accuracy; weight matrix views (matrix-animator per layer).

Questions: NAT (forward pass compute for given weights — the classic 2-2-1), conceptual (why non-linear activations), matrix (layer dims), visual (boundary complexity vs width), trap (XOR needs hidden layer).

Mistakes: no bias; wrong dims in W·a; linear activation collapse; init to zero (symmetry).

Comparisons: vs logistic (single neuron), vs SVM (learned features vs kernel), vs decision tree.

Failures: overfitting (wide nets), vanishing gradients (sigmoid deep), dead ReLU (negative region), bad init.

### Task 26: backpropagation

**Files:** `src/topics/backpropagation/{...}.ts`

TestCases:
1. `gradient matches finite differences` — numeric check ∂L/∂w vs (L(w+ε)−L(w−ε))/2ε for a small net (the gold test).
2. `chain rule composition` — hand-derived gradient for 2-2-1 matches backprop output.
3. `gradient flow scales with init` — large init → large gradients (explode); sigmoid deep → small (vanish).
4. `weight update decreases loss` — one SGD step with η small → loss decreases (verify).
5. `dead ReLU produces zero gradient` — neuron stuck at 0 → gradient exactly 0.

Math content: chain rule, computational graph, forward pass, loss gradient at output (ŷ−y for MSE / CE), local gradients, delta rule δ^(l) = (W^(l+1)ᵀδ^(l+1)) ⊙ σ′(z^(l)), weight gradient ∂L/∂W^(l) = δ^(l)a^(l−1)ᵀ, parameter update, vanishing/exploding gradient analysis (product of σ′ and W), initialization strategies.

Derivations: (1) backprop equations for MSE; (2) for CE + softmax (δ = ŷ − y — beautiful cancellation); (3) why sigmoid vanishes.

Simulation: small net (2-2-1 or 2-3-1 configurable), step-by-step: forward pass (nn-inspector lighting), loss, output δ, hidden δs (backward glow animation — gradients colored arrows), weight updates (edge thickness changes), epoch loop; controls: play/pause/step per FULL backprop step or per layer; vanish/explode slider (initScale, activation, depth) with gradient magnitude readout (log-scale meter).

Questions: NAT (compute δ for given values — the classic GATE numeric), conceptual (why chain rule), matrix (Wᵀδ dimension dance), trap (elementwise vs matrix multiply in δ), indirect (CE+softmax cancellation).

Mistakes: transposing W wrong in δ^(l) = Wᵀδ; forgetting σ′ factor; applying δ as row vs column; updating with wrong sign.

Comparisons: vs analytical gradient (finite diff check), vs chain rule in calculus, vs GD (same update, computed via BP).

Failures: vanishing (deep sigmoid), exploding (large init), dead ReLU, NaN loss.

### Task 27: optimization-foundations (GD variants — GD/SGD/Mini-batch/Momentum/Adam view)

**Files:** `src/topics/optimization-foundations/{...}.ts`, `src/visualizers/OptimizerView.tsx` (registry component), `src/visualizers/LossLandscape.tsx` (registry component)

- **LossLandscape (NEW registry component, required — spec §4):** 3D-perspective surface for 2-weight loss functions. Own small projection math (`src/lib/math/project3d.ts`: rotate → perspective divide → screen), renders wireframe/height-colored mesh via CanvasStage, optional 2D-contour fallback view toggle. Emits `{type:'surface'}` VisualCommands; consumes `highlight` events to draw a trail point at (θ₁, θ₂, loss). Register as `loss-landscape` in viewRegistry (Task 6 of Wave 0 pattern). TDD it before the topic (surface rotation keeps viewpoint stable; a known bowl maps to expected projected center; contour fallback equals analytic level set).
- OptimizerView: 2D loss surface (bowl + optional saddle), trails of multiple optimizers simultaneously (GD, SGD, mini-batch, momentum, Adam), step-by-step animation, per-optimizer loss curves, parameter traces. Reusable for gradient-descent topic too (register as `optimizer-view`).

TestCases:
1. `SGD noisy but converges` — SGD on quadratic converges in distribution (final loss < ε with averaging).
2. `momentum accelerates` — momentum converges in fewer epochs than plain GD on ill-conditioned bowl.
3. `Adam adapts lr` — Adam handles large lr better than GD on scaled features (no divergence).
4. `mini-batch variance` — batch=1 (SGD) noisier loss curve than batch=32 (smoother).

Math content: batch/mini-batch/SGD tradeoffs, learning rate schedules, momentum (velocity), Nesterov (mention), AdaGrad/RMSProp (mention), Adam (moments, bias correction), convergence rates (convex: O(1/T) GD vs O(1/√T) SGD), stochastic gradient noise.

Simulation: surface select (bowl, ill-conditioned ellipse, saddle), `optimizer` select (compare up to 3 simultaneously), `lr`, `batchSize`, `momentum`, `beta1/beta2`, `epochs`; trails animated on `loss-landscape` (3D-perspective surface) with 2D-contour fallback toggle, loss curves per optimizer (curve-comparator).

Questions: conceptual (why SGD generalizes better — flat minima), NAT (momentum update formula), trap (Adam hyperparameters), matrix (parameter vector updates).

Mistakes: momentum direction errors; Adam bias correction omission; assuming SGD always slower in epochs.

Comparisons: table of optimizers (complexity, memory, convergence, when to use).

Failures: lr too high (Adam mitigates but not immune); saddle points (GD slows, momentum escapes).

---

## Wave 7 — Evaluation & meta cluster (6 topics)

### Task 28: New registry components — confusion-explorer, roc-viewer, curve-comparator

**Files:** `src/visualizers/ConfusionExplorer.tsx`, `src/visualizers/RocViewer.tsx`, `src/visualizers/CurveComparator.tsx`, tests.

- ConfusionExplorer: 2×2 (or k×k) confusion matrix, TP/FP/TN/FN cells clickable → highlights which points are which (scatter overlay), animated threshold sweep, precision/recall/F1/accuracy readouts (metric-grid).
- RocViewer: ROC curve build-up as threshold sweeps (points appear on curve), TPR/FPR readouts, AUC fill animation, threshold marker ↔ confusion matrix sync via event bus.
- CurveComparator: multi-series line chart (train vs val loss, error vs complexity), ideal for bias-variance and CV.

TDD each, commit: `feat: confusion, roc, curve-comparator components`

### Task 29: classification-metrics (precision/recall/F1/confusion matrix)

**Files:** `src/topics/classification-metrics/{...}.ts`

TestCases:
1. `metrics from confusion matrix` — given TP/FP/TN/FN: precision, recall, F1, accuracy, specificity computed correctly.
2. `F1 is harmonic mean` — F1 = 2PR/(P+R) numeric.
3. `imbalanced data demo` — 99:1 imbalance: accuracy high but recall of minority low (the classic trap).
4. `threshold changes metrics` — increasing threshold ↑ precision, ↓ recall (monotone trends on crafted scores).

Math content: confusion matrix entries, precision = TP/(TP+FP), recall = TP/(TP+FN), F1, accuracy, specificity, TPR/FPR, macro vs micro averaging, threshold as tradeoff knob, imbalanced class effects.

Simulation: synthetic scores + labels, threshold slider → confusion matrix updates live (confusion-explorer), scatter with classification at threshold, precision/recall/F1 bars animate, imbalanced toggle.

Questions: NAT (compute F1 from confusion matrix — the classic), conceptual (why accuracy fails on imbalance), trap (precision vs recall confusion — frequently-confused edge), visual (threshold effect), indirect (link to ROC next topic).

Mistakes: using accuracy on imbalance; F1 harmonic vs arithmetic mean; macro vs micro confusion; defining positive class carelessly.

Comparisons: vs ROC/AUC (threshold-free), vs accuracy, vs log loss.

Failures: F1 with P+R=0 (undefined → 0 convention); extreme imbalance.

### Task 30: roc-auc

**Files:** `src/topics/roc-auc/{...}.ts`

TestCases:
1. `AUC = probability random positive ranked above random negative` — numeric simulation check.
2. `perfect classifier AUC = 1` — separable scores.
3. `random classifier AUC = 0.5` — independent scores (within tolerance).
4. `AUC invariant to score scaling` — AUC same for scores vs 2×scores (monotone transform).

Math content: TPR/FPR across thresholds, ROC curve construction (sort by score, sweep), AUC as integral, AUC interpretation (Mann-Whitney U relation), comparing classifiers via ROC, why AUC is threshold-independent.

Simulation: scores slider (two distributions overlap slider), animate threshold sweep (roc-viewer + confusion sync), AUC fill, point on ROC ↔ threshold ↔ confusion matrix (event bus link — the highlight showcase), classifier comparison (two curves).

Questions: NAT (given scores + labels compute AUC via ranking — the classic numerical GATE question), conceptual (why AUC threshold-independent), visual (curve reading: which classifier better), trap (AUC 0.5 = random; inverted classifier AUC < 0.5 → flip predictions).

Mistakes: plotting TPR vs precision; wrong AUC computation order; thinking AUC = accuracy.

Comparisons: vs precision-recall (imbalance: PR better), vs accuracy, vs log loss (proper scoring rule).

Failures: extreme imbalance (PR curve more informative), ties in scores (handling).

### Task 31: bias-variance

**Files:** `src/topics/bias-variance/{...}.ts`

TestCases:
1. `decomposition on synthetic` — expected MSE ≈ bias² + variance + irreducible (numeric check via many seeds).
2. `high bias ↔ underfit` — degree-1 on quadratic: bias² dominates.
3. `high variance ↔ overfit` — degree-15: variance dominates.
4. `irreducible error floor` — even perfect model has noise variance floor.

Math content: bias-variance decomposition E[(ŷ−y)²] = bias² + variance + σ², derivation (expand, cross term vanishes), tradeoff curves (train vs test error vs complexity), how λ/degree/C/k control it, ensemble intuition (bagging reduces variance).

Simulation: polynomial regression with `degree` slider, many seeds (nRuns slider, e.g., 20 runs), animate: individual fits (translucent lines), mean fit, bias²/variance/σ² bars updating per degree (curve-comparator + metric-grid), sweet spot highlighting (min test error).

Questions: conceptual (decomposition terms), NAT (compute expected error given bias/variance/noise), visual (which complexity), trap (bias-variance always tradeoff — can both be reduced with more data), indirect (regularization link).

Mistakes: forgetting irreducible error; thinking more data fixes high bias; confusing train error drop with generalization.

Comparisons: vs regularization (λ controls), vs model complexity (degree), vs ensemble.

Failures: small n (noisy estimates of bias/variance), heteroscedastic noise.

### Task 32: overfitting-underfitting (+ evaluation framing)

**Files:** `src/topics/overfitting-underfitting/{...}.ts`

TestCases:
1. `train error < val error on overfit` — deep polynomial: train ≈ 0, val high.
2. `capacity curves` — error vs capacity U-shape for val, monotone ↓ for train.
3. `data size effect` — more data reduces gap (both errors ↓ toward irreducible).
4. `regularization reduces gap` — ridge λ reduces val error on overfit regime.

Math content: capacity, train/validation/test split (why 3-way), error decomposition, learning curves (error vs n), signs of over/underfitting, remedies table (more data, regularization, simpler model, ensembles, CV).

Simulation: polynomial regression, `degree`, `nTrain`, `noise`, `regularization` toggle; animate fits + learning curves (curve-comparator), shaded train/val regions, diagnosis panel ("Overfitting: train error is low but val error is high — increase λ or reduce degree").

Questions: conceptual (identify overfit from curves), NAT (given errors, classify regime), visual (learning curve reading), trap (val set used for tuning → needs test set), indirect (CV).

Mistakes: using test set for tuning; assuming more features help; ignoring val error.

Comparisons: vs bias-variance (same story), vs regularization topics.

Failures: data leakage, small val set noisy.

### Task 33: cross-validation

**Files:** `src/topics/cross-validation/{...}.ts`

TestCases:
1. `k-fold estimate unbiased-ish` — mean CV error ≈ true generalization error on synthetic (within tolerance across seeds).
2. `LOO = k-fold with k=n` — LOO estimate matches k=n numerically.
3. `stratified preserves class ratio` — stratified k-fold fold class ratios ≈ overall ratio.
4. `high k → lower bias, higher variance` — LOO (k=n) has higher variance of estimate than 5-fold (empirical across seeds).

Math content: train/val/test, k-fold, LOO, stratified k-fold, bias/variance of CV estimates, repeated CV, nested CV (model selection), leave-one-out error = hat matrix diagonal (Allen's PRESS — advanced note), choice of k (bias-variance of estimate).

Simulation: dataset, `k` slider (2..n), `repeats`, animate: fold coloring (which subset is val), train on k−1 folds → error on held-out, accumulate CV error, fold-by-fold error bars, LOO mode (n folds, fast path), stratified toggle showing class ratios per fold.

Questions: NAT (k-fold error computation from fold errors — classic), conceptual (why LOO high variance), trap (shuffling vs stratification; data leakage in preprocessing), visual (fold coloring), indirect (model selection).

Mistakes: leaking preprocessing (fit scaler on full data before split); non-stratified regression (use KFold not Stratified for regression); using CV error as test error.

Comparisons: vs holdout, vs bootstrap (mention), vs nested CV.

Failures: small n (LOO variance), time series (need time-series split — note).

### Task 34: introduction (ML fundamentals module)

**Files:** `src/topics/introduction/{...}.ts`

A lighter-weight module (no heavy simulation — focus on concept taxonomy + interactive examples):
- TestCases: small (e.g., `supervised vs unsupervised demo`: k-means on unlabeled vs regression on labeled — verify outputs match expectation), `train/test split correctness`.
- Content: what ML is, learning paradigms (supervised/unsupervised/semi/RL), hypothesis space, generalization, features/labels, when ML works/fails, terminology (epoch, batch, loss).
- Simulation: tiny playground: scatter with label toggle (colored vs not) — points draggable; a "ML pipeline" step viewer (data → split → train → evaluate) with a decision-boundary classifier.
- Questions: conceptual only (5-6).
- Mistakes: ML vs statistics confusion, memorization vs learning.

---

## Wave 7.5: Domain completion & polish

### Task 35: Concept Explorer page (concept-first navigation)

**Files:** `src/pages/ConceptPage.tsx`, `src/visualizers/conceptExplorer/conceptData.ts`, route `/concept/:conceptId`, sidebar link.

- conceptData.ts: concept nodes (sigmoid, entropy, eigenvector, margin, gradient, likelihood, softmax, information-gain, slack, kernel, centroid...) each with: definition (KaTeX), topics referencing it (with "how it appears" notes), related concepts with edge type. Reuse `graphEdges` where possible; concepts link to topics.
- ConceptPage: renders concept detail (definition, math, "appears in" topic chips → navigate), plus a mini local graph (D3, same style as KnowledgeGraph) centered on the concept (1-hop neighborhood).
- Search integration: CommandPalette also matches concept names.

TestCases: none heavy; add unit test for conceptData integrity (every topic ref exists).

### Task 36: Resources layer — datasets, CSV upload, images (spec §9)

**Files:** `src/resources/datasets/`, `src/resources/images/`, `src/resources/index.ts`, `src/resources/ResourcesPage.tsx` (route `/resources`), `src/lib/math/csv.ts`

- Built-in datasets (synthetic, generated at runtime with seeded PRNG — no binaries): iris-like 3-class (only as *synthetic* stand-in, clearly labeled, do NOT claim real Fisher iris values), wine-like 3-class stand-in, gaussian-blobs 2-class (used by several topics), housing-like regression (n=50, features = area, bedrooms, age), XOR, spiral. Each: `{ id, name, description, n, d, classes?, generator(params) → { X: number[][], y: (number|string)[] } , license: 'synthetic' }`.
- `src/resources/index.ts`: `getDataset(id)`, `listDatasets()`, `saveCustomDataset(id, {X, y})` (localStorage), `loadCustomDatasets()`.
- CSV upload: `csv.ts` — parse CSV text (handles quoted fields, header row, mixed numeric/categorical columns) → dataset; store parsed result in localStorage via progressStore (`customDatasets` key); UI on ResourcesPage: upload button (file input → text → parse → preview table → save → appears in topic "dataset" selects for topics that support dataset selection: knn, decision-trees, kmeans, logistic-regression, naive-bayes, svm).
- Images folder: keep empty placeholder with README note (no binary assets; everything procedural).
- TestCases: `csv parse correctness` (quotes, commas in quotes, numeric coercion, header detection), `roundtrip save/load` (localStorage mock), `dataset shape contract` (every built-in returns consistent X/y shapes).

### Task 37: Telemetry overlay + performance guard

**Files:** `src/visualizers/TelemetryPanel.tsx`, update `ViewHost`.

- Reads `run.telemetry` (snapshotCount, genMs, memBytes, failedAtStep) + render FPS (measured via rAF timestamps in ViewHost).
- Shown when `settings.showTelemetry`; warns when genMs > 500 (heavy topic suggestion: reduce maxSteps).
- Also used by performance-engineer during Wave reviews.

### Task 38: Recorder finalization (GIF/PNG sequence download)

**Files:** update `src/lib/exporters/recorder.ts`, `src/visualizers/Recorder.tsx`.

- PNG sequence → client-side ZIP (use `jszip` — add dependency) with `frames/0001.png...`.
- Download button in Recorder view.
- GIF: optional via `gif.js` (add dependency) — encode up to 60 frames at 12 fps; keep behind a "GIF (beta)" button.

### Task 39: Analytics & revision dashboard

**Files:** `src/pages/DashboardPage.tsx`, route `/dashboard`, sidebar link.

- Shows: topics completed, time per topic (bar), question accuracy per topic, weakest topics (from analyticsStore), most-revisited simulations, bookmarks list, session resume list.
- Revision planner: given `metadata.revision` (5/15/30/60-min) and analytics gaps, suggests next study path (simple heuristic: weakest topic → shortest revision path).

### Task 40: Final wave — full suite + QA + deployment config

**Files:** `vercel.json` (or GitHub Pages config via `base` in vite.config), final QA.

- [ ] Run: `npm run lint && npm run test && npm run build`
- [ ] Run: `npx playwright test` — extend e2e smoke to cover 5 representative topics across waves.
- [ ] Verify: every topic listed in the knowledge graph has a registered module (`loadAllTopics` count == graph topic nodes count — write a test).
- [ ] Manual QA pass per Wave 0 checklist across all 31 topics.
- [ ] Deployment: add `vercel.json` (SPA rewrite) + document `npm run deploy` script; or set `base: './'` for GitHub Pages.
- [ ] Commit: `feat: wave 7 completion — full syllabus coverage`

---

## Self-review checklist (before wave completion)

1. **Spec coverage:** every spec §5 topic has a module; every spec §4 component exists in registry or was added; spec §7 question modes all used; spec §8 UX features wired (themes, shortcuts, palette, reduced motion, session replay, analytics, journey, recorder); spec §6 edge types present in graphData.
2. **No placeholders:** every topic file fully implemented — no `TODO`, no `as any` hacks in simulation math, no unimplemented views referenced.
3. **Type consistency:** new components conform to `ViewProps`; topic modules conform to `TopicModule`; new registry functions match their declarations; `testCases` use the same `expect` shape as Wave 0.
4. **Determinism:** every simulation seeded; `npx vitest run src/test/runTestCases.test.ts` passes with all topics registered.
5. **Performance:** no topic exceeds 2000 snapshots default; heavy topics (backprop, pca) use lazy compute + telemetry check.
