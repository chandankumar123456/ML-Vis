# GATE ML Visualizer — Design Specification

**Date:** 2026-08-01
**Status:** Approved (pending user review of written spec)
**Version:** 1.0

## 1. Vision

A professional-grade, interactive Machine Learning visualizer for GATE Data Science & AI aspirants. Not a notes app — a tool where students *see the algorithm thinking*: every concept is explored visually, mathematically, geometrically, algorithmically, and through GATE exam practice. The user should understand not only *what* an algorithm does, but *why* it works, how it evolves step-by-step, where it fails, how it relates to other concepts, and how GATE tests it.

**Golden Rule:** This is not a visualization of Machine Learning. This is a visualization of the *thinking process* behind Machine Learning.

## 2. Decisions (confirmed with user)

| Decision | Choice |
|---|---|
| Stack | React 18 + TypeScript (strict) + Vite, custom Canvas 2D engine, D3, KaTeX, Zustand |
| GATE content | Original GATE-style questions authored by us (no copyrighted GO Classes material), data-driven question banks |
| Persistence | Pure static SPA, local-first (localStorage + export/import), no backend |
| Sequencing | Platform-first (Wave 0), then parallel subagent waves filling all topics |
| Simulation | Deterministic step-based engine: `snapshot = f(params, stepIndex)` |

## 3. Architecture

### 3.1 High-level pipeline

```
TopicModule.simulation (pure) → Snapshot[] → Renderer[] → Canvas/SVG/PNG/PDF/GIF
                                      ↓
                        Playback cursor (play/pause/step/prev/speed/scrub/reset)
```

Simulation is fully render-agnostic. Rendering targets plug in via a renderer registry. Every snapshot is precomputed on param change (debounced), so all playback controls are cursor math on an array.

### 3.2 Folder structure

```
src/
  app/           App shell, router, layout, theme, keyboard shortcuts, command palette
  engine/        Deterministic simulation engine, snapshot types, playback, telemetry
  registry/      ViewComponentRegistry, TopicRegistry, domain registration
  bus/           Global Event Bus (cross-panel highlighting)
  visualizers/   Shared rendering components (see §4.1)
  topics/        One folder per topic, each with module.ts + view components
  ui/            Reusable controls: PlaybackBar, Slider, Toggle, Tabs, Heatmap, etc.
  store/         Zustand: progress, bookmarks, analytics, settings, playback, sessions
  data/          Syllabus data, topic metadata, knowledge graph edges, question banks
  resources/     datasets/, images/, icons/, audio/ (shared, lazy-loaded)
  lib/           Math helpers (gradients, linear algebra), question engine, exporters
  test/          Deterministic test runner for topic testCases, unit, component, e2e
```

### 3.3 TopicModule contract

Every topic implements the same interface — guarantees equal depth across all topics and consistent builds by parallel subagents.

```ts
interface TopicModule {
  id: string;
  title: string;
  version: number;                       // semantic version, migrations supported
  migrations?: Record<number, (state) => state>;  // old bookmarks keep working
  metadata: {
    gateWeightage: 'High' | 'Medium' | 'Low';
    difficultyHeatmap: {                 // multi-dimensional difficulty
      conceptual: 1-5; mathematical: 1-5; coding: 1-5;
      visualization: 1-5; gateFrequency: 1-5;
    };
    estimatedHours: number;
    revisionPriority: 'P0' | 'P1' | 'P2' | 'P3';
    examFrequency: 'Every year' | 'Frequent' | 'Occasional' | 'Rare';
    prerequisites: string[];             // topic ids → Prerequisite Graph
    relatedTopics: string[];
    revision: { quick: '5m', standard: '15m', deep: '30m', mastery: '60m' };
  };
  layers: {                              // Concept Layers: progressive reveal
    foundation: ViewRef[];
    core: ViewRef[];
    advanced: ViewRef[];
  };
  views: ViewRef[];                      // pluggable — pick from registry, no empty tabs
  params: ParamSchema[];                 // playground: sliders, toggles, seeds
  simulation: SimulationDef;             // deterministic step function
  formulas: Formula[];                   // formula explorer (+ derivesFrom edges)
  derivations: Derivation[];             // line-by-line animated derivations
  questions: QuestionBankRef;            // references data/questions/<id>.json
  comparisons: Comparison[];             // compare mode
  failureDemos: FailureDemo[];           // failure mode
  mistakes: Mistake[];                   // Mistake Explorer (GATE traps)
  testCases: TestCase[];                 // deterministic math validation
  // lifecycle hooks
  initialize(params): void;              // preprocessing (build dataset, precompute entropy...)
  dispose(): void;                       // free resources, cancel workers
  validateParams(p): string[];           // violations, e.g. "k must be < n_samples"
  exportState(): SessionBundle;          // for bookmarks, session replay, migration
}
```

### 3.4 SimulationDef

```ts
interface SimulationDef {
  step(params, state): SimState | null;      // pure, deterministic
  annotate(state): SimState;                 // enrich with narration + explanation
  explain(state): StepExplanation;           // "Explain This Step" payload
}
```

### 3.5 SimState — game-engine-grade snapshot channels

```ts
interface SimState {
  algorithm: Record<string, number | string>;  // weight=2.3, gradient=-1.8, lr=0.1
  visuals: VisualCommand[];                    // point positions, line params, arrows
  math: MathStep[];                            // cost, derivative, active equation refs
  narration: string;                           // "Slope is negative. Move right."
  explanation: StepExplanation;                // changed / why / formulaRef / dependsOn / gateConcepts
  highlights: HighlightRef[];                  // {panel, id, intensity} → Event Bus
  metrics: Record<string, number>;             // cost, accuracy, margin...
  events: SimEvent[];                          // {type, label} — reached local min, converged...
  timeline: string[];                          // stage labels for Timeline View
}
```

Every panel (algorithm readout, canvas, equation, narration, metrics) reads the **same snapshot** — synchronized by construction.

### 3.6 Lifecycle

`initialize(params)` → precompute snapshots → render. `validateParams` guards invariants before compute. `dispose` cleans up. Param change → debounced recompute → cursor reset (or preserved by step continuity where sensible).

### 3.7 Global Event Bus

Cross-panel interaction via semantic ids, not DOM nodes:

```
hover Matrix cell (2,3) → bus.emit('highlight', {panel:'matrix', id:'w23'})
  → equation panel highlights w₂₃
  → network edge highlights
  → loss curve highlights
```

Panels subscribe to `highlight:weight:2,3`-style topics. Works across canvas/SVG/KaTeX.

### 3.8 Error handling

Simulations run in a sandbox. Any exception / NaN / Inf is captured and rendered as a "the algorithm broke here" overlay with the failing step index — doubling as Failure Mode pedagogy. App never crashes. All simulation code wrapped; errors reported to telemetry.

### 3.9 Versioning & migrations

`version` per module + `migrations` map. Saved bookmarks/sessions exported with version; on load, migrated forward. Old data always works.

### 3.10 Performance telemetry

Engine records per run: snapshot count, generation time (ms), memory estimate (bytes), render time, FPS. Dev overlay + stored metrics. Used to optimize heavy topics (Backpropagation, PCA, Neural Networks).

### 3.11 Domain-agnostic engine

The engine, registry, graph layer, and question system are a generic **simulation platform**. Machine Learning is the first domain. Future domains reserved: Probability, Linear Algebra, Optimization, Statistics, Deep Learning, Reinforcement Learning. Domain = a registered namespace with its own topics, resources, and graph.

## 4. View Component Registry

Pluggable components; topics pick what they need (no forced empty tabs).

| Registry ID | Purpose |
|---|---|
| `scatter-plot` | Data points, regression lines, residual/projection lines, pan/zoom |
| `loss-surface` | 2D loss curve + animated optimizer trail |
| `loss-landscape` | 3D-perspective surface (bowl, saddle, ridges) |
| `decision-boundary` | Class regions, margins, support vectors |
| `matrix-animator` | Animated matmul: dimensions, broadcasting, cell dot products, shape changes |
| `nn-inspector` | Neural net: activations, weights, biases, gradients, dead neurons, vanish/explode |
| `tree-builder` | Decision tree growth, entropy/gini bars per split |
| `dendrogram` | Hierarchical clustering dendrogram + distance matrix |
| `cluster-animator` | K-means centroid movement, assignments, loss per iteration |
| `eigenviewer` | PCA: rotation, projection, reconstruction, explained variance |
| `knn-animator` | Decision regions by k, distance rings |
| `bayes-viewer` | Priors, likelihoods, posterior fields |
| `confusion-explorer` | TP/FP/TN/FN, precision/recall/F1 animations |
| `roc-viewer` | ROC sweep, threshold movement, AUC fill |
| `curve-comparator` | Train/val error, bias-variance decomposition |
| `formula-explorer` | Symbol meanings, dimensions, assumptions, derivation links, failure cases |
| `derivation-player` | Line-by-line animated derivation with justifications |
| `question-player` | GATE question with animated step-by-step solution (5 modes) |
| `compare-view` | Side-by-side geometry/math/complexity of 2+ algorithms |
| `failure-view` | "Break it" playground: outliers, collinearity, imbalance, noise |
| `timeline-view` | Algorithm evolution roadmap (Init → Prediction → Loss → Gradient → Update → Converge) |
| `equation-graph` | Equation dependency graph ("where did this equation come from?") |
| `feature-space` | Feature space viewer (LR, SVM, PCA, LDA, KNN) |
| `optimizer-view` | GD / SGD / Mini-batch / Momentum / Adam (extensible) |
| `distribution-view` | Probability distributions (Naive Bayes, MLE, Gaussian, LDA) |
| `activation-view` | Activation function explorer (Perceptron, Logistic, NN) |
| `knowledge-graph` | Global concept graph (D3 force, zoom/pan, click-to-open) |
| `concept-explorer` | Concept-first navigation: Sigmoid → odds → log-odds → CE → MLE → gradient → backprop |
| `learning-journey` | Curated progression roadmap with "you are here" |
| `prereq-view` | Prerequisites, dependencies, difficulty, weightage, time |
| `mistake-view` | Mistake Explorer: common errors, why they're wrong, GATE trap flags |
| `recorder` | Export run to GIF / MP4 (WebCodecs) / PNG sequence / PDF |

## 5. Topic Inventory (complete syllabus coverage)

Every module gets the full ecosystem: Concept, Math, Geometry, Optimization, Matrix, Simulation, Playground, Formulas, Derivations, GATE mode, Comparison, Failure, Performance, Mistakes, Timeline, Explain-This-Step, Record.

1. **introduction** — ML in layman terms, learning paradigms, when ML fails
2. **optimization-foundations** — calculus view of gradient descent (cross-module)
3. **simple-linear-regression** — OLS, best-fit line, residual geometry (optimization + linear algebra views)
4. **multiple-linear-regression** — normal equation, design matrix, multicollinearity intro
5. **gradient-descent** — batch / mini-batch / SGD, learning rate, convergence, local minima
6. **polynomial-regression** — basis expansion, degree slider, over/underfitting link
7. **overfitting-underfitting** — train/val split, capacity, error decomposition
8. **ridge-regression** — L2 penalty, shrinkage, bias-variance tradeoff
9. **lasso-regression** — L1 penalty, sparsity, vs ridge geometry (diamond vs circle)
10. **logistic-regression** — sigmoid, decision boundary, probability, log-odds
11. **cross-entropy-loss** — CE derivation, MLE connection, class imbalance behavior
12. **mle** — Maximum Likelihood Estimate, likelihood vs probability, CE link
13. **softmax-regression** — multiclass, categorical CE, gradient derivation
14. **knn** — distance metrics, k effects, decision regions, curse of dimensionality
15. **naive-bayes** — priors/likelihoods/posteriors, independence assumption, smoothing
16. **svm-hard-margin** — hyperplane, margin, support vectors, convex optimization
17. **svm-soft-margin** — slack variables, C parameter, hinge loss
18. **cross-validation** — LOO, k-fold, bias/variance of estimates, stratification
19. **classification-metrics** — precision, recall, F1, confusion matrix, thresholds
20. **roc-auc** — ROC curve, AUC, threshold sweep, TPR/FPR
21. **bias-variance** — decomposition, tradeoff curves, irreducible error
22. **perceptron** — linear separability, convergence, learning rule, vs SGD
23. **pca** — variance maximization, eigenvectors, two derivations, reconstruction
24. **pca-svd** — SVD relation, eigenvalues, low-rank approximation
25. **decision-trees** — entropy, information gain, splits, overfitting/pruning
26. **decision-trees-regression** — Gini impurity, regression trees, CART
27. **kmeans** — loss function, initialization, empty clusters, elbow
28. **hierarchical-clustering** — agglomerative/divisive, linkage, dendrogram
29. **neural-networks** — FFN, activations, forward pass, universal approximation
30. **backpropagation** — chain rule, gradient flow, vanish/explode, weight init
31. **lda** — Fisher's LDA, between/within scatter, projection direction, DR view

## 6. Knowledge Graph & Concept Explorer

One graph data structure (`data/graph.ts`), two views (Knowledge Graph + Concept Explorer + Learning Journey all derive from it).

**Nodes:** ~31 topic nodes + concept nodes (Sigmoid, Entropy, Eigenvalue, Margin, ...).

**Edge types:**
- `requires` — prerequisite edges (Linear Algebra → Projection → OLS → Linear Regression → GD → Regularization → Logistic Regression → CE → MLE → Softmax → NN → Backprop)
- `related` — comparison edges (Ridge↔LASSO, SVM↔Perceptron, PCA↔LDA, KNN↔NaiveBayes)
- `extends` — (PCA → PCA-SVD, SVM hard → soft)
- `derives-from` — (Cross Entropy → MLE)
- `contrasts-with` — (L1 vs L2)
- `frequently-confused` — (Precision↔Recall, Bias↔Variance, PCA↔LDA, SVM↔Perceptron)
- `hidden-gate-link` — cross-module dashed edges (Linear Algebra → Eigenvalues → PCA → SVD; Probability → MLE → CE → Logistic Regression; Calculus → Gradient → Backprop → NN)

**Rendering:** D3 force layout, zoom/pan, node size = GATE weightage, color = category, hover tooltip = "why this connection", click = open topic. Edge data also drives per-topic "related topics" chips and the Learning Journey.

## 7. GATE Exam Mode

### 7.1 Question formats (data-driven)

Banks live in `data/questions/<topic>.json` — structured JSON, not embedded code. Each question may reference simulation states (`snapshot refs`) so solutions animate via the same deterministic engine. Supports import/export, community contribution, and automated validation (CI).

Five modes:
1. Previous-GATE-style questions (original, mirroring patterns)
2. Conceptual MCQs
3. Numerical Answer Type (NAT)
4. Matrix-based problems (interactive solution)
5. Visual reasoning questions (read the plot)

### 7.2 Question schema

```json
{
  "id": "gd-004",
  "mode": "nat",
  "prompt": "...",
  "options": [...],            // for MCQ
  "answer": 0.42,
  "tolerance": 0.01,           // NAT grading
  "animatedSolution": { "script": "gd-convergence", "params": {...} },
  "trapExplanations": { "B": "Students pick B because they forget the negative sign in ∂L/∂w" },
  "concepts": ["gradient-descent", "learning-rate", "derivative"],
  "difficulty": 3,
  "tags": ["trap", "indirect", "formula", "numerical"]
}
```

### 7.3 Practice / Exam modes

- Practice mode: drills with instant feedback + explanations
- Exam mode: timed GATE-style paper (30 questions / 100-mark format), scored, analytics recorded
- Trap detection: every wrong option explains "why students pick this"
- Indirect questions: cross-topic problems with linked "concepts involved" chain

## 8. UX System

- Light/dark themes (CSS variables, persisted, instant)
- Colorblind-safe palettes (deuteranopia / protanopia / tritanopia) — settings toggle
- Reduced-motion mode: snapshot jumps instead of interpolation
- Keyboard shortcuts: `Space` play/pause · `←/→` step · `R` reset · `1-9` view jump · `F` formulas · `S` search · `?` cheat sheet
- Keyboard-only navigation; screen-reader-friendly math descriptions (LaTeX → speech text) where practical
- Global search (command palette, fuzzy): topics, concepts, formulas
- Progress tracking: per-topic completion %, mastered/bookmarked flags
- Learning analytics: topics completed, time per topic, questions attempted, weakest concepts, most-revisited simulations → revision suggestions
- Learning Journey: curated visual roadmap with "you are here" markers
- Session replay: `SessionBundle { params, step, activeView, bookmarks, timestamp }` saved to localStorage, one-click resume
- Revision mode: 5/15/30/60-min curated paths per topic
- Recording: GIF / MP4 / PNG sequence / PDF export per run
- Responsive: adaptive panels; stacked single-pane on mobile; zoom/pan on all plots

## 9. Resources Layer

```
resources/
  datasets/    iris, wine, synthetic generators, custom CSV loader
  images/      icons, illustrations
  audio/       optional narration
```

Topics reference datasets by id. Lazy loading + caching. Custom CSV upload → any topic consumes it.

## 10. Build Roadmap (platform-first, wave-fill)

- **Wave 0 — Foundation** (engine + shell + reference topics): Vite scaffold, engine core (snapshot/step/events/timeline), registry + lifecycle hooks, playback UI, event bus, theming, store (progress/bookmarks/analytics/sessions), router shell, command palette, knowledge graph base, question engine, telemetry. Reference topics: **gradient-descent** + **simple-linear-regression** proving every subsystem end-to-end. Includes testCases, mistake explorer, explain-this-step.
- **Wave 1 — Regression cluster** (4): multiple-linear-regression, polynomial-regression, ridge-regression, lasso-regression
- **Wave 2 — Classification cluster** (5): logistic-regression, cross-entropy-loss, mle, softmax-regression, knn
- **Wave 3 — Geometry cluster** (4): svm-hard-margin, svm-soft-margin, perceptron, lda
- **Wave 4 — Dim-reduction cluster** (3): pca, pca-svd, naive-bayes (moved here so it shares the `distribution-view` component built in this wave with LDA)
- **Wave 5 — Trees & clustering** (4): decision-trees, decision-trees-regression, kmeans, hierarchical-clustering
- **Wave 6 — Deep learning cluster** (3): neural-networks, backpropagation, optimization-foundations
- **Wave 7 — Evaluation & meta** (6): cross-validation, classification-metrics, roc-auc, bias-variance, overfitting-underfitting, introduction

Each wave: 3-5 parallel subagents (frontend-engineer per topic + ui-ux polish + qa validation of testCases).

## 11. Testing & QA

- Deterministic unit tests: every topic's `testCases` verified (math correctness, convergence, edge cases) — `npm test`
- Engine tests: playback, scrubbing, param-change recompute, error sandbox, determinism (same params → same snapshots)
- Component tests (Vitest + Testing Library) for registry components
- E2E smoke (Playwright): load each topic, play/pause/step, change params, assert no console errors
- Question bank validation (CI): schema check, answer plausibility, solution script references
- QA subagent reviews each wave before merge; scrutiny-agent review for overengineering

## 12. Performance

- Route-level lazy loading (each topic = async chunk)
- Canvas pooling, snapshot caching, `requestIdleCallback` precompute
- 60 FPS with snapshot interpolation; reduced-motion fallback
- Telemetry-driven optimization for heavy topics (Backprop, PCA, NN)

## 13. Deployment

- Static SPA → GitHub Pages / Vercel / Netlify, one-command deploy
- No secrets, no backend, offline-friendly
- Export/import of progress for backup

## 14. Out of Scope (this phase)

- User accounts / cloud sync (architecture allows later)
- Audio narration content (infrastructure only)
- Future domains beyond ML (engine designed for them, not built)
