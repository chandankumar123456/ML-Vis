// ===== Generic param system =====
export type ParamValue = number | string | boolean;
export type Params = Record<string, ParamValue>;

export interface ParamSchema {
  id: string;
  label: string;
  type: 'number' | 'select' | 'toggle' | 'seed';
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
  default: ParamValue;
  group?: string;
}

// ===== SimState — game-engine-grade snapshot =====
export interface VisualCommand {
  type: string;               // 'point' | 'line' | 'arrow' | 'region' | 'matrix' | ...
  id?: string;                // semantic id for highlighting
  [key: string]: unknown;
}

// ===== Eigenviewer visual commands =====
// Consumed by the 'eigenviewer' registry view. angle is the direction of the
// candidate axis (radians, measured from the +x axis, through the data
// centroid); topics animate rotation by re-emitting snapshots with new angles.
export interface AxisCommand extends VisualCommand {
  type: 'axis';
  angle: number;              // radians
  color?: string;
}

// A data point projected onto the candidate axis. `onto` is the orthogonal
// projection of `point` on the axis line; `residual` is the optional
// perpendicular distance (reconstruction error) if the topic pre-computed it.
export interface ProjectionCommand extends VisualCommand {
  type: 'projection';
  point: [number, number];
  onto: [number, number];
  residual?: number;
  color?: string;
}

// ===== Gaussian class density =====
// A univariate Gaussian used by the 'distribution-view' registry view
// (naive-bayes per-feature densities, mle likelihood curves, lda class PDFs).
export interface Distribution {
  label: string;
  mean: number;
  variance: number;
  color?: string;
}

export interface MathStep {
  latex: string;
  caption?: string;
  id?: string;                // formula id for equation-graph links
}

export interface HighlightRef {
  panel: string;              // 'canvas' | 'equation' | 'matrix' | 'metrics' | ...
  id: string;                 // semantic id e.g. 'weight:2,3'
  intensity: number;          // 0..1
}

export interface SimEvent {
  type: string;               // 'local_min' | 'converged' | 'overflow' | ...
  label: string;
  step: number;               // 1-based: index of the snapshot that raised it
}

export interface StepExplanation {
  changed: string[];          // "weight decreased from 2.3 to 2.12"
  why: string;                // "gradient was positive, so weight moves opposite"
  formulaRef?: string;        // formula id
  dependsOn: string[];        // prerequisite concept ids
  gateConcepts: string[];     // GATE topics exercised
}

export interface SimState {
  algorithm: Record<string, ParamValue>;
  visuals: VisualCommand[];
  math: MathStep[];
  narration: string;
  explanation: StepExplanation;
  highlights: HighlightRef[];
  metrics: Record<string, number>;
  events: SimEvent[];
  timeline: string[];
}

// ===== Simulation definition =====
export interface SimulationDef {
  initialState(params: Params): SimState;
  /** Pure. Return null to signal termination (convergence/failure). */
  step(params: Params, state: SimState): SimState | null;
}

// ===== Test cases =====
// Semantics for the centralized test runner (src/test/runTestCases):
// - maxSteps defaults to 500. The run terminates when step() returns null OR
//   when snapshots reach maxSteps (then telemetry.failedAtStep = maxSteps, "step budget").
// - converged: true → expect failedAtStep undefined; false → expect defined.
//   If unset, the run may terminate either way.
export interface TestCase {
  name: string;
  params: Params;
  maxSteps?: number;
  expect: {
    finalMetrics?: Record<string, number | ((v: number) => boolean)>;
    finalAlgorithm?: Record<string, ParamValue | ((v: ParamValue) => boolean)>;
    eventLabels?: string[];
    converged?: boolean;      // null termination expected
  };
}

// ===== Content blocks =====
export interface Formula {
  id: string;
  latex: string;
  symbols: { symbol: string; meaning: string; units?: string; dimensions?: string }[];
  assumptions: string[];
  derivationIds?: string[];
  derivesFrom?: string[];     // formula ids — Equation Dependency Graph
  failureCases: string[];
  connections: string[];      // "connects to X formula"
  whyWorks: string;
}

export interface DerivationStep {
  latex: string;
  justification: string;
}

export interface Derivation {
  id: string;
  title: string;
  steps: DerivationStep[];
  derivedFrom?: string[];
}

export interface Question {
  id: string;
  mode: 'gate-mcq' | 'conceptual-mcq' | 'nat' | 'matrix' | 'visual';
  prompt: string;
  options?: string[];
  answer: string | number | (string | number)[];   // matrix mode: arrays (match two columns)
  tolerance?: number;                 // NAT
  explanation: string;                // why correct
  trapExplanations?: Record<string, string>;  // option letter -> why wrong
  animatedSolution?: {                // refs a sim run to animate
    simId?: string;                   // 'gradient-descent'
    params?: Params;
    steps?: number[];
  };
  concepts: string[];
  difficulty: 1 | 2 | 3 | 4 | 5;
  tags: string[];                     // 'trap' | 'indirect' | 'formula' | 'numerical' | 'matrix' | 'visual'
}

export interface Comparison {
  id: string;
  title: string;
  topics: string[];                   // topic ids
  axes: { axis: string; entries: { topic: string; value: string }[] }[];
  notes: string[];
}

export interface FailureDemo {
  id: string;
  title: string;
  scenario: string;                   // 'outliers' | 'collinearity' | ...
  params: Params;
  narration: string;
  whyItBreaks: string;
}

export interface Mistake {
  id: string;
  pattern: string;                    // "Using wrong sign in ∂L/∂w"
  example?: string;                   // LaTeX
  whyWrong: string;
  gateTrap: boolean;
  relatedConcept?: string;
}

// ===== Session bundle =====
export interface SessionBundle {
  topicId: string;
  moduleVersion: number;
  params: Params;
  step: number;
  activeView: string;
  bookmarks: string[];
  savedAt: string;
}

// ===== TopicModule =====
// Layer membership is the SINGLE source of truth via TopicModule.layers buckets
// (foundation/core/advanced). ViewRef intentionally has no layer field to prevent drift.
export interface ViewRef {
  slot: 'primary' | 'secondary' | 'sidebar';
  component: string;                  // registry id
  title: string;
}

export interface TopicMetadata {
  gateWeightage: 'High' | 'Medium' | 'Low';
  difficultyHeatmap: {
    conceptual: number; mathematical: number;
    coding: number; visualization: number; gateFrequency: number;
  };
  estimatedHours: number;
  revisionPriority: 'P0' | 'P1' | 'P2' | 'P3';
  examFrequency: 'Every year' | 'Frequent' | 'Occasional' | 'Rare';
  prerequisites: string[];
  relatedTopics: string[];
  revision: { quick: string; standard: string; deep: string; mastery: string };
}

export interface TopicModule {
  id: string;
  title: string;
  version: number;
  migrations?: Record<number, (bundle: SessionBundle) => SessionBundle>;
  metadata: TopicMetadata;
  layers: { foundation: ViewRef[]; core: ViewRef[]; advanced: ViewRef[] };
  params: ParamSchema[];
  simulation: SimulationDef;
  formulas: Formula[];
  derivations: Derivation[];
  questions: Question[];
  comparisons: Comparison[];
  failureDemos: FailureDemo[];
  mistakes: Mistake[];
  testCases: TestCase[];
  // metric key the topic's loss-curve view plots
  lossMetricKey?: string;
  // optional second metric series (train vs test): enables two-line or bar rendering
  lossMetricKey2?: string;
  // lifecycle hooks
  initialize?(params: Params): void;
  dispose?(): void;
  validateParams?(params: Params): string[];
  exportState?(): SessionBundle;
}

// ===== Run telemetry =====
export interface RunTelemetry {
  snapshotCount: number;
  genMs: number;
  memBytes: number;
  failedAtStep?: number;
  failureReason?: string;
}

export interface SnapshotRun {
  params: Params;
  snapshots: SimState[];
  telemetry: RunTelemetry;
}

// ===== Wave-5 tree / clustering visual commands =====
// Consumed by the 'tree-builder', 'cluster-animator' and 'dendrogram' registry
// views. Units are documented per field — normalized vs world space matters:
// each view consumes the same coordinate very differently.

// A decision-tree node for the 'tree-builder' view. x/y are NORMALIZED
// fractions of the drawing area (the topic computes the layout; the view
// scales it to its container). Linkage is explicit: `children` lists child
// node ids — topics emit NO separate edge command. Duplicate ids: last wins.
export interface NodeCommand extends VisualCommand {
  type: 'node';
  x: number;                // normalized [0,1] of the drawing area width
  y: number;                // normalized [0,1] of the drawing area height
  label: string;            // feature-test or class label shown beside the node
  splitInfo?: string;       // optional split summary under the label
  children?: string[];      // child node ids (explicit linkage)
  purity?: number;          // 0..1 — entropy/gini bar fill fraction (topic computes)
  color?: string;
  className?: string;
}

// A 2-D data point for the 'cluster-animator' view. x/y are WORLD units (px
// independent — the view fits bounds like scatter-plot).
export interface PointCommand extends VisualCommand {
  type: 'point';
  x: number;
  y: number;
  color?: string;
}

// A k-means centroid for the 'cluster-animator' view. x/y are WORLD units.
// Topics MUST keep ids stable across snapshots: the view records each
// centroid's position after every draw and renders a faint trail at the
// PREVIOUS snapshot's positions (the "animated convergence" visual).
export interface CentroidCommand extends VisualCommand {
  type: 'centroid';
  x: number;
  y: number;
  color?: string;
}

// Assignment of a point to a centroid for the 'cluster-animator' view. The
// view draws a thin line from `point` to the centroid with matching
// centroidId and colors the point with the centroid's color.
export interface AssignmentCommand extends VisualCommand {
  type: 'assignment';
  point: [number, number];  // world units
  centroidId: string;       // must match a CentroidCommand id
  color?: string;
}

// A free-form text readout (used as the 'cluster-animator' loss-caption
// fallback when the snapshot carries no loss/cost/j metric).
export interface TextCommand extends VisualCommand {
  type: 'text';
  text: string;
  color?: string;
}

// An agglomerative merge for the 'dendrogram' view. `height` is in distance
// units (world space; the view auto-ranges its distance axis from the max
// height). `children` reference leaf ids OR ids of previously-emitted merges
// (topics emit merges in chronological order; the view lays them out
// bottom-up). Unknown child ids render as dangling stubs; cyclic references
// are guarded against hanging.
export interface MergeCommand extends VisualCommand {
  type: 'merge';
  height: number;           // distance units, non-decreasing across the run
  children: string[];       // leaf ids or earlier merge ids
}
