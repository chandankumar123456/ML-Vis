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
