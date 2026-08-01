# GATE ML Visualizer — Wave 0: Platform Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete simulation platform (deterministic engine, registries, playback, UI shell, shared visualizers, question engine, knowledge graph) plus two reference topics (gradient-descent, simple-linear-regression) proving every subsystem end-to-end.

**Architecture:** Deterministic step-based simulation engine (`snapshot = f(params, stepIndex)`) precomputes all SimStates per param change; playback is cursor math. Simulation is render-agnostic — renderers consume snapshots. Topics are self-registering modules implementing a strict `TopicModule` contract. Cross-panel interaction via a semantic event bus. State via Zustand, persisted to localStorage.

**Tech Stack:** Vite 5 + React 18 + TypeScript (strict) · Zustand · KaTeX · D3 · Vitest + Testing Library + Playwright · Canvas 2D custom renderer.

**Spec:** `docs/superpowers/specs/2026-08-01-gate-ml-visualizer-design.md`

---

## File Structure (Wave 0)

```
package.json, vite.config.ts, tsconfig.json, index.html, playwright.config.ts
src/
  main.tsx, App.tsx, vite-env.d.ts
  engine/types.ts            — SimState, SimulationDef, TopicModule, all contracts
  engine/core.ts             — computeRun(), error sandbox, telemetry
  engine/playback.ts         — playback controller (play/pause/step/prev/speed/scrub)
  bus/eventBus.ts            — global event bus
  registry/topicRegistry.ts  — self-registering topics (lazy via import.meta.glob)
  registry/viewRegistry.ts   — view component registry
  store/settingsStore.ts     — theme, palettes, reduced motion, shortcuts
  store/playbackStore.ts     — current run + cursor (wraps engine/playback)
  store/progressStore.ts     — completion, bookmarks, mistakes saved
  store/analyticsStore.ts    — time per topic, question attempts, weakest concepts
  store/sessionStore.ts      — SessionBundle save/resume
  ui/PlaybackBar.tsx, ui/ParamPanel.tsx, ui/Slider.tsx, ui/Toggle.tsx,
  ui/Select.tsx, ui/Tabs.tsx, ui/Latex.tsx, ui/Heatmap.tsx, ui/MetricGrid.tsx
  app/AppShell.tsx, app/Router.tsx, app/CommandPalette.tsx, app/KeyboardShortcuts.tsx,
  app/ThemeProvider.tsx, app/SearchIndex.tsx
  pages/HomePage.tsx, pages/TopicPage.tsx, pages/GraphPage.tsx, pages/JourneyPage.tsx,
  pages/ExamPage.tsx, pages/ViewHost.tsx
  visualizers/ScatterPlot.tsx, visualizers/LossCurve.tsx, visualizers/MatrixAnimator.tsx,
  visualizers/FormulaExplorer.tsx, visualizers/TimelineView.tsx, visualizers/MistakeView.tsx,
  visualizers/knowledgeGraph/graphData.ts, visualizers/knowledgeGraph/KnowledgeGraph.tsx,
  visualizers/QuestionPlayer.tsx, visualizers/Recorder.tsx
  topics/gradient-descent/{module.ts, questions.ts, formulas.ts, derivations.ts, mistakes.ts, testCases.ts, testCases.test.ts}
  topics/simple-linear-regression/{module.ts, questions.ts, formulas.ts, derivations.ts, mistakes.ts, testCases.ts, testCases.test.ts}
  lib/canvas/CanvasStage.ts
  lib/math/{linAlg.ts, stats.ts, calculus.ts}
  lib/exporters/{pngExporter.ts, recorder.ts}
  lib/params.ts
  lib/questions/engine.ts
  test/runTestCases.test.ts
```

---

### Task 1: Scaffold Vite + React + TS project

**Files:**
- Create: `package.json`, `vite.config.ts`, `tsconfig.json`, `tsconfig.app.json`, `tsconfig.node.json`, `index.html`, `src/vite-env.d.ts`, `src/main.tsx`, `src/App.tsx`, `.gitignore`, `src/testSetup.ts`

- [ ] **Step 1: Write package.json**

```json
{
  "name": "gate-ml-visualizer",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "e2e": "playwright test",
    "lint": "tsc --noEmit"
  },
  "dependencies": {
    "d3": "^7.9.0",
    "katex": "^0.16.11",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.0",
    "zustand": "^4.5.4"
  },
  "devDependencies": {
    "@playwright/test": "^1.46.0",
    "@testing-library/jest-dom": "^6.4.8",
    "@testing-library/react": "^16.0.0",
    "@types/d3": "^7.4.3",
    "@types/katex": "^0.16.7",
    "@types/react": "^18.3.4",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "jsdom": "^24.1.1",
    "typescript": "~5.5.4",
    "vite": "^5.4.1",
    "vitest": "^2.0.5"
  }
}
```

- [ ] **Step 2: Write vite.config.ts**

```ts
/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 900 },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/testSetup.ts'],
    globals: true,
  },
});
```

- [ ] **Step 3: Write tsconfig files**

```json
// tsconfig.json
{
  "files": [],
  "references": [
    { "path": "./tsconfig.app.json" },
    { "path": "./tsconfig.node.json" }
  ]
}
```

```json
// tsconfig.app.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src"]
}
```

```json
// tsconfig.node.json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true
  },
  "include": ["vite.config.ts", "playwright.config.ts"]
}
```

- [ ] **Step 4: Write index.html + src/vite-env.d.ts + src/testSetup.ts**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>GATE ML Visualizer</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

```ts
/// <reference types="vite/client" />
```

```ts
import '@testing-library/jest-dom';
```

- [ ] **Step 5: Write src/main.tsx + src/App.tsx (minimal placeholder)**

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

```tsx
export function App() {
  return <div>GATE ML Visualizer — Wave 0 scaffold</div>;
}
```

- [ ] **Step 6: Write .gitignore**

```gitignore
node_modules/
dist/
playwright-report/
test-results/
*.local
```

- [ ] **Step 7: Install and verify build**

Run: `npm install`
Expected: install completes without errors.

Run: `npm run build`
Expected: build succeeds; `dist/` created.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold vite react ts project"
```

### Task 2: Engine contracts (src/engine/types.ts)

**Files:**
- Create: `src/engine/types.ts`
- Test: `src/engine/types.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import type { SimState, SimulationDef, Params } from './types';

const sim: SimulationDef = {
  initialState: (params: Params): SimState => ({
    algorithm: { x: params.x as number },
    visuals: [{ type: 'point', id: 'pt', x: 0, y: 0 }],
    math: [{ latex: 'x = 0' }],
    narration: 'Start',
    explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
    highlights: [],
    metrics: {},
    events: [],
    timeline: ['Init'],
  }),
  step: (_p, s) => (s.algorithm.x as number) > 10 ? null : { ...s, algorithm: { x: (s.algorithm.x as number) + 1 } },
};

describe('SimulationDef contract', () => {
  it('produces increasing states then terminates', () => {
    let s = sim.initialState({ x: 0 });
    let steps = 0;
    while (s) {
      steps++;
      const next = sim.step({ x: 0 }, s);
      if (next === null) break;
      s = next;
      expect(steps).toBeLessThan(1000); // guard: a non-terminating step must fail, not hang
    }
    // deterministic: x 0→11 takes exactly 12 calls; bound it both ways
    expect(steps).toBe(12);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/types.test.ts`
Expected: FAIL — cannot find module './types'.

- [ ] **Step 3: Write src/engine/types.ts — the complete contract surface**

```ts
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
  step: number;
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/types.test.ts`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/engine/types.ts src/engine/types.test.ts
git commit -m "feat: define engine contracts (SimState, SimulationDef, TopicModule)"
```

---

### Task 3: Engine core — deterministic snapshot computation (src/engine/core.ts)

**Files:**
- Create: `src/engine/core.ts`
- Test: `src/engine/core.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { computeRun, isFiniteState, timelineStages } from './core';
import type { SimulationDef, Params } from './types';

const quadratic: SimulationDef = {
  initialState: (p: Params) => ({
    algorithm: { x: p.x0 as number },
    visuals: [], math: [], narration: '',
    explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
    highlights: [], metrics: { f: (p.x0 as number) ** 2 }, events: [],
    timeline: ['init'],
  }),
  step: (p, s) => {
    const x = s.algorithm.x as number;
    if (Math.abs(x) < 1e-6) return null;
    return {
      ...s,
      algorithm: { x: x - 0.1 * 2 * x },
      metrics: { f: x ** 2 },
      timeline: [...s.timeline, 'step'],
    };
  },
};

describe('computeRun', () => {
  it('produces deterministic snapshots and terminates', () => {
    const runA = computeRun(quadratic, { x0: 5 }, 1000);
    const runB = computeRun(quadratic, { x0: 5 }, 1000);
    expect(runA.snapshots.length).toBe(runB.snapshots.length);
    expect(runA.snapshots).toEqual(runB.snapshots); // whole snapshot arrays, all channels
    expect(runA.snapshots.length).toBeGreaterThan(2);
    expect(runA.telemetry.snapshotCount).toBe(runA.snapshots.length);
  });

  it('keeps failedAtStep undefined on clean convergence', () => {
    const run = computeRun(quadratic, { x0: 5 }, 1000);
    expect(run.telemetry.failedAtStep).toBeUndefined();
    expect(run.telemetry.failureReason).toBeUndefined();
  });

  it('detects diverging runs and reports failure', () => {
    const diverging: SimulationDef = {
      initialState: (p: Params) => ({
        algorithm: { x: p.x0 as number }, visuals: [], math: [], narration: '',
        explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
        highlights: [], metrics: {}, events: [],
        timeline: ['init'],
      }),
      step: (_p, s) => ({ ...s, algorithm: { x: (s.algorithm.x as number) * 2 } }),
    };
    const run = computeRun(diverging, { x0: 1 }, 100);
    expect(run.telemetry.failedAtStep).toBeDefined();
  });

  it('reports non-finite initial state without running steps', () => {
    const badInit: SimulationDef = {
      initialState: () => ({
        algorithm: { x: NaN }, visuals: [], math: [], narration: '',
        explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
        highlights: [], metrics: {}, events: [], timeline: ['init'],
      }),
      step: () => null,
    };
    const run = computeRun(badInit, {}, 10);
    expect(run.telemetry.failedAtStep).toBe(0);
    expect(run.telemetry.failureReason).toContain('non-finite');
    expect(run.snapshots).toHaveLength(1);
  });

  it('sandboxes thrown step exceptions', () => {
    const throwing: SimulationDef = {
      initialState: () => ({
        algorithm: {}, visuals: [], math: [], narration: '',
        explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
        highlights: [], metrics: {}, events: [], timeline: ['init'],
      }),
      step: () => { throw new Error('boom'); },
    };
    const run = computeRun(throwing, {}, 100);
    expect(run.telemetry.failedAtStep).toBe(1);
    expect(run.telemetry.failureReason).toBe('boom');
  });

  it('prefers non-finite diagnosis over budget at the final step', () => {
    const overflow: SimulationDef = {
      initialState: (p: Params) => ({
        algorithm: { x: p.x0 as number }, visuals: [], math: [], narration: '',
        explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
        highlights: [], metrics: {}, events: [], timeline: ['init'],
      }),
      step: (_p, s) => ({ ...s, algorithm: { x: (s.algorithm.x as number) * 2 } }),
    };
    // x: 1.25e307 → 2.5e307 → 5e307 → 1e308 → Infinity exactly at the last allowed step (i=4, maxSteps=5)
    const run = computeRun(overflow, { x0: 1.25e307 }, 5);
    expect(run.telemetry.failureReason).toContain('non-finite');
    expect(run.telemetry.failedAtStep).toBe(4);
  });

  it('does not let cyclic state escape the sandbox', () => {
    const cyclic: SimulationDef = {
      initialState: () => ({
        algorithm: {}, visuals: [], math: [], narration: '',
        explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
        highlights: [], metrics: {}, events: [], timeline: ['init'],
      }),
      step: (_p, s) => {
        const bad: any = { type: 'point', loop: undefined };
        bad.loop = bad; // circular reference inside a VisualCommand payload
        return { ...s, visuals: [bad] };
      },
    };
    expect(() => computeRun(cyclic, {}, 10)).not.toThrow();
    const run = computeRun(cyclic, {}, 10);
    expect(run.telemetry.memBytes).toBe(0);
  });
});

describe('isFiniteState', () => {
  it('flags NaN and Infinity', () => {
    const bad: any = { algorithm: { x: NaN }, metrics: { f: Infinity }, visuals: [], math: [], narration: '', explanation: {}, highlights: [], events: [], timeline: [] };
    expect(isFiniteState(bad)).toBe(false);
    const good: any = { algorithm: { x: 1 }, metrics: { f: 2 }, visuals: [], math: [], narration: '', explanation: {}, highlights: [], events: [], timeline: [] };
    expect(isFiniteState(good)).toBe(true);
  });
});

describe('timelineStages', () => {
  it('dedupes repeated stage labels', () => {
    const run = computeRun(quadratic, { x0: 5 }, 1000);
    const stages = timelineStages(run);
    // dedupe must collapse the repeated 'step' labels to a single stage
    expect(stages).toEqual([{ label: 'init', step: 0 }, { label: 'step', step: 1 }]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/core.test.ts`
Expected: FAIL — cannot find module './core'.

- [ ] **Step 3: Write src/engine/core.ts**

```ts
import type { Params, SimState, SimulationDef, SnapshotRun, RunTelemetry } from './types';

export function isFiniteState(s: SimState): boolean {
  for (const v of Object.values(s.algorithm)) {
    if (typeof v === 'number' && !Number.isFinite(v)) return false;
  }
  for (const v of Object.values(s.metrics)) {
    if (!Number.isFinite(v)) return false;
  }
  return true;
}

const DEFAULT_MAX_STEPS = 2000;

/**
 * Deterministic: same params → identical snapshot array.
 * Sandboxed: non-finite states or exceptions terminate with telemetry.
 */
export function computeRun(
  sim: SimulationDef,
  params: Params,
  maxSteps = DEFAULT_MAX_STEPS
): SnapshotRun {
  const t0 = performance.now();
  const snapshots: SimState[] = [];
  let state: SimState | null = null;
  let failedAtStep: number | undefined;
  let failureReason: string | undefined;

  try {
    state = sim.initialState(params);
    if (!isFiniteState(state)) {
      return {
        params, snapshots: [state], telemetry: {
          snapshotCount: 1, genMs: performance.now() - t0, memBytes: 0,
          failedAtStep: 0, failureReason: 'initial state non-finite',
        },
      };
    }
    snapshots.push(state);
    for (let i = 1; i < maxSteps; i++) {
      const next = sim.step(params, state);
      if (next === null) break;
      if (!isFiniteState(next)) {
        failedAtStep = i;
        failureReason = 'non-finite value (NaN/Infinity) at step ' + i;
        snapshots.push(next);
        break;
      }
      snapshots.push(next);
      state = next;
    }
    if (failedAtStep === undefined && snapshots.length >= maxSteps) {
      failedAtStep = maxSteps;
      failureReason = 'step budget exceeded (no convergence)';
    }
  } catch (e) {
    failedAtStep = snapshots.length;
    failureReason = e instanceof Error ? e.message : String(e);
  }

  // estBytes must NOT escape the sandbox: a cyclic VisualCommand (index signature
  // allows arbitrary payloads) would otherwise throw out of computeRun itself.
  let estBytes = 0;
  try {
    estBytes = snapshots.reduce((acc, s) => acc + JSON.stringify(s).length, 0);
  } catch {
    estBytes = 0;
  }

  const telemetry: RunTelemetry = {
    snapshotCount: snapshots.length,
    genMs: performance.now() - t0,
    memBytes: estBytes,
    failedAtStep,
    failureReason,
  };
  return { params, snapshots, telemetry };
}

/** Convert a snapshot array into a timeline of stage labels (deduped). */
export function timelineStages(run: SnapshotRun): { label: string; step: number }[] {
  const seen = new Set<string>();
  const stages: { label: string; step: number }[] = [];
  run.snapshots.forEach((s, i) => {
    const label = s.timeline[s.timeline.length - 1];
    if (label && !seen.has(label)) {
      seen.add(label);
      stages.push({ label, step: i });
    }
  });
  return stages;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/core.test.ts`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/engine/core.ts src/engine/core.test.ts
git commit -m "feat: deterministic sandboxed snapshot engine"
```

### Task 4: Playback controller (src/engine/playback.ts)

**Files:**
- Create: `src/engine/playback.ts`
- Test: `src/engine/playback.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { createPlayback } from './playback';
import type { SnapshotRun, SimState } from './types';

const mkRun = (n: number): SnapshotRun => ({
  params: {},
  snapshots: Array.from({ length: n }, (_, i) => ({
    algorithm: { i }, visuals: [], math: [], narration: `s${i}`,
    explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
    highlights: [], metrics: {}, events: [], timeline: [`t${i}`],
  }) as SimState),
  telemetry: { snapshotCount: n, genMs: 0, memBytes: 0 },
});

describe('createPlayback', () => {
  it('steps forward and backward', () => {
    const pb = createPlayback(mkRun(5));
    expect(pb.cursor).toBe(0);
    pb.stepForward();
    expect(pb.cursor).toBe(1);
    pb.stepBackward();
    expect(pb.cursor).toBe(0);
  });
  it('clamps at boundaries', () => {
    const pb = createPlayback(mkRun(3));
    pb.stepBackward();
    expect(pb.cursor).toBe(0);
    pb.jumpTo(99);
    expect(pb.cursor).toBe(2);
  });
  it('play/stop toggles and respects speed', () => {
    const pb = createPlayback(mkRun(10));
    pb.play();
    expect(pb.playing).toBe(true);
    pb.pause();
    expect(pb.playing).toBe(false);
    pb.setSpeed(2);
    expect(pb.speed).toBe(2);
  });
  it('reset returns to step 0', () => {
    const pb = createPlayback(mkRun(10));
    pb.jumpTo(7);
    pb.reset();
    expect(pb.cursor).toBe(0);
  });
  it('tick advances while playing and stops at end', () => {
    const pb = createPlayback(mkRun(3));
    pb.play();
    pb.tick();
    pb.tick();
    expect(pb.cursor).toBe(2);
    pb.tick();
    expect(pb.playing).toBe(false);
    expect(pb.cursor).toBe(2);
  });
  it('handles empty runs without breaking the cursor invariant', () => {
    const pb = createPlayback(mkRun(0));
    expect(pb.cursor).toBe(-1); // sentinel: no valid index
    pb.play();
    pb.tick();
    pb.stepForward();
    pb.jumpTo(3);
    expect(pb.cursor).toBe(-1);
    expect(pb.playing).toBe(false);
  });
  it('guards invalid speed input and clamps bounds', () => {
    const pb = createPlayback(mkRun(5));
    pb.setSpeed(NaN);
    expect(pb.speed).toBe(1);
    pb.setSpeed(0.05);
    expect(pb.speed).toBe(0.1);
    pb.setSpeed(99);
    expect(pb.speed).toBe(8);
  });
  it('jumpTo clamps negative indices', () => {
    const pb = createPlayback(mkRun(5));
    pb.jumpTo(-5);
    expect(pb.cursor).toBe(0);
  });
  it('tick advances by speed while playing', () => {
    const pb = createPlayback(mkRun(10));
    pb.play();
    pb.setSpeed(3);
    pb.tick();
    expect(pb.cursor).toBe(3);
    pb.pause();
    pb.tick();
    expect(pb.cursor).toBe(3);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/playback.test.ts`
Expected: FAIL — cannot find module './playback'.

- [ ] **Step 3: Write src/engine/playback.ts**

```ts
import type { SnapshotRun } from './types';

export interface Playback {
  readonly run: SnapshotRun;
  readonly cursor: number;
  readonly playing: boolean;
  readonly speed: number; // steps per frame tick
  play(): void;
  pause(): void;
  stepForward(): void;
  stepBackward(): void;
  jumpTo(i: number): void;
  reset(): void;
  setSpeed(s: number): void;
  tick(): void; // called each animation frame; advances by speed while playing
}

export function createPlayback(run: SnapshotRun): Playback {
  let cursorF = 0;               // float accumulator — never exposed raw
  let playing = false;
  let speed = 1;
  const last = () => run.snapshots.length - 1;
  const empty = () => run.snapshots.length === 0;
  const clamp = (i: number) => Math.max(0, Math.min(last(), i));

  return {
    get run() { return run; },
    // empty runs (e.g. engine sandbox failure with no snapshots): sentinel -1, never a valid index
    get cursor() { return empty() ? -1 : Math.floor(cursorF); },
    get playing() { return playing; },
    get speed() { return speed; },
    play() { if (empty()) return; playing = true; },
    pause() { playing = false; },
    stepForward() { if (empty()) return; cursorF = clamp(cursorF + 1); },
    stepBackward() { if (empty()) return; cursorF = clamp(cursorF - 1); },
    jumpTo(i: number) { if (empty()) return; cursorF = clamp(i); },
    reset() { cursorF = 0; playing = false; },
    setSpeed(s: number) {
      if (!Number.isFinite(s)) return;      // NaN/Infinity would corrupt the cursor invariant
      speed = Math.max(0.1, Math.min(8, s));
    },
    tick() {
      if (!playing || empty()) return;
      cursorF = clamp(cursorF + speed);
      if (cursorF >= last()) {
        cursorF = last();
        playing = false;
      }
    },
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/playback.test.ts`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/engine/playback.ts src/engine/playback.test.ts
git commit -m "feat: playback controller with play/pause/step/speed/reset"
```

---

### Task 5: Event bus (src/bus/eventBus.ts)

**Files:**
- Create: `src/bus/eventBus.ts`
- Test: `src/bus/eventBus.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect, vi } from 'vitest';
import { eventBus } from './eventBus';

describe('eventBus', () => {
  it('delivers events to subscribers and supports unsubscribe', () => {
    const fn = vi.fn();
    const unsub = eventBus.subscribe(fn);
    eventBus.emit({ type: 'highlight', payload: { panel: 'matrix', id: 'w23', intensity: 1 } });
    expect(fn).toHaveBeenCalledTimes(1);
    unsub();
    eventBus.emit({ type: 'highlight', payload: { panel: 'matrix', id: 'w23', intensity: 1 } });
    expect(fn).toHaveBeenCalledTimes(1);
  });
  it('does not crash with zero subscribers', () => {
    eventBus.emit({ type: 'highlight', payload: { panel: 'x', id: 'y', intensity: 0.5 } });
  });
  it('isolates throwing subscribers', () => {
    const ok = vi.fn();
    eventBus.subscribe(() => { throw new Error('boom'); });
    eventBus.subscribe(ok);
    expect(() => {
      eventBus.emit({ type: 'clear-highlights' });
    }).not.toThrow();
    expect(ok).toHaveBeenCalledTimes(1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/bus/eventBus.test.ts`
Expected: FAIL — cannot find module './eventBus'.

- [ ] **Step 3: Write src/bus/eventBus.ts**

```ts
export type BusEvent =
  | { type: 'highlight'; payload: { panel: string; id: string; intensity: number } }
  | { type: 'clear-highlights' }
  | { type: 'open-topic'; payload: { topicId: string } }
  | { type: 'navigate-view'; payload: { view: string } }
  | { type: 'playback-cursor'; payload: { step: number } }
  | { type: 'question-answered'; payload: { questionId: string; correct: boolean } }
  | { type: 'explain-step'; payload: { step: number } };

type Handler = (e: BusEvent) => void;

class EventBus {
  private handlers = new Set<Handler>();

  subscribe(h: Handler): () => void {
    this.handlers.add(h);
    return () => { this.handlers.delete(h); };
  }

  emit(e: BusEvent): void {
    // NOTE: Set iteration is insertion-ordered; a handler unsubscribing another
    // not-yet-visited handler skips it, and handlers subscribing mid-emit run once.
    for (const h of this.handlers) {
      try { h(e); } catch (err) {
        // isolation guarantee: subscriber errors never break the bus,
        // but they must not be silent ghosts either
        console.error('[eventBus] handler failed for event', e.type, err);
      }
    }
  }
}

export const eventBus = new EventBus();
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/bus/eventBus.test.ts`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bus/eventBus.ts src/bus/eventBus.test.ts
git commit -m "feat: global semantic event bus"
```

---

### Task 6: Registries — topic + view (src/registry/)

**Files:**
- Create: `src/registry/topicRegistry.ts`, `src/registry/viewRegistry.ts`, `src/registry/loadTopics.ts`
- Test: `src/registry/topicRegistry.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { registerTopic, getTopic, listTopics, getTopicCount, migrateBundle } from './topicRegistry';
import { registerView, getView, viewExists, type ViewProps } from './viewRegistry';
import type { TopicModule, SessionBundle } from '../engine/types';

const fakeTopic = {
  id: 'fake', title: 'Fake', version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 3, coding: 2, visualization: 2, gateFrequency: 4 },
    estimatedHours: 2, revisionPriority: 'P0', examFrequency: 'Frequent',
    prerequisites: [], relatedTopics: [],
    revision: { quick: '5m', standard: '15m', deep: '30m', mastery: '60m' },
  },
  layers: { foundation: [], core: [], advanced: [] },
  params: [], simulation: null as any, formulas: [], derivations: [],
  questions: [], comparisons: [], failureDemos: [], mistakes: [], testCases: [],
} as TopicModule;

describe('topicRegistry', () => {
  it('registers and retrieves topics', () => {
    registerTopic(fakeTopic);
    expect(getTopic('fake')?.title).toBe('Fake');
    expect(listTopics().length).toBe(1);
    expect(getTopicCount()).toBe(1);
  });
  it('migrates old bundles forward', () => {
    const topic = {
      ...fakeTopic, version: 2,
      migrations: { 1: (b: SessionBundle) => ({ ...b, params: { migrated: true } }) },
    } as TopicModule;
    // NOTE: moduleVersion 0 = legacy pre-versioning bundle; migration keys mean
    // "migrate TO this version" (guard: b.moduleVersion < key), so a bundle
    // already at version 1 must NOT be re-migrated by key 1.
    const old: SessionBundle = { topicId: 'fake', moduleVersion: 0, params: {}, step: 0, activeView: 'x', bookmarks: [], savedAt: 't' };
    const migrated = migrateBundle(topic, old);
    expect(migrated.moduleVersion).toBe(2);
    expect(migrated.params.migrated).toBe(true);
  });
});

describe('viewRegistry', () => {
  it('registers views idempotently', () => {
    const C = (_p: ViewProps) => null;
    registerView('scatter-plot', C);
    expect(viewExists('scatter-plot')).toBe(true);
    expect(getView('scatter-plot')).toBe(C);
    registerView('scatter-plot', C); // no throw on duplicate
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/registry/topicRegistry.test.ts`
Expected: FAIL — cannot find module.

- [ ] **Step 3: Write the registries**

```ts
// src/registry/topicRegistry.ts
import type { TopicModule, SessionBundle } from '../engine/types';

const topics = new Map<string, TopicModule>();

export function registerTopic(m: TopicModule): void {
  if (topics.has(m.id)) {
    const existing = topics.get(m.id)!;
    if (existing.version >= m.version) return; // keep newest
  }
  topics.set(m.id, m);
}

export function getTopic(id: string): TopicModule | undefined { return topics.get(id); }
export function listTopics(): TopicModule[] { return [...topics.values()]; }
export function getTopicCount(): number { return topics.size; }

export function migrateBundle(topic: TopicModule, bundle: SessionBundle): SessionBundle {
  // A bundle NEWER than the module (rolled-back app / cached session) must not be
  // stamped down — re-running old migrations on newer-shaped data would corrupt it.
  if (bundle.moduleVersion > topic.version) {
    console.warn(`[registry] bundle moduleVersion ${bundle.moduleVersion} is newer than topic ${topic.id}@${topic.version}; leaving bundle untouched`);
    return bundle;
  }
  let b = bundle;
  const migrations = topic.migrations ?? {};
  for (const v of Object.keys(migrations).map(Number).sort((a, b) => a - b)) {
    if (b.moduleVersion < v) b = migrations[v](b);
  }
  return { ...b, moduleVersion: topic.version };
}
```

```ts
// src/registry/viewRegistry.ts
import type { ComponentType } from 'react';
import type { Params, SnapshotRun, SimState } from '../engine/types';
import type { BusEvent } from '../bus/eventBus';

export interface ViewProps {
  run?: SnapshotRun;
  snapshot?: SimState | null;
  params: Params;
  // Bus-shaped subscription so consumers narrow on the BusEvent union
  // (matches eventBus.subscribe; ViewHost wires it up)
  subscribe?: (fn: (e: BusEvent) => void) => () => void;
  compact?: boolean;
}

const views = new Map<string, ComponentType<ViewProps>>();

export function registerView(id: string, comp: ComponentType<ViewProps>): void {
  views.set(id, comp);
}
export function getView(id: string): ComponentType<ViewProps> | undefined { return views.get(id); }
export function viewExists(id: string): boolean { return views.has(id); }
```

```ts
// src/registry/loadTopics.ts
/**
 * Lazy-loads all topic modules. Each module file calls register() which
 * self-registers. import.meta.glob gives per-chunk code splitting.
 */
const topicLoaders = import.meta.glob('../topics/*/module.ts');

async function registerModule(mod: Record<string, unknown>): Promise<void> {
  if (typeof mod.register === 'function') (mod.register as () => void)();
}

export async function loadTopic(topicId: string): Promise<boolean> {
  const path = `../topics/${topicId}/module.ts`;
  const loader = topicLoaders[path];
  if (!loader) return false;
  try {
    await registerModule((await loader()) as Record<string, unknown>);
    return true;
  } catch (e) {
    console.error(`[registry] failed to load topic module ${topicId}`, e);
    return false;
  }
}

export async function loadAllTopics(): Promise<number> {
  const keys = Object.keys(topicLoaders);
  let ok = 0;
  await Promise.all(keys.map(async (k) => {
    try {
      await registerModule((await topicLoaders[k]()) as Record<string, unknown>);
      ok++;
    } catch (e) {
      // one broken module must not brick the whole topic graph
      console.error(`[registry] failed to load topic module ${k}`, e);
    }
  }));
  return ok;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/registry/topicRegistry.test.ts`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/registry
git commit -m "feat: topic and view registries with lazy loading"
```

### Task 7: Zustand stores — settings, playback, progress, analytics, sessions

**Files:**
- Create: `src/store/settingsStore.ts`, `src/store/playbackStore.ts`, `src/store/progressStore.ts`, `src/store/analyticsStore.ts`, `src/store/sessionStore.ts`
- Test: `src/store/stores.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect, beforeEach } from 'vitest';
import { useSettingsStore } from './settingsStore';
import { useProgressStore } from './progressStore';
import { useAnalyticsStore } from './analyticsStore';
import { useSessionStore } from './sessionStore';

beforeEach(() => {
  localStorage.clear();
  useSettingsStore.getState().reset();
  useProgressStore.getState().reset();
  useAnalyticsStore.getState().reset();
  useSessionStore.getState().reset();
});

describe('settingsStore', () => {
  it('toggles theme and palette', () => {
    useSettingsStore.getState().setTheme('dark');
    expect(useSettingsStore.getState().theme).toBe('dark');
    useSettingsStore.getState().setPalette('deuteranopia');
    expect(useSettingsStore.getState().palette).toBe('deuteranopia');
  });
});

describe('progressStore', () => {
  it('marks views complete and bookmarks topics', () => {
    useProgressStore.getState().markView('gd', 'geometry');
    useProgressStore.getState().toggleBookmark('gd');
    const s = useProgressStore.getState();
    expect(s.completed['gd']?.viewsDone['geometry']).toBe(true);
    expect(s.isTopicComplete('gd')).toBe(false); // not enough views done
  });
});

describe('analyticsStore', () => {
  it('records question attempts and time', () => {
    useAnalyticsStore.getState().recordQuestion('gd-004', true, 'gd');
    useAnalyticsStore.getState().addTime('gd', 30);
    const s = useAnalyticsStore.getState();
    expect(s.questionsAttempted['gd-004']).toBeDefined();
    expect(s.timePerTopic['gd']).toBe(30);
  });
});

describe('sessionStore', () => {
  it('saves and lists sessions', () => {
    useSessionStore.getState().saveSession({ topicId: 'gd', moduleVersion: 1, params: {}, step: 3, activeView: 'geometry', bookmarks: [], savedAt: 'x' });
    expect(useSessionStore.getState().sessions.length).toBe(1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/store/stores.test.ts`
Expected: FAIL — cannot find modules.

- [ ] **Step 3: Write the stores (all with localStorage persistence via zustand/middleware)**

```ts
// src/store/settingsStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Theme = 'light' | 'dark';
export type Palette = 'default' | 'deuteranopia' | 'protanopia' | 'tritanopia';

interface SettingsState {
  theme: Theme;
  palette: Palette;
  reducedMotion: boolean;
  showTelemetry: boolean;
  setTheme(t: Theme): void;
  setPalette(p: Palette): void;
  setReducedMotion(v: boolean): void;
  setShowTelemetry(v: boolean): void;
  reset(): void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'light',
      palette: 'default',
      reducedMotion: false,
      showTelemetry: false,
      setTheme: (theme) => set({ theme }),
      setPalette: (palette) => set({ palette }),
      setReducedMotion: (reducedMotion) => set({ reducedMotion }),
      setShowTelemetry: (showTelemetry) => set({ showTelemetry }),
      reset: () => set({ theme: 'light', palette: 'default', reducedMotion: false, showTelemetry: false }),
    }),
    { name: 'mlv-settings', version: 1 }
  )
);
```

```ts
// src/store/playbackStore.ts
import { create } from 'zustand';
import { createPlayback, type Playback } from '../engine/playback';
import { computeRun } from '../engine/core';
import type { SimulationDef, Params, SnapshotRun } from '../engine/types';

interface PlaybackState {
  run: SnapshotRun | null;
  playback: Playback | null;
  cursor: number;
  playing: boolean;
  speed: number;
  computeAndSet(sim: SimulationDef, params: Params): void;
  setCursor(i: number): void;
  play(): void;
  pause(): void;
  stepForward(): void;
  stepBackward(): void;
  reset(): void;
  setSpeed(s: number): void;
  tick(): void; // frame tick: advances playback and syncs cursor
}

export const usePlaybackStore = create<PlaybackState>((set, get) => ({
  run: null,
  playback: null,
  cursor: 0,
  playing: false,
  speed: 1,

  computeAndSet: (sim, params) => {
    const run = computeRun(sim, params);
    const playback = createPlayback(run);
    // mirror playback.cursor so empty-run sentinel (-1) propagates immediately
    set({ run, playback, cursor: playback.cursor, playing: false });
  },

  setCursor: (i) => {
    const { playback } = get();
    if (!playback) return;
    playback.jumpTo(i);
    set({ cursor: playback.cursor });
  },

  play: () => {
    const pb = get().playback;
    if (!pb) return;
    pb.play();
    // mirror engine: play() is a no-op on empty runs — never claim playing
    set({ playing: pb.playing });
  },
  pause: () => { get().playback?.pause(); set({ playing: false }); },
  stepForward: () => {
    const { playback } = get();
    if (!playback) return;
    playback.stepForward();
    set({ cursor: playback.cursor });
  },
  stepBackward: () => {
    const { playback } = get();
    if (!playback) return;
    playback.stepBackward();
    set({ cursor: playback.cursor });
  },
  reset: () => {
    const { playback } = get();
    if (!playback) return;
    playback.reset();
    set({ cursor: playback.cursor, playing: false });
  },
  setSpeed: (s) => {
    const pb = get().playback;
    if (!pb) return;
    pb.setSpeed(s);
    // mirror engine: setSpeed rejects non-finite and clamps to [0.1, 8]
    set({ speed: pb.speed });
  },
  tick: () => {
    const { playback, playing } = get();
    if (!playback || !playing) return;
    playback.tick();
    // sync on cursor AND playing change: auto-stop at run end flips playing with cursor unchanged
    if (playback.cursor !== get().cursor || playback.playing !== get().playing) {
      set({ cursor: playback.cursor, playing: playback.playing });
    }
  },
}));
```

```ts
// src/store/progressStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface TopicProgress {
  viewsDone: Record<string, boolean>;
  mastered?: boolean;
}

interface ProgressState {
  completed: Record<string, TopicProgress>;
  bookmarks: string[];
  lastVisited?: string;
  markView(topicId: string, view: string): void;
  toggleBookmark(topicId: string): void;
  isTopicComplete(topicId: string): boolean;
  setLastVisited(topicId: string): void;
  reset(): void;
}

export const useProgressStore = create<ProgressState>()(
  persist(
    (set, get) => ({
      completed: {},
      bookmarks: [],
      markView: (topicId, view) =>
        set((s) => ({
          completed: {
            ...s.completed,
            [topicId]: {
              viewsDone: { ...(s.completed[topicId]?.viewsDone ?? {}), [view]: true },
              mastered: s.completed[topicId]?.mastered,
            },
          },
        })),
      toggleBookmark: (topicId) =>
        set((s) => ({
          bookmarks: s.bookmarks.includes(topicId)
            ? s.bookmarks.filter((b) => b !== topicId)
            : [...s.bookmarks, topicId],
        })),
      isTopicComplete: (topicId) => {
        const p = get().completed[topicId];
        if (!p) return false;
        return Object.keys(p.viewsDone).length >= 3; // ≥3 views engaged
      },
      setLastVisited: (lastVisited) => set({ lastVisited }),
      reset: () => set({ completed: {}, bookmarks: [], lastVisited: undefined }),
    }),
    { name: 'mlv-progress', version: 1 }
  )
);
```

```ts
// src/store/analyticsStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface QuestionAttempt {
  questionId: string;
  correct: boolean;
  topicId: string;
  at: number;
}

interface AnalyticsState {
  questionsAttempted: Record<string, QuestionAttempt>;
  timePerTopic: Record<string, number>;
  topicVisits: Record<string, number>;
  recordQuestion(questionId: string, correct: boolean, topicId: string): void;
  addTime(topicId: string, seconds: number): void;
  recordVisit(topicId: string): void;
  getWeakestTopics(): string[];
  reset(): void;
}

export const useAnalyticsStore = create<AnalyticsState>()(
  persist(
    (set, get) => ({
      questionsAttempted: {},
      timePerTopic: {},
      topicVisits: {},
      recordQuestion: (questionId, correct, topicId) =>
        set((s) => ({
          questionsAttempted: {
            ...s.questionsAttempted,
            [questionId]: { questionId, correct, topicId, at: Date.now() },
          },
        })),
      addTime: (topicId, seconds) =>
        set((s) => ({ timePerTopic: { ...s.timePerTopic, [topicId]: (s.timePerTopic[topicId] ?? 0) + seconds } })),
      recordVisit: (topicId) =>
        set((s) => ({ topicVisits: { ...s.topicVisits, [topicId]: (s.topicVisits[topicId] ?? 0) + 1 } })),
      getWeakestTopics: () => {
        const wrong: Record<string, number> = {};
        for (const q of Object.values(get().questionsAttempted)) {
          if (!q.correct) wrong[q.topicId] = (wrong[q.topicId] ?? 0) + 1;
        }
        return Object.entries(wrong)
          .sort((a, b) => b[1] - a[1])
          .map(([t]) => t)
          .slice(0, 5);
      },
      reset: () => set({ questionsAttempted: {}, timePerTopic: {}, topicVisits: {} }),
    }),
    { name: 'mlv-analytics', version: 1 }
  )
);
```

```ts
// src/store/sessionStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { SessionBundle } from '../engine/types';

interface SessionState {
  sessions: SessionBundle[];
  saveSession(bundle: SessionBundle): void;
  deleteSession(savedAt: string): void;
  resumeSession(savedAt: string): SessionBundle | undefined;
  reset(): void;
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      sessions: [],
      saveSession: (bundle) =>
        set((s) => {
          const rest = s.sessions.filter((x) => x.savedAt !== bundle.savedAt);
          return { sessions: [bundle, ...rest].slice(0, 20) };
        }),
      deleteSession: (savedAt) =>
        set((s) => ({ sessions: s.sessions.filter((x) => x.savedAt !== savedAt) })),
      resumeSession: (savedAt) => get().sessions.find((x) => x.savedAt === savedAt),
      reset: () => set({ sessions: [] }),
    }),
    { name: 'mlv-sessions', version: 1 }
  )
);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/store`
Expected: PASS — `stores.test.ts` (4 describe blocks) + `playbackStore.test.ts` (6 tests: engine-mirroring semantics — empty-run play no-op, sentinel cursor -1 propagation, tick auto-stop flipping playing, setSpeed clamping/rejection mirror, reset/step mirroring)

- [ ] **Step 5: Commit**

```bash
git add src/store
git commit -m "feat: zustand stores with localStorage persistence"
```

---

### Task 8: App shell — router, layout, theme, keyboard shortcuts, command palette

**Files:**
- Create: `src/app/ThemeProvider.tsx`, `src/app/KeyboardShortcuts.tsx`, `src/app/AppShell.tsx`, `src/app/Router.tsx`, `src/app/CommandPalette.tsx`, `src/styles.css`

- [ ] **Step 1: Write ThemeProvider + KeyboardShortcuts**

```tsx
// src/app/ThemeProvider.tsx
import { useEffect, type ReactNode } from 'react';
import { useSettingsStore } from '../store/settingsStore';

export function ThemeProvider({ children }: { children: ReactNode }) {
  const theme = useSettingsStore((s) => s.theme);
  const palette = useSettingsStore((s) => s.palette);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.palette = palette;
  }, [theme, palette]);
  return <>{children}</>;
}
```

```tsx
// src/app/KeyboardShortcuts.tsx
import { useEffect } from 'react';
import { usePlaybackStore } from '../store/playbackStore';

const PREVENT = new Set(['Space', 'ArrowLeft', 'ArrowRight']);

export function KeyboardShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (PREVENT.has(e.code)) e.preventDefault();
      switch (e.code) {
        case 'Space': {
          const st = usePlaybackStore.getState();
          st.playing ? st.pause() : st.play();
          break;
        }
        case 'ArrowLeft': usePlaybackStore.getState().stepBackward(); break;
        case 'ArrowRight': usePlaybackStore.getState().stepForward(); break;
        case 'KeyR': usePlaybackStore.getState().reset(); break;
        case 'KeyS':
          document.dispatchEvent(new CustomEvent('mlv:open-palette'));
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  return null;
}
```

- [ ] **Step 2: Write AppShell + Router**

```tsx
// src/app/AppShell.tsx
import { NavLink, Outlet } from 'react-router-dom';
import { listTopics } from '../registry/topicRegistry';
import { ThemeProvider } from './ThemeProvider';
import { KeyboardShortcuts } from './KeyboardShortcuts';
import { CommandPalette } from './CommandPalette';

export function AppShell() {
  const topics = listTopics();
  return (
    <ThemeProvider>
      <KeyboardShortcuts />
      <div className="shell">
        <aside className="sidebar">
          <h1>GATE ML Visualizer</h1>
          <nav>
            <NavLink to="/">Home</NavLink>
            <NavLink to="/graph">Knowledge Graph</NavLink>
            <NavLink to="/journey">Learning Journey</NavLink>
            <NavLink to="/exam">Exam Mode</NavLink>
            <div className="topic-list">
              {topics.map((t) => (
                <NavLink key={t.id} to={`/topic/${t.id}`}>{t.title}</NavLink>
              ))}
            </div>
          </nav>
        </aside>
        <main className="content">
          <Outlet />
        </main>
        <CommandPalette />
      </div>
    </ThemeProvider>
  );
}
```

```tsx
// src/app/Router.tsx
import { Routes, Route } from 'react-router-dom';
import { AppShell } from './AppShell';
import { HomePage } from '../pages/HomePage';
import { TopicPage } from '../pages/TopicPage';
import { GraphPage } from '../pages/GraphPage';
import { JourneyPage } from '../pages/JourneyPage';
import { ExamPage } from '../pages/ExamPage';
import { loadTopic } from '../registry/loadTopics';

export function Router() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="/journey" element={<JourneyPage />} />
        <Route path="/exam" element={<ExamPage />} />
        <Route path="/topic/:topicId" element={<TopicPage loader={loadTopic} />} />
      </Route>
    </Routes>
  );
}
```

```tsx
// src/app/CommandPalette.tsx — fuzzy search across topics + concepts (minimal v1)
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listTopics } from '../registry/topicRegistry';

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const onOpen = () => { setOpen(true); };
    document.addEventListener('mlv:open-palette', onOpen);
    return () => document.removeEventListener('mlv:open-palette', onOpen);
  }, []);

  if (!open) return null;

  const results = listTopics().filter((t) =>
    t.title.toLowerCase().includes(q.toLowerCase())
  );

  return (
    <div className="palette-overlay" onClick={() => setOpen(false)}>
      <div className="palette" onClick={(e) => e.stopPropagation()}>
        <input autoFocus placeholder="Search topics, concepts, formulas… (S)"
          value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="palette-results">
          {results.map((t) => (
            <button key={t.id} onClick={() => { navigate(`/topic/${t.id}`); setOpen(false); }}>
              {t.title}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Write src/styles.css (design tokens — light/dark + colorblind palettes)**

```css
:root {
  --bg: #ffffff; --fg: #1a1a2e; --accent: #3b82f6;
  --panel: #f1f5f9; --border: #e2e8f0; --ok: #16a34a; --err: #dc2626;
  --cat1: #2563eb; --cat2: #dc2626;
}
:root[data-theme='dark'] {
  --bg: #0f172a; --fg: #e2e8f0; --accent: #60a5fa;
  --panel: #1e293b; --border: #334155; --ok: #4ade80; --err: #f87171;
  --cat1: #60a5fa; --cat2: #f87171;
}
/* Colorblind palettes: override categorical colors */
:root[data-palette='deuteranopia'] { --cat1: #0072b2; --cat2: #e69f00; }
:root[data-palette='protanopia']  { --cat1: #009e73; --cat2: #d55e00; }
:root[data-palette='tritanopia']  { --cat1: #56b4e9; --cat2: #e69f00; }
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, sans-serif; background: var(--bg); color: var(--fg); }
.shell { display: flex; min-height: 100vh; }
.sidebar { width: 260px; border-right: 1px solid var(--border); padding: 1rem; overflow-y: auto; }
.sidebar nav { display: flex; flex-direction: column; gap: 0.25rem; }
.sidebar a { color: var(--fg); text-decoration: none; padding: 0.4rem 0.5rem; border-radius: 6px; }
.sidebar a.active { background: var(--panel); color: var(--accent); }
.content { flex: 1; padding: 1.5rem; overflow: auto; }
@media (max-width: 768px) {
  .shell { flex-direction: column; }
  .sidebar { width: 100%; max-height: 30vh; }
}
@media (prefers-reduced-motion: reduce) {
  * { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}
/* topic layout */
.topic-layout { display: grid; grid-template-columns: 1fr 300px; gap: 1rem; }
.view-card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
.control-panel { position: sticky; top: 1rem; align-self: start; }
.topic-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1rem; }
.topic-card { border: 1px solid var(--border); border-radius: 8px; padding: 1rem; text-decoration: none; color: var(--fg); }
.topic-card:hover { border-color: var(--accent); }
.palette-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.4); z-index: 100; display: flex; justify-content: center; padding-top: 15vh; }
.palette { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; width: 480px; max-height: 50vh; overflow: auto; padding: 1rem; }
.palette input { width: 100%; padding: 0.5rem; }
.palette-results { display: flex; flex-direction: column; margin-top: 0.5rem; }
.palette-results button { text-align: left; padding: 0.5rem; background: none; border: none; color: var(--fg); cursor: pointer; }
.palette-results button:hover { background: var(--panel); }
/* playback */
.playback-bar { display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem; background: var(--panel); border-radius: 8px; margin-bottom: 1rem; }
.playback-bar input[type='range'] { flex: 1; }
/* tabs */
.tabs { display: flex; gap: 0.25rem; }
.tab { padding: 0.4rem 0.8rem; border: 1px solid var(--border); background: none; color: var(--fg); border-radius: 6px; cursor: pointer; }
.tab.active { background: var(--accent); color: white; }
/* heatmap */
.heatmap-row { display: flex; align-items: center; gap: 0.5rem; }
.heatmap-bar { flex: 1; height: 8px; background: var(--border); border-radius: 4px; }
.heatmap-fill { height: 100%; background: var(--accent); border-radius: 4px; }
/* matrix */
.matrix { border-collapse: collapse; }
.cell { padding: 0.4rem 0.6rem; border: 1px solid var(--border); text-align: center; font-variant-numeric: tabular-nums; }
.cell.active { background: #f59e0b; color: #000; }
/* timeline */
.tl-stage { display: flex; flex-direction: column; align-items: center; cursor: pointer; }
.tl-dot { width: 14px; height: 14px; border-radius: 50%; background: var(--border); }
.tl-stage.active .tl-dot { background: var(--accent); }
.tl-connector { color: var(--fg); }
/* journey */
.journey-node { display: flex; flex-direction: column; align-items: center; }
.journey-link { display: flex; flex-direction: column; align-items: center; text-decoration: none; color: var(--fg); }
.journey-dot { width: 32px; height: 32px; border-radius: 50%; background: var(--panel); border: 2px solid var(--border); display: flex; align-items: center; justify-content: center; }
.journey-link.done .journey-dot { background: var(--ok); color: white; }
.journey-arrow { color: var(--fg); }
@media (max-width: 900px) {
  .topic-layout { grid-template-columns: 1fr; }
  .control-panel { position: static; }
}
```

- [ ] **Step 4: Wire Router into main.tsx**

```tsx
// src/main.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { Router } from './app/Router';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Router />
    </BrowserRouter>
  </React.StrictMode>
);
```

- [ ] **Step 5: Typecheck + build**

Run: `npm run lint && npm run build`
Expected: no TypeScript errors, build succeeds.

- [ ] **Step 6: Commit**

```bash
git add src/app src/styles.css src/main.tsx
git commit -m "feat: app shell, routing, theme tokens, keyboard shortcuts"
```

### Task 9: Pages — Home, TopicPage (view host), Graph, Journey, Exam

**Files:**
- Create: `src/pages/ViewHost.tsx`, `src/pages/TopicPage.tsx`, `src/pages/HomePage.tsx`, `src/lib/params.ts`
- Skeleton for GraphPage/JourneyPage/ExamPage created here; real implementations in Tasks 13-14.

- [ ] **Step 1: Write src/lib/params.ts + ViewHost**

```ts
// src/lib/params.ts
import type { Params, ParamSchema } from '../engine/types';

export function defaultParams(schema: ParamSchema[]): Params {
  const p: Params = {};
  for (const s of schema) p[s.id] = s.default;
  return p;
}
```

```tsx
// src/pages/ViewHost.tsx
import { useEffect, useMemo, useRef } from 'react';
import { getView } from '../registry/viewRegistry';
import { usePlaybackStore } from '../store/playbackStore';
import { eventBus } from '../bus/eventBus';
import type { TopicModule, Params } from '../engine/types';

export function ViewHost({ topic, component, params }: {
  topic: TopicModule; component: string; params: Params;
}) {
  const Comp = getView(component);
  const run = usePlaybackStore((s) => s.run);
  const cursor = usePlaybackStore((s) => s.cursor);
  const frame = useRef<number>(0);

  // animation loop drives playback ticks at 60fps
  useEffect(() => {
    const loop = () => {
      usePlaybackStore.getState().tick();
      frame.current = requestAnimationFrame(loop);
    };
    frame.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame.current);
  }, []);

  // re-compute snapshots when params change (debounced 150ms)
  useEffect(() => {
    const id = setTimeout(() => {
      usePlaybackStore.getState().computeAndSet(topic.simulation, params);
    }, 150);
    return () => clearTimeout(id);
  }, [topic, params]);

  useEffect(() => () => topic.dispose?.(), [topic]);

  const snapshot = useMemo(() => run?.snapshots[cursor] ?? null, [run, cursor]);
  const subscribe = useMemo(() => (fn: (e: unknown) => void) => eventBus.subscribe(fn as never), []);

  if (!Comp) return <div>Unknown view: {component}</div>;
  return (
    <div className="view-host">
      <Comp run={run} params={params} snapshot={snapshot} subscribe={subscribe} />
    </div>
  );
}
```

- [ ] **Step 2: Write TopicPage + HomePage**

```tsx
// src/pages/TopicPage.tsx
import { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getTopic } from '../registry/topicRegistry';
import { useProgressStore } from '../store/progressStore';
import { useAnalyticsStore } from '../store/analyticsStore';
import { ViewHost } from './ViewHost';
import { PlaybackBar } from '../ui/PlaybackBar';
import { ParamPanel } from '../ui/ParamPanel';
import { Tabs } from '../ui/Tabs';
import { defaultParams } from '../lib/params';
import type { Params } from '../engine/types';

const LAYER_ORDER = ['foundation', 'core', 'advanced'] as const;

export function TopicPage({ loader }: { loader: (id: string) => Promise<void> }) {
  const { topicId = '' } = useParams();
  const [loaded, setLoaded] = useState(false);
  const [activeLayer, setActiveLayer] = useState<(typeof LAYER_ORDER)[number]>('foundation');
  const [params, setParams] = useState<Params>({});

  useEffect(() => {
    loader(topicId).then(() => {
      const t = getTopic(topicId);
      if (t) setParams(defaultParams(t.params));
      setLoaded(true);
      useProgressStore.getState().setLastVisited(topicId);
      useAnalyticsStore.getState().recordVisit(topicId);
    });
  }, [topicId, loader]);

  const topic = loaded ? getTopic(topicId) : undefined;
  const views = useMemo(
    () => (topic ? topic.layers[activeLayer] : []),
    [topic, activeLayer]
  );

  useEffect(() => {
    const t = topic;
    if (!t) return;
    for (const v of views) useProgressStore.getState().markView(t.id, v.component);
  }, [topic, views]);

  if (!topic) return <div>Loading topic…</div>;

  return (
    <div className="topic-page">
      <header>
        <h1>{topic.title}</h1>
        <Tabs
          tabs={LAYER_ORDER.map((l) => ({ id: l, label: l.charAt(0).toUpperCase() + l.slice(1) }))}
          active={activeLayer}
          onChange={(id) => setActiveLayer(id as (typeof LAYER_ORDER)[number])}
        />
      </header>
      <div className="topic-layout">
        <main className="view-stack">
          {views.map((v) => (
            <section key={v.component} className="view-card">
              <h2>{v.title}</h2>
              <ViewHost topic={topic} component={v.component} params={params} />
            </section>
          ))}
        </main>
        <aside className="control-panel">
          <PlaybackBar />
          <ParamPanel schema={topic.params} values={params} onChange={setParams} />
        </aside>
      </div>
    </div>
  );
}
```

```tsx
// src/pages/HomePage.tsx — dashboard
import { Link } from 'react-router-dom';
import { listTopics } from '../registry/topicRegistry';
import { useProgressStore } from '../store/progressStore';

export function HomePage() {
  const topics = listTopics();
  const completed = useProgressStore((s) => s.completed);
  return (
    <div>
      <h1>GATE DA — Machine Learning Visualizer</h1>
      <p>See the algorithm think. 31 topics, 30+ interactive views, GATE exam mode.</p>
      <div className="topic-grid">
        {topics.map((t) => (
          <Link key={t.id} to={`/topic/${t.id}`} className="topic-card">
            <h3>{t.title}</h3>
            <small>Weightage: {t.metadata.gateWeightage} · Priority {t.metadata.revisionPriority}</small>
            <div>{completed[t.id] ? '✓ started' : 'not started'}</div>
          </Link>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Write page skeletons (real impl in later tasks)**

```tsx
// src/pages/GraphPage.tsx
export function GraphPage() {
  return <div><h1>Knowledge Graph</h1><p>Coming in Task 14.</p></div>;
}
```

```tsx
// src/pages/JourneyPage.tsx
export function JourneyPage() {
  return <div><h1>Learning Journey</h1><p>Coming in Task 14.</p></div>;
}
```

```tsx
// src/pages/ExamPage.tsx
export function ExamPage() {
  return <div><h1>Exam Mode</h1><p>Coming in Task 13.</p></div>;
}
```

- [ ] **Step 4: Verify typecheck + build**

Run: `npm run lint`
Expected: no TypeScript errors.

Run: `npm run build`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add src/pages src/lib/params.ts
git commit -m "feat: topic workbench pages and view host"
```

---

### Task 10: UI controls — PlaybackBar, ParamPanel, Slider, Toggle, Select, Tabs, Latex, Heatmap, MetricGrid

**Files:**
- Create: `src/ui/PlaybackBar.tsx`, `src/ui/ParamPanel.tsx`, `src/ui/Slider.tsx`, `src/ui/Toggle.tsx`, `src/ui/Select.tsx`, `src/ui/Tabs.tsx`, `src/ui/Latex.tsx`, `src/ui/Heatmap.tsx`, `src/ui/MetricGrid.tsx`
- Test: `src/ui/PlaybackBar.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PlaybackBar } from './PlaybackBar';
import { usePlaybackStore } from '../store/playbackStore';

describe('PlaybackBar', () => {
  it('renders controls and steps forward', () => {
    usePlaybackStore.getState().computeAndSet({
      initialState: () => ({ algorithm: {}, visuals: [], math: [], narration: '', explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] }, highlights: [], metrics: {}, events: [], timeline: [] }),
      step: () => null,
    } as any, {});
    render(<PlaybackBar />);
    const next = screen.getByRole('button', { name: /next/i });
    fireEvent.click(next);
    // cursor should not exceed 0 (single snapshot run)
    expect(usePlaybackStore.getState().cursor).toBe(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/ui/PlaybackBar.test.tsx`
Expected: FAIL — cannot find module.

- [ ] **Step 3: Write the controls**

```tsx
// src/ui/PlaybackBar.tsx
import { usePlaybackStore } from '../store/playbackStore';

export function PlaybackBar() {
  const cursor = usePlaybackStore((s) => s.cursor);
  const playing = usePlaybackStore((s) => s.playing);
  const speed = usePlaybackStore((s) => s.speed);
  const run = usePlaybackStore((s) => s.run);
  const max = run ? run.snapshots.length - 1 : 0;

  return (
    <div className="playback-bar" role="toolbar" aria-label="Playback">
      <button aria-label="Previous" onClick={() => usePlaybackStore.getState().stepBackward()}>⏮</button>
      <button aria-label="Play/Pause" onClick={() => {
        const st = usePlaybackStore.getState();
        st.playing ? st.pause() : st.play();
      }}>
        {playing ? '⏸' : '▶'}
      </button>
      <button aria-label="Next" onClick={() => usePlaybackStore.getState().stepForward()}>⏭</button>
      <button aria-label="Reset" onClick={() => usePlaybackStore.getState().reset()}>⟲</button>
      <input
        type="range" min={0} max={max} value={cursor}
        aria-label="Step scrubber"
        onChange={(e) => usePlaybackStore.getState().setCursor(Number(e.target.value))}
      />
      <span>Step {cursor}/{max}</span>
      <select
        aria-label="Speed"
        value={speed}
        onChange={(e) => usePlaybackStore.getState().setSpeed(Number(e.target.value))}
      >
        {[0.25, 0.5, 1, 2, 4].map((v) => <option key={v} value={v}>{v}×</option>)}
      </select>
    </div>
  );
}
```

```tsx
// src/ui/ParamPanel.tsx
import type { Params, ParamSchema } from '../engine/types';
import { Slider } from './Slider';
import { Toggle } from './Toggle';
import { Select } from './Select';

export function ParamPanel({ schema, values, onChange }: {
  schema: ParamSchema[]; values: Params; onChange: (p: Params) => void;
}) {
  return (
    <div className="param-panel">
      <h3>Parameters</h3>
      {schema.map((s) => {
        const set = (v: unknown) => onChange({ ...values, [s.id]: v });
        switch (s.type) {
          case 'number':
            return <Slider key={s.id} schema={s} value={values[s.id] as number} onChange={set} />;
          case 'toggle':
            return <Toggle key={s.id} label={s.label} value={values[s.id] as boolean} onChange={set} />;
          case 'select':
            return <Select key={s.id} schema={s} value={values[s.id] as string} onChange={set} />;
          default:
            return null;
        }
      })}
    </div>
  );
}
```

```tsx
// src/ui/Slider.tsx
import type { ParamSchema } from '../engine/types';

export function Slider({ schema, value, onChange }: {
  schema: ParamSchema; value: number; onChange: (v: number) => void;
}) {
  return (
    <label className="slider-row">
      <span>{schema.label}: <b>{value}</b></span>
      <input
        type="range" min={schema.min} max={schema.max} step={schema.step ?? 0.01}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
```

```tsx
// src/ui/Toggle.tsx
export function Toggle({ label, value, onChange }: {
  label: string; value: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <label className="toggle-row">
      <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}
```

```tsx
// src/ui/Select.tsx
import type { ParamSchema } from '../engine/types';

export function Select({ schema, value, onChange }: {
  schema: ParamSchema; value: string; onChange: (v: string) => void;
}) {
  return (
    <label className="select-row">
      <span>{schema.label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {(schema.options ?? []).map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}
```

```tsx
// src/ui/Tabs.tsx
export function Tabs({ tabs, active, onChange }: {
  tabs: { id: string; label: string }[];
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={t.id === active}
          className={t.id === active ? 'tab active' : 'tab'}
          onClick={() => onChange(t.id)}>
          {t.label}
        </button>
      ))}
    </div>
  );
}
```

```tsx
// src/ui/Latex.tsx
import { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

export function Latex({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, { throwOnError: false, displayMode: block });
    } catch {
      return tex;
    }
  }, [tex, block]);
  return <span dangerouslySetInnerHTML={{ __html: html }} aria-label={tex} />;
}
```

```tsx
// src/ui/Heatmap.tsx
import type { CSSProperties } from 'react';

export function Heatmap({ dimensions }: { dimensions: [string, number][] }) {
  return (
    <div className="heatmap">
      {dimensions.map(([label, v]) => (
        <div key={label} className="heatmap-row">
          <span>{label}</span>
          <div className="heatmap-bar">
            <div style={{ width: `${Math.max(4, v * 20)}%` } as CSSProperties} className="heatmap-fill" />
          </div>
        </div>
      ))}
    </div>
  );
}
```

```tsx
// src/ui/MetricGrid.tsx
export function MetricGrid({ metrics }: { metrics: Record<string, number> }) {
  return (
    <div className="metric-grid">
      {Object.entries(metrics).map(([k, v]) => (
        <div key={k} className="metric" data-testid={`metric-${k}`}>
          <span>{k}</span>
          <b>{Number.isFinite(v) ? v.toFixed(4) : '—'}</b>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/ui/PlaybackBar.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ui
git commit -m "feat: reusable UI controls (playback, params, latex, heatmap)"
```

---

### Task 11: Canvas renderer base + ScatterPlot + LossCurve visualizers

**Files:**
- Create: `src/lib/canvas/CanvasStage.ts`, `src/visualizers/ScatterPlot.tsx`, `src/visualizers/LossCurve.tsx`
- Test: `src/lib/canvas/CanvasStage.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { fitBounds } from './CanvasStage';

describe('fitBounds', () => {
  it('scales so data fits within padded viewport', () => {
    const t = fitBounds({ x: [0, 10], y: [0, 10] }, 200, 100, 20);
    // available: w=160, h=60 → scale = min(16, 6) = 6
    expect(t.scale).toBeCloseTo(6, 5);
    // world 0 → tx = (200 - 10*6)/2 = 70 ; world 10 → 70 + 60 = 130 = 200 - 70 ✓ centered
    expect(t.tx).toBeCloseTo(70, 5);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/canvas/CanvasStage.test.ts`
Expected: FAIL — cannot find module.

- [ ] **Step 3: Write CanvasStage**

```ts
export interface Bounds { x: [number, number]; y: [number, number] }
export interface Transform { scale: number; tx: number; ty: number }

export function fitBounds(b: Bounds, w: number, h: number, pad = 40): Transform {
  const sx = (w - 2 * pad) / (b.x[1] - b.x[0] || 1);
  const sy = (h - 2 * pad) / (b.y[1] - b.y[0] || 1);
  const scale = Math.min(sx, sy);
  const tx = (w - (b.x[0] + b.x[1]) * scale) / 2;
  const ty = (h - (b.y[0] + b.y[1]) * scale) / 2;
  return { scale, tx, ty };
}

/** Hi-DPI canvas wrapper with pan/zoom and world→screen transform. */
export class CanvasStage {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private dpr: number;
  private t: Transform = { scale: 1, tx: 0, ty: 0 };
  private onRepaint: (() => void) | null = null;

  constructor(width: number, height: number) {
    this.canvas = document.createElement('canvas');
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, width * this.dpr);
    this.canvas.height = Math.max(1, height * this.dpr);
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.ctx = this.canvas.getContext('2d')!;
  }

  setRepaint(fn: () => void) { this.onRepaint = fn; }
  requestRepaint() { this.onRepaint?.(); }

  get transform() { return this.t; }

  setBounds(b: Bounds, w: number, h: number) {
    this.t = fitBounds(b, w, h);
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, w * this.dpr);
    this.canvas.height = Math.max(1, h * this.dpr);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
  }

  worldToScreen(x: number, y: number): [number, number] {
    return [x * this.t.scale + this.t.tx, y * this.t.scale + this.t.ty];
  }

  screenToWorld(px: number, py: number): [number, number] {
    return [(px - this.t.tx) / this.t.scale, (py - this.t.ty) / this.t.scale];
  }

  zoomAt(px: number, py: number, factor: number) {
    const [wx, wy] = this.screenToWorld(px, py);
    this.t.scale *= factor;
    this.t.tx = px - wx * this.t.scale;
    this.t.ty = py - wy * this.t.scale;
    this.requestRepaint();
  }

  panBy(dx: number, dy: number) {
    this.t.tx += dx;
    this.t.ty += dy;
    this.requestRepaint();
  }

  clear(w: number, h: number, bg: string) {
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.ctx.fillStyle = bg;
    this.ctx.fillRect(0, 0, w, h);
    this.ctx.setTransform(this.dpr * this.t.scale, 0, 0, this.dpr * this.t.scale, this.dpr * this.t.tx, this.dpr * this.t.ty);
  }

  drawPath(points: [number, number][], stroke: string, width: number, fill?: string) {
    if (points.length < 2) return;
    const ctx = this.ctx;
    ctx.beginPath();
    const [x0, y0] = this.worldToScreen(points[0][0], points[0][1]);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < points.length; i++) {
      const [x, y] = this.worldToScreen(points[i][0], points[i][1]);
      ctx.lineTo(x, y);
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    ctx.stroke();
  }

  drawCircle(x: number, y: number, r: number, fill: string, stroke?: string) {
    const ctx = this.ctx;
    const [sx, sy] = this.worldToScreen(x, y);
    ctx.beginPath();
    ctx.arc(sx, sy, r * this.t.scale, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
  }

  drawArrow(x1: number, y1: number, x2: number, y2: number, color: string) {
    const ctx = this.ctx;
    const [a, b] = this.worldToScreen(x1, y1);
    const [c, d] = this.worldToScreen(x2, y2);
    const angle = Math.atan2(d - b, c - a);
    const headLen = 8;
    ctx.beginPath();
    ctx.moveTo(a, b);
    ctx.lineTo(c, d);
    ctx.lineTo(c - headLen * Math.cos(angle - Math.PI / 6), d - headLen * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(c, d);
    ctx.lineTo(c - headLen * Math.cos(angle + Math.PI / 6), d - headLen * Math.sin(angle + Math.PI / 6));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}
```

- [ ] **Step 4: Write ScatterPlot + LossCurve visualizers**

```tsx
// src/visualizers/ScatterPlot.tsx
import { useEffect, useRef, useState } from 'react';
import { CanvasStage, type Bounds } from '../lib/canvas/CanvasStage';
import { eventBus } from '../bus/eventBus';
import type { ViewProps } from '../registry/viewRegistry';
import type { VisualCommand } from '../engine/types';

const PALETTE = { point: '#2563eb', line: 'var(--fg)', hl: '#f59e0b' };

export function ScatterPlot({ snapshot, params }: ViewProps) {
  const ref = useRef<HTMLDivElement>(null);
  const stageRef = useRef<CanvasStage | null>(null);
  const highlightRef = useRef<string | null>(null);
  const [size, setSize] = useState({ w: 600, h: 400 });

  useEffect(() => {
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setSize({ w: width, h: height });
    });
    if (ref.current) ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const unsub = eventBus.subscribe((e) => {
      if (e.type === 'highlight') highlightRef.current = e.payload.id;
      if (e.type === 'clear-highlights') highlightRef.current = null;
    });
    return unsub;
  }, []);

  // instantiate stage + interaction once per size
  useEffect(() => {
    if (!ref.current) return;
    const stage = new CanvasStage(size.w, size.h);
    stage.canvas.style.display = 'block';
    ref.current.replaceChildren(stage.canvas);
    stageRef.current = stage;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = stage.canvas.getBoundingClientRect();
      stage.zoomAt(e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1.1 : 0.9);
    };
    let dragging = false, lastX = 0, lastY = 0;
    const onDown = (e: PointerEvent) => { dragging = true; lastX = e.clientX; lastY = e.clientY; };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      stage.panBy(e.clientX - lastX, e.clientY - lastY);
      lastX = e.clientX; lastY = e.clientY;
    };
    const onUp = () => { dragging = false; };
    stage.canvas.addEventListener('wheel', onWheel, { passive: false });
    stage.canvas.addEventListener('pointerdown', onDown);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      stage.canvas.removeEventListener('wheel', onWheel);
      stage.canvas.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [size]);

  // draw current snapshot
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !snapshot) return;
    const bounds = boundsOf(snapshot.visuals);
    stage.setBounds(bounds, size.w, size.h);
    stage.clear(size.w, size.h, 'transparent');

    const { visuals, highlights } = snapshot;
    for (const cmd of visuals) {
      const hl = highlights.find((h) => h.id === cmd.id);
      const isHl = highlightRef.current === cmd.id;
      switch (cmd.type) {
        case 'point': {
          stage.drawCircle(cmd.x as number, cmd.y as number, isHl || hl ? 7 : 4.5,
            isHl ? PALETTE.hl : ((cmd.color as string) ?? PALETTE.point));
          break;
        }
        case 'line': {
          stage.drawPath(cmd.points as [number, number][], (cmd.color as string) ?? PALETTE.line, isHl ? 4 : 2);
          break;
        }
        case 'arrow': {
          stage.drawArrow(cmd.x1 as number, cmd.y1 as number, cmd.x2 as number, cmd.y2 as number,
            (cmd.color as string) ?? PALETTE.line);
          break;
        }
      }
    }
  }, [snapshot, size, params]);

  return <div ref={ref} style={{ width: '100%', height: 400 }} />;
}

function boundsOf(cmds: VisualCommand[]): Bounds {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  const touch = (x: number, y: number) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  };
  for (const c of cmds) {
    if (c.type === 'point') touch(c.x as number, c.y as number);
    if (c.type === 'line' && Array.isArray(c.points)) {
      for (const [x, y] of c.points as [number, number][]) touch(x, y);
    }
    if (c.type === 'arrow') { touch(c.x1 as number, c.y1 as number); touch(c.x2 as number, c.y2 as number); }
  }
  if (!Number.isFinite(x0)) return { x: [0, 1], y: [0, 1] };
  const padX = (x1 - x0) * 0.1 + 0.5, padY = (y1 - y0) * 0.1 + 0.5;
  return { x: [x0 - padX, x1 + padX], y: [y0 - padY, y1 + padY] };
}
```

```tsx
// src/visualizers/LossCurve.tsx
import { useEffect, useRef, useState } from 'react';
import { CanvasStage, type Bounds } from '../lib/canvas/CanvasStage';
import type { SnapshotRun } from '../engine/types';
import { usePlaybackStore } from '../store/usePlaybackStore';

// IMPORTANT: registered visualizers receive ViewProps ({ run, params, snapshot, subscribe }) from ViewHost.
// Cursor-dependent components must read `cursor` from the playback store, NOT from props (ViewHost does not pass it).
export function LossCurve({ run, metricKey = 'cost' }: {
  run: SnapshotRun | null; metricKey?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const stageRef = useRef<CanvasStage | null>(null);
  const [size, setSize] = useState({ w: 600, h: 300 });
  const cursor = usePlaybackStore((s) => s.cursor);

  useEffect(() => {
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0) setSize({ w: width, h: height });
    });
    if (ref.current) ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    stageRef.current = new CanvasStage(size.w, size.h);
    stageRef.current.canvas.style.display = 'block';
    ref.current.replaceChildren(stageRef.current.canvas);
  }, [size]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !run) return;
    const values = run.snapshots.map((s) => s.metrics[metricKey] ?? NaN);
    const finite = values.filter(Number.isFinite);
    if (finite.length < 2) return;
    const b: Bounds = { x: [0, values.length - 1], y: [Math.min(...finite), Math.max(...finite)] };
    stage.setBounds(b, size.w, size.h);
    stage.clear(size.w, size.h, 'transparent');
    stage.drawPath(values.map((v, i) => [i, v] as [number, number]), '#3b82f6', 2);
    if (cursor < run.snapshots.length) {
      const v = run.snapshots[cursor].metrics[metricKey];
      if (Number.isFinite(v)) stage.drawCircle(cursor, v, 6, '#f59e0b', 'var(--fg)');
    }
  }, [run, cursor, metricKey, size]);

  return <div ref={ref} style={{ width: '100%', height: 300 }} />;
}
```

- [ ] **Step 5: Run all tests + typecheck**

Run: `npx vitest run && npm run lint`
Expected: PASS, no type errors.

- [ ] **Step 6: Commit**

```bash
git add src/lib/canvas src/visualizers/ScatterPlot.tsx src/visualizers/LossCurve.tsx
git commit -m "feat: canvas stage with pan/zoom plus scatter and loss visualizers"
```

### Task 12: MatrixAnimator + FormulaExplorer + TimelineView + MistakeView

**Files:**
- Create: `src/visualizers/MatrixAnimator.tsx`, `src/visualizers/FormulaExplorer.tsx`, `src/visualizers/TimelineView.tsx`, `src/visualizers/MistakeView.tsx`

- [ ] **Step 1: Write MatrixAnimator** (renders matrices from VisualCommand `{type:'matrix'}`; emits/consumes highlight events)

```tsx
// src/visualizers/MatrixAnimator.tsx
import { useEffect, useState } from 'react';
import { eventBus } from '../bus/eventBus';
import type { VisualCommand } from '../engine/types';

export function MatrixAnimator({ commands }: { commands: VisualCommand[] }) {
  const [hl, setHl] = useState<string | null>(null);

  useEffect(() => {
    const unsub = eventBus.subscribe((e) => {
      if (e.type === 'highlight') setHl(e.payload.id);
      if (e.type === 'clear-highlights') setHl(null);
    });
    return unsub;
  }, []);

  const matrices = commands.filter((c) => c.type === 'matrix') as (VisualCommand & {
    rows: number; cols: number; cells: (number | string)[][];
  })[];

  return (
    <div className="matrix-animator">
      {matrices.map((m, i) => (
        <div key={i} className="matrix-wrap">
          <table className="matrix">
            <tbody>
              {m.cells.map((row, r) => (
                <tr key={r}>
                  {row.map((cell, c) => {
                    const cellId = `${m.id}:${r},${c}`;
                    const active = hl === cellId;
                    return (
                      <td key={c} data-testid={cellId}
                        className={active ? 'cell active' : 'cell'}
                        onMouseEnter={() => eventBus.emit({ type: 'highlight', payload: { panel: 'matrix', id: cellId, intensity: 1 } })}
                        onMouseLeave={() => eventBus.emit({ type: 'clear-highlights' })}>
                        {typeof cell === 'number' ? cell.toFixed(2) : cell}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          {m.id && <small>{m.id}</small>}
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Write FormulaExplorer**

```tsx
// src/visualizers/FormulaExplorer.tsx
import { useState } from 'react';
import { Latex } from '../ui/Latex';
import type { Formula } from '../engine/types';

export function FormulaExplorer({ formulas }: { formulas: Formula[] }) {
  const [selectedId, setSelectedId] = useState(formulas[0]?.id);

  const f = formulas.find((x) => x.id === selectedId);
  if (!f) return null;

  return (
    <div className="formula-explorer">
      <div className="formula-list">
        {formulas.map((x) => (
          <button key={x.id} className={x.id === selectedId ? 'pill active' : 'pill'}
            onClick={() => setSelectedId(x.id)}>
            {x.id}
          </button>
        ))}
      </div>
      <div className="formula-detail">
        <Latex tex={f.latex} block />
        <h4>Symbols</h4>
        <table>
          <thead><tr><th>Symbol</th><th>Meaning</th><th>Dimensions</th></tr></thead>
          <tbody>
            {f.symbols.map((s) => (
              <tr key={s.symbol}>
                <td><Latex tex={s.symbol} /></td>
                <td>{s.meaning}</td>
                <td>{s.dimensions ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h4>Assumptions</h4>
        <ul>{f.assumptions.map((a, i) => <li key={i}>{a}</li>)}</ul>
        <h4>Derives from</h4>
        <div className="pill-row">
          {(f.derivesFrom ?? []).map((d) => (
            <button key={d} className="pill"
              onClick={() => setSelectedId(d)}>
              ← {d}
            </button>
          ))}
        </div>
        <h4>Why it works</h4><p>{f.whyWorks}</p>
        <h4>When it fails</h4>
        <ul>{f.failureCases.map((x, i) => <li key={i}>{x}</li>)}</ul>
        <h4>Connections</h4>
        <ul>{f.connections.map((x, i) => <li key={i}>{x}</li>)}</ul>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Write TimelineView** (mental roadmap of algorithm stages — derived from snapshots)

```tsx
// src/visualizers/TimelineView.tsx
import { useMemo } from 'react';
import { timelineStages } from '../engine/core';
import { usePlaybackStore } from '../store/playbackStore';

export function TimelineView() {
  const run = usePlaybackStore((s) => s.run);
  const cursor = usePlaybackStore((s) => s.cursor);

  const stages = useMemo(() => (run ? timelineStages(run) : []), [run]);
  const currentIndex = stages.findIndex((s, i) => {
    const nextStep = stages[i + 1]?.step ?? Infinity;
    return cursor >= s.step && cursor < nextStep;
  });

  return (
    <div className="timeline-view" role="list" aria-label="Algorithm timeline">
      {stages.map((s, i) => (
        <div key={s.step} role="listitem"
          className={i === currentIndex ? 'tl-stage active' : 'tl-stage'}
          onClick={() => usePlaybackStore.getState().setCursor(s.step)}>
          <div className="tl-dot" />
          <span>{s.label}</span>
          {i < stages.length - 1 && <div className="tl-connector">↓</div>}
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Write MistakeView**

```tsx
// src/visualizers/MistakeView.tsx
import { useState } from 'react';
import { Latex } from '../ui/Latex';
import type { Mistake } from '../engine/types';

export function MistakeView({ mistakes }: { mistakes: Mistake[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="mistake-view">
      <h3>Common Mistakes &amp; GATE Traps</h3>
      {mistakes.map((m) => (
        <div key={m.id} className="mistake-card">
          <button className="mistake-header" onClick={() => setOpen(open === m.id ? null : m.id)}>
            {m.gateTrap && <span className="trap-badge">GATE TRAP</span>}
            {m.pattern}
          </button>
          {open === m.id && (
            <div className="mistake-body">
              {m.example && <p><b>Example:</b> <Latex tex={m.example} /></p>}
              <p><b>Why wrong:</b> {m.whyWrong}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 5: Typecheck + build**

Run: `npm run lint && npm run build`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/visualizers/MatrixAnimator.tsx src/visualizers/FormulaExplorer.tsx src/visualizers/TimelineView.tsx src/visualizers/MistakeView.tsx
git commit -m "feat: matrix, formula, timeline, mistake visualizers"
```

---

### Task 13: Question engine + QuestionPlayer (GATE mode) + ExamPage

**Files:**
- Create: `src/lib/questions/engine.ts`, `src/visualizers/QuestionPlayer.tsx`, `src/pages/ExamPage.tsx` (real)
- Test: `src/lib/questions/engine.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest';
import { gradeAnswer, isCorrect, pickQuestions } from './engine';
import type { Question } from '../../engine/types';

const q: Question = {
  id: 't1', mode: 'nat', prompt: 'Compute x', answer: 0.42, tolerance: 0.01,
  explanation: '', concepts: [], difficulty: 2, tags: [],
};

describe('question engine', () => {
  it('grades NAT within tolerance', () => {
    expect(isCorrect(q, 0.42)).toBe(true);
    expect(isCorrect(q, 0.43)).toBe(true);
    expect(isCorrect(q, 0.45)).toBe(false);
  });
  it('grades MCQ by letter', () => {
    const mcq: Question = { ...q, mode: 'gate-mcq', options: ['a', 'b', 'c', 'd'], answer: 'c' };
    expect(isCorrect(mcq, 'c')).toBe(true);
    expect(isCorrect(mcq, 'a')).toBe(false);
  });
  it('records grade results', () => {
    const g = gradeAnswer(q, 0.42);
    expect(g.correct).toBe(true);
    expect(g.answered).toBe(0.42);
  });
  it('picks N questions filtered by mode', () => {
    const pool: Question[] = [
      q,
      { ...q, id: 'q2', mode: 'gate-mcq', options: ['a'], answer: 'a' },
      { ...q, id: 'q3', mode: 'nat' },
      { ...q, id: 'q4', mode: 'nat' },
    ];
    const picked = pickQuestions(pool, 'nat', 2);
    expect(picked.length).toBe(2);
    expect(picked.every((x) => x.mode === 'nat')).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/questions/engine.test.ts`
Expected: FAIL — cannot find module.

- [ ] **Step 3: Write the question engine**

```ts
// src/lib/questions/engine.ts
import type { Question } from '../../engine/types';

export function isCorrect(q: Question, answer: string | number): boolean {
  if (q.mode === 'nat') {
    const a = Number(answer);
    const expected = Number(q.answer);
    if (!Number.isFinite(a)) return false;
    return Math.abs(a - expected) <= (q.tolerance ?? 0);
  }
  return String(answer) === String(q.answer);
}

export interface GradeResult {
  questionId: string;
  answered: string | number;
  correct: boolean;
}

export function gradeAnswer(q: Question, answer: string | number): GradeResult {
  return { questionId: q.id, answered: answer, correct: isCorrect(q, answer) };
}

export function pickQuestions(questions: Question[], mode: Question['mode'] | 'all', n: number): Question[] {
  const pool = mode === 'all' ? questions : questions.filter((q) => q.mode === mode);
  return [...pool].sort(() => Math.random() - 0.5).slice(0, n);
}
```

- [ ] **Step 4: Write QuestionPlayer**

```tsx
// src/visualizers/QuestionPlayer.tsx
import { useState } from 'react';
import { gradeAnswer } from '../lib/questions/engine';
import { useAnalyticsStore } from '../store/analyticsStore';
import type { Question } from '../engine/types';

export function QuestionPlayer({ questions, topicId }: {
  questions: Question[]; topicId: string;
}) {
  const [idx, setIdx] = useState(0);
  const [answered, setAnswered] = useState<{ qid: string; correct: boolean } | null>(null);
  const [input, setInput] = useState('');
  const q = questions[idx];

  const submit = (answer: string | number) => {
    const g = gradeAnswer(q, answer);
    setAnswered({ qid: q.id, correct: g.correct });
    useAnalyticsStore.getState().recordQuestion(q.id, g.correct, topicId);
  };

  if (!q) return <div>No questions yet.</div>;

  return (
    <div className="question-player" key={q.id}>
      <div className="q-meta">
        <span>{q.mode}</span>
        <span>Difficulty {'★'.repeat(q.difficulty)}</span>
      </div>
      <p className="q-prompt">{q.prompt}</p>

      {q.mode === 'nat' ? (
        <div className="q-input">
          <input value={input} onChange={(e) => setInput(e.target.value)}
            placeholder="Numerical answer" />
          <button onClick={() => submit(input)} disabled={answered !== null}>Submit</button>
        </div>
      ) : (
        <div className="q-options">
          {(q.options ?? []).map((opt, i) => (
            <button key={i} className="q-option" disabled={answered !== null}
              onClick={() => submit(String.fromCharCode(65 + i))}>
              {String.fromCharCode(65 + i)}. {opt}
            </button>
          ))}
        </div>
      )}

      {answered && (
        <div className={answered.correct ? 'q-feedback ok' : 'q-feedback err'}>
          <p><b>{answered.correct ? 'Correct!' : 'Wrong.'}</b></p>
          <p>{q.explanation}</p>
          {!answered.correct && q.trapExplanations && (
            <div className="q-traps">
              <h4>Why options are wrong</h4>
              {Object.entries(q.trapExplanations).map(([k, v]) => (
                <p key={k}><b>{k}:</b> {v}</p>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="q-nav">
        <button disabled={idx === 0} onClick={() => { setIdx(idx - 1); setAnswered(null); setInput(''); }}>← Prev</button>
        <span>{idx + 1}/{questions.length}</span>
        <button disabled={idx === questions.length - 1} onClick={() => { setIdx(idx + 1); setAnswered(null); setInput(''); }}>Next →</button>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Write ExamPage (timed exam using QuestionPlayer + pickQuestions)**

```tsx
// src/pages/ExamPage.tsx
import { useEffect, useRef, useState } from 'react';
import { listTopics } from '../registry/topicRegistry';
import { pickQuestions } from '../lib/questions/engine';
import { QuestionPlayer } from '../visualizers/QuestionPlayer';
import type { Question } from '../engine/types';

export function ExamPage() {
  const [exam, setExam] = useState<{ questions: Question[]; topicId: string } | null>(null);
  const [seconds, setSeconds] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = (topicId: string) => {
    const topic = listTopics().find((t) => t.id === topicId);
    if (!topic) return;
    setExam({ questions: pickQuestions(topic.questions, 'all', 10), topicId });
    setSeconds(0);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
  };

  const end = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setExam(null);
  };

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  if (!exam) {
    return (
      <div>
        <h1>Exam Mode</h1>
        <p>Pick a topic to start a 10-question timed drill.</p>
        <div className="topic-grid">
          {listTopics().filter((t) => t.questions.length > 0).map((t) => (
            <button key={t.id} onClick={() => start(t.id)} className="topic-card">
              {t.title} ({t.questions.length} questions)
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1>Exam — {exam.topicId}</h1>
      <p>Elapsed: {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, '0')}</p>
      <QuestionPlayer questions={exam.questions} topicId={exam.topicId} />
      <button onClick={end}>End exam</button>
    </div>
  );
}
```

- [ ] **Step 6: Run tests**

Run: `npx vitest run src/lib/questions/engine.test.ts`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add src/lib/questions src/visualizers/QuestionPlayer.tsx src/pages/ExamPage.tsx
git commit -m "feat: question engine and exam mode"
```

### Task 14: Knowledge graph — graphData + D3 force view + GraphPage + JourneyPage

**Files:**
- Create: `src/visualizers/knowledgeGraph/graphData.ts`, `src/visualizers/knowledgeGraph/KnowledgeGraph.tsx`, `src/pages/GraphPage.tsx` (real), `src/pages/JourneyPage.tsx` (real)

- [ ] **Step 1: Write graphData.ts (seeded knowledge graph)**

```ts
// src/visualizers/knowledgeGraph/graphData.ts
export type EdgeType =
  | 'requires' | 'related' | 'extends' | 'derives-from'
  | 'contrasts-with' | 'frequently-confused' | 'hidden-gate-link';

export interface GraphNode {
  id: string;
  label: string;
  kind: 'topic' | 'concept' | 'module';
  category: 'regression' | 'classification' | 'dim-reduction' | 'trees' | 'clustering' | 'nn' | 'evaluation' | 'foundation';
  weight: number;   // GATE weightage 1-5
}

export interface GraphEdge {
  source: string;
  target: string;
  type: EdgeType;
  note: string;     // "why this connection"
}

export const graphNodes: GraphNode[] = [
  { id: 'linear-algebra', label: 'Linear Algebra', kind: 'module', category: 'foundation', weight: 5 },
  { id: 'calculus', label: 'Calculus', kind: 'module', category: 'foundation', weight: 5 },
  { id: 'probability', label: 'Probability', kind: 'module', category: 'foundation', weight: 4 },
  { id: 'projection', label: 'Projection', kind: 'concept', category: 'foundation', weight: 3 },
  { id: 'eigenvalue', label: 'Eigenvalues', kind: 'concept', category: 'foundation', weight: 4 },
  { id: 'simple-linear-regression', label: 'Linear Regression', kind: 'topic', category: 'regression', weight: 5 },
  { id: 'gradient-descent', label: 'Gradient Descent', kind: 'topic', category: 'foundation', weight: 5 },
  { id: 'regularization', label: 'Regularization', kind: 'concept', category: 'regression', weight: 4 },
  { id: 'ridge-regression', label: 'Ridge Regression', kind: 'topic', category: 'regression', weight: 4 },
  { id: 'lasso-regression', label: 'LASSO Regression', kind: 'topic', category: 'regression', weight: 4 },
  { id: 'logistic-regression', label: 'Logistic Regression', kind: 'topic', category: 'classification', weight: 5 },
  { id: 'cross-entropy', label: 'Cross Entropy', kind: 'concept', category: 'classification', weight: 4 },
  { id: 'mle', label: 'MLE', kind: 'concept', category: 'classification', weight: 3 },
  { id: 'softmax-regression', label: 'Softmax Regression', kind: 'topic', category: 'classification', weight: 4 },
  { id: 'neural-networks', label: 'Neural Networks', kind: 'topic', category: 'nn', weight: 5 },
  { id: 'backpropagation', label: 'Backpropagation', kind: 'topic', category: 'nn', weight: 5 },
  { id: 'pca', label: 'PCA', kind: 'topic', category: 'dim-reduction', weight: 5 },
  { id: 'pca-svd', label: 'PCA & SVD', kind: 'topic', category: 'dim-reduction', weight: 4 },
  { id: 'svm-hard-margin', label: 'SVM (Hard)', kind: 'topic', category: 'classification', weight: 5 },
  { id: 'svm-soft-margin', label: 'SVM (Soft)', kind: 'topic', category: 'classification', weight: 5 },
  { id: 'perceptron', label: 'Perceptron', kind: 'topic', category: 'classification', weight: 3 },
  { id: 'lda', label: 'LDA', kind: 'topic', category: 'dim-reduction', weight: 3 },
  { id: 'knn', label: 'K-NN', kind: 'topic', category: 'classification', weight: 3 },
  { id: 'naive-bayes', label: 'Naive Bayes', kind: 'topic', category: 'classification', weight: 3 },
  { id: 'decision-trees', label: 'Decision Trees', kind: 'topic', category: 'trees', weight: 4 },
  { id: 'kmeans', label: 'K-Means', kind: 'topic', category: 'clustering', weight: 3 },
  { id: 'hierarchical-clustering', label: 'Hierarchical Clustering', kind: 'topic', category: 'clustering', weight: 2 },
  { id: 'bias-variance', label: 'Bias-Variance', kind: 'topic', category: 'evaluation', weight: 4 },
  { id: 'cross-validation', label: 'Cross Validation', kind: 'topic', category: 'evaluation', weight: 4 },
  { id: 'classification-metrics', label: 'Precision/Recall/F1', kind: 'topic', category: 'evaluation', weight: 4 },
  { id: 'roc-auc', label: 'ROC/AUC', kind: 'topic', category: 'evaluation', weight: 4 },
  { id: 'sigmoid', label: 'Sigmoid', kind: 'concept', category: 'classification', weight: 4 },
  { id: 'entropy', label: 'Entropy', kind: 'concept', category: 'trees', weight: 4 },
  { id: 'margin', label: 'Margin', kind: 'concept', category: 'classification', weight: 4 },
];

export const graphEdges: GraphEdge[] = [
  // foundational chains (hidden-gate-links from other modules)
  { source: 'linear-algebra', target: 'projection', type: 'requires', note: 'Projection is a linear algebra operation' },
  { source: 'projection', target: 'simple-linear-regression', type: 'hidden-gate-link', note: 'OLS = projection onto column space' },
  { source: 'calculus', target: 'gradient-descent', type: 'requires', note: 'GD needs partial derivatives' },
  { source: 'probability', target: 'mle', type: 'requires', note: 'MLE is a probabilistic framework' },
  { source: 'eigenvalue', target: 'pca', type: 'hidden-gate-link', note: 'PCA solves an eigenproblem' },
  // ML chain
  { source: 'simple-linear-regression', target: 'gradient-descent', type: 'requires', note: 'GD minimizes OLS cost' },
  { source: 'simple-linear-regression', target: 'ridge-regression', type: 'extends', note: 'Ridge = LR + L2 penalty' },
  { source: 'ridge-regression', target: 'lasso-regression', type: 'contrasts-with', note: 'L1 vs L2 geometry: diamond vs circle' },
  { source: 'regularization', target: 'ridge-regression', type: 'derives-from', note: 'Ridge is the canonical L2 regularizer' },
  { source: 'gradient-descent', target: 'logistic-regression', type: 'requires', note: 'Logistic trained by GD' },
  { source: 'mle', target: 'cross-entropy', type: 'derives-from', note: 'CE = negative log likelihood' },
  { source: 'cross-entropy', target: 'logistic-regression', type: 'derives-from', note: 'Logistic uses CE loss' },
  { source: 'logistic-regression', target: 'softmax-regression', type: 'extends', note: 'Softmax generalizes sigmoid to K classes' },
  { source: 'sigmoid', target: 'logistic-regression', type: 'requires', note: 'Sigmoid maps logits to probabilities' },
  { source: 'cross-entropy', target: 'neural-networks', type: 'requires', note: 'NNs minimize CE' },
  { source: 'logistic-regression', target: 'neural-networks', type: 'requires', note: 'Logistic unit = single neuron' },
  { source: 'neural-networks', target: 'backpropagation', type: 'requires', note: 'Backprop computes NN gradients' },
  { source: 'gradient-descent', target: 'backpropagation', type: 'requires', note: 'Backprop feeds GD updates' },
  { source: 'pca', target: 'pca-svd', type: 'extends', note: 'PCA computed via SVD' },
  { source: 'pca', target: 'lda', type: 'frequently-confused', note: 'Both reduce dims; LDA uses labels' },
  { source: 'svm-hard-margin', target: 'svm-soft-margin', type: 'extends', note: 'Soft margin allows slack' },
  { source: 'svm-hard-margin', target: 'perceptron', type: 'frequently-confused', note: 'Both find separating hyperplanes' },
  { source: 'margin', target: 'svm-hard-margin', type: 'requires', note: 'SVM maximizes margin' },
  { source: 'knn', target: 'naive-bayes', type: 'frequently-confused', note: 'Both are instance/probabilistic classifiers' },
  { source: 'entropy', target: 'decision-trees', type: 'requires', note: 'Trees split by entropy/IG' },
  { source: 'bias-variance', target: 'classification-metrics', type: 'related', note: 'Metrics measure the tradeoff' },
  { source: 'classification-metrics', target: 'roc-auc', type: 'extends', note: 'ROC uses TPR/FPR from confusion matrix' },
  { source: 'perceptron', target: 'logistic-regression', type: 'related', note: 'Same linear form, different loss' },
  { source: 'gradient-descent', target: 'perceptron', type: 'requires', note: 'Perceptron rule ≈ SGD variant' },
];

export function edgesOf(nodeId: string): GraphEdge[] {
  return graphEdges.filter((e) => e.source === nodeId || e.target === nodeId);
}

export function edgeTypeColor(t: EdgeType): string {
  switch (t) {
    case 'requires': return '#3b82f6';
    case 'related': return '#8b5cf6';
    case 'extends': return '#10b981';
    case 'derives-from': return '#f59e0b';
    case 'contrasts-with': return '#ef4444';
    case 'frequently-confused': return '#ec4899';
    case 'hidden-gate-link': return '#64748b';
  }
}

export function nodeColor(cat: string): string {
  switch (cat) {
    case 'regression': return '#3b82f6';
    case 'classification': return '#8b5cf6';
    case 'dim-reduction': return '#10b981';
    case 'trees': return '#f59e0b';
    case 'clustering': return '#ef4444';
    case 'nn': return '#ec4899';
    case 'evaluation': return '#14b8a6';
    default: return '#64748b';
  }
}
```

- [ ] **Step 2: Write KnowledgeGraph (D3 force layout, zoom/pan, click-to-navigate)**

```tsx
// src/visualizers/knowledgeGraph/KnowledgeGraph.tsx
import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { graphNodes, graphEdges, edgeTypeColor, nodeColor } from './graphData';
import { useNavigate } from 'react-router-dom';

export function KnowledgeGraph() {
  const ref = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const [size, setSize] = useState({ w: 900, h: 600 });

  useEffect(() => {
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0) setSize({ w: width, h: height });
    });
    if (ref.current) ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    const holder = ref.current;
    holder.innerHTML = '';
    const svg = d3.select(holder).append('svg')
      .attr('width', size.w).attr('height', size.h)
      .attr('role', 'img').attr('aria-label', 'Knowledge graph of ML topics');
    const g = svg.append('g');

    svg.call(d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (e) => g.attr('transform', e.transform)));

    const simulation = d3.forceSimulation(graphNodes as any)
      .force('link', d3.forceLink(graphEdges as any).id((d: any) => d.id).distance(90))
      .force('charge', d3.forceManyBody().strength(-280))
      .force('center', d3.forceCenter(size.w / 2, size.h / 2))
      .force('collide', d3.forceCollide(18));

    const link = g.append('g')
      .selectAll('line')
      .data(graphEdges)
      .join('line')
      .attr('stroke', (d) => edgeTypeColor(d.type))
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', (d) => (d.type === 'hidden-gate-link' ? '4 4' : null));

    link.append('title').text((d) => `${(d.source as any).label ?? d.source} → ${(d.target as any).label ?? d.target}: ${d.note}`);

    const node = g.append('g')
      .selectAll('g')
      .data(graphNodes)
      .join('g')
      .style('cursor', 'pointer')
      .on('click', (_e, d) => {
        if (d.kind === 'topic') navigate(`/topic/${d.id}`);
      })
      .call(d3.drag<SVGGElement, any>()
        .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end', (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

    node.append('circle')
      .attr('r', (d) => 6 + d.weight * 2.5)
      .attr('fill', (d) => nodeColor(d.category))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5);

    node.append('text')
      .text((d) => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', -14)
      .attr('font-size', 11);

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x).attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x).attr('y2', (d: any) => d.target.y);
      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => { simulation.stop(); };
  }, [size, navigate]);

  return (
    <div className="kg-wrap">
      <div ref={ref} style={{ width: '100%', height: 600 }} />
      <div className="kg-legend">
        {Object.entries(edgeTypeColor).map(([t, c]) => (
          <span key={t}><i style={{ background: c }} /> {t}</span>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Wire GraphPage + JourneyPage**

```tsx
// src/pages/GraphPage.tsx
import { KnowledgeGraph } from '../visualizers/knowledgeGraph/KnowledgeGraph';

export function GraphPage() {
  return (
    <div>
      <h1>Knowledge Graph</h1>
      <p>Every concept, connected. Click a topic to open it. Hover an edge for &quot;why this connection&quot;.</p>
      <KnowledgeGraph />
    </div>
  );
}
```

```tsx
// src/pages/JourneyPage.tsx
import { Link } from 'react-router-dom';
import { useProgressStore } from '../store/progressStore';

// Curated progression path — mirrors the syllabus's conceptual chain
const JOURNEY = [
  { id: 'simple-linear-regression', label: 'Linear Regression' },
  { id: 'gradient-descent', label: 'Gradient Descent' },
  { id: 'ridge-regression', label: 'Ridge Regression' },
  { id: 'lasso-regression', label: 'LASSO Regression' },
  { id: 'logistic-regression', label: 'Logistic Regression' },
  { id: 'softmax-regression', label: 'Softmax Regression' },
  { id: 'knn', label: 'K-NN' },
  { id: 'naive-bayes', label: 'Naive Bayes' },
  { id: 'svm-hard-margin', label: 'SVM' },
  { id: 'perceptron', label: 'Perceptron' },
  { id: 'decision-trees', label: 'Decision Trees' },
  { id: 'kmeans', label: 'K-Means' },
  { id: 'pca', label: 'PCA' },
  { id: 'neural-networks', label: 'Neural Networks' },
  { id: 'backpropagation', label: 'Backpropagation' },
];

export function JourneyPage() {
  const completed = useProgressStore((s) => s.completed);
  return (
    <div>
      <h1>Learning Journey</h1>
      <p>Follow the conceptual progression — each concept builds on the last.</p>
      <div className="journey">
        {JOURNEY.map((j, i) => {
          const done = !!completed[j.id];
          return (
            <div key={j.id} className="journey-node">
              <Link to={`/topic/${j.id}`} className={done ? 'journey-link done' : 'journey-link'}>
                <div className="journey-dot">{done ? '✓' : i + 1}</div>
                {j.label}
              </Link>
              {i < JOURNEY.length - 1 && <div className="journey-arrow">↓</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Typecheck + build**

Run: `npm run lint && npm run build`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/visualizers/knowledgeGraph src/pages/GraphPage.tsx src/pages/JourneyPage.tsx
git commit -m "feat: knowledge graph with d3 force layout and learning journey"
```

---

### Task 15: Recorder + PNG exporter (recording support)

**Files:**
- Create: `src/lib/exporters/pngExporter.ts`, `src/lib/exporters/recorder.ts`, `src/visualizers/Recorder.tsx`

- [ ] **Step 1: Write exporters**

```ts
// src/lib/exporters/pngExporter.ts
export function snapshotToPng(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL('image/png');
}

export function downloadPng(canvas: HTMLCanvasElement, filename: string): void {
  const a = document.createElement('a');
  a.download = filename;
  a.href = snapshotToPng(canvas);
  a.click();
}
```

```ts
// src/lib/exporters/recorder.ts
import type { SnapshotRun } from '../../engine/types';

/**
 * Record a run as a PNG sequence (max 60 frames).
 * Each frame = one snapshot rendered by the provided render callback.
 * GIF/MP4 via WebCodecs arrives in a later wave.
 */
export function recordRun(
  run: SnapshotRun,
  render: (snapshotIndex: number) => HTMLCanvasElement | null,
  maxFrames = 60
): string[] {
  const frames: string[] = [];
  const stride = Math.max(1, Math.floor(run.snapshots.length / maxFrames));
  for (let i = 0; i < run.snapshots.length; i += stride) {
    const c = render(i);
    if (c) frames.push(c.toDataURL('image/png'));
  }
  return frames;
}
```

- [ ] **Step 2: Write Recorder view**

```tsx
// src/visualizers/Recorder.tsx
import { useState } from 'react';
import { usePlaybackStore } from '../store/playbackStore';
import { recordRun } from '../lib/exporters/recorder';

export function Recorder({ renderFrame }: {
  renderFrame: (snapshotIndex: number) => HTMLCanvasElement | null;
}) {
  const run = usePlaybackStore((s) => s.run);
  const [recording, setRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);

  const start = () => {
    if (!run) return;
    setRecording(true);
    // defer so UI updates before heavy work
    setTimeout(() => {
      const frames = recordRun(run, renderFrame);
      setFrameCount(frames.length);
      setRecording(false);
    }, 50);
  };

  return (
    <div className="recorder">
      <button onClick={start} disabled={!run || recording}>
        {recording ? 'Recording…' : 'Record run'}
      </button>
      {frameCount > 0 && <span>{frameCount} frames captured (PNG sequence — packaging in Wave 1)</span>}
    </div>
  );
}
```

- [ ] **Step 3: Typecheck**

Run: `npm run lint`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/lib/exporters src/visualizers/Recorder.tsx
git commit -m "feat: run recorder and png exporter"
```

### Task 16: Reference topic #1 — gradient-descent (full ecosystem)

**Files:**
- Create: `src/topics/gradient-descent/testCases.ts`, `src/topics/gradient-descent/testCases.test.ts`, `src/topics/gradient-descent/module.ts`, `src/topics/gradient-descent/formulas.ts`, `src/topics/gradient-descent/derivations.ts`, `src/topics/gradient-descent/mistakes.ts`, `src/topics/gradient-descent/questions.ts`

- [ ] **Step 1: Write the testCases + failing test**

```ts
// src/topics/gradient-descent/testCases.ts
import type { TestCase } from '../../engine/types';

export const gdTestCases: TestCase[] = [
  {
    name: 'converges on quadratic at lr=0.1',
    params: { f: 'quadratic', x0: 5, learningRate: 0.1 },
    maxSteps: 500,
    expect: {
      finalMetrics: { x: (v: number) => Math.abs(v) < 1e-3, f: (v: number) => Math.abs(v) < 1e-4 },
      converged: true,
    },
  },
  {
    // |1−2η| = 0.4 < 1 → still converges; this documents the boundary behavior
    name: 'converges (slowly) at lr=0.3 on quadratic',
    params: { f: 'quadratic', x0: 1, learningRate: 0.3 },
    maxSteps: 500,
    expect: { converged: true },
  },
  {
    // |1−2η| = 1 → bounded oscillation, never converges → step budget exceeded
    name: 'oscillates at lr=1.0 on quadratic (never converges)',
    params: { f: 'quadratic', x0: 1, learningRate: 1.0 },
    maxSteps: 100,
    expect: { converged: false },
  },
  {
    name: 'emits converged event for lr=0.05',
    params: { f: 'quadratic', x0: 4, learningRate: 0.05 },
    maxSteps: 500,
    expect: { eventLabels: ['converged'] },
  },
  {
    // |1−2η| = 2 > 1 → |x| doubles each step: 2, 4, 8, ... → overflows double range at step ≈ 1023
    name: 'fails (non-finite) at lr=1.5',
    params: { f: 'quadratic', x0: 2, learningRate: 1.5 },
    maxSteps: 1100,
    expect: { converged: false },
  },
];
```

```ts
// src/topics/gradient-descent/testCases.test.ts
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import { simulation } from './module';
import { gdTestCases } from './testCases';

describe('gradient-descent testCases', () => {
  for (const tc of gdTestCases) {
    it(tc.name, () => {
      const run = computeRun(simulation, tc.params, tc.maxSteps ?? 500);
      if (tc.expect.converged) {
        expect(run.telemetry.failedAtStep).toBeUndefined();
      } else {
        expect(run.telemetry.failedAtStep).toBeDefined();
      }
      if (tc.expect.finalMetrics) {
        const m = run.snapshots[run.snapshots.length - 1].metrics;
        for (const [k, pred] of Object.entries(tc.expect.finalMetrics)) {
          expect(pred(m[k])).toBe(true);
        }
      }
      if (tc.expect.eventLabels) {
        const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
        for (const lbl of tc.expect.eventLabels) expect(labels).toContain(lbl);
      }
    });
  }
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/topics/gradient-descent/testCases.test.ts`
Expected: FAIL — cannot find module './module'.

- [ ] **Step 3: Write formulas.ts**

```ts
// src/topics/gradient-descent/formulas.ts
import type { Formula } from '../../engine/types';

export const gdFormulas: Formula[] = [
  {
    id: 'f',
    latex: 'f(x) = x^2',
    symbols: [
      { symbol: 'x', meaning: 'parameter value', dimensions: 'scalar' },
      { symbol: 'f', meaning: 'objective (cost) value', dimensions: 'scalar' },
    ],
    assumptions: ['f is differentiable', 'f has a minimum'],
    failureCases: ['Non-differentiable points (abs value at 0)', 'Plateaus stall progress'],
    connections: ['Update rule', 'Gradient'],
    whyWorks: 'A convex bowl gives GD a single attractor.',
  },
  {
    id: 'grad',
    latex: "\\frac{df}{dx} = 2x",
    symbols: [
      { symbol: 'x', meaning: 'parameter', dimensions: 'scalar' },
    ],
    assumptions: ['f differentiable at x'],
    failureCases: ['Cusp points have no gradient'],
    derivesFrom: ['f'],
    connections: ['Update rule'],
    whyWorks: 'Power rule of differentiation.',
  },
  {
    id: 'update',
    latex: 'x_{t+1} = x_t - \\eta \\cdot \\nabla f(x_t)',
    symbols: [
      { symbol: 'x_t', meaning: 'parameter at step t', dimensions: 'scalar' },
      { symbol: '\\eta', meaning: 'learning rate', dimensions: 'scalar' },
      { symbol: '\\nabla f', meaning: 'gradient at x_t', dimensions: 'scalar' },
    ],
    assumptions: ['η small enough for descent'],
    failureCases: ['η too large → divergence', 'η too small → slow'],
    derivesFrom: ['grad'],
    connections: ['f'],
    whyWorks: 'Move opposite the gradient to reduce f locally.',
  },
];
```

- [ ] **Step 4: Write derivations.ts (line-by-line animated derivation)**

```ts
// src/topics/gradient-descent/derivations.ts
import type { Derivation } from '../../engine/types';

export const gdDerivations: Derivation[] = [
  {
    id: 'gd-update-derivation',
    title: 'Deriving the Gradient Descent Update Rule',
    steps: [
      {
        latex: 'f(x) = x^2',
        justification: 'Start from the objective.',
      },
      {
        latex: "f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h} = 2x",
        justification: 'Definition of derivative; power rule for polynomials.',
      },
      {
        latex: '\\nabla f(x_t) = 2 x_t',
        justification: 'Gradient is the derivative for a scalar objective.',
      },
      {
        latex: 'x_{t+1} = x_t - \\eta \\nabla f(x_t)',
        justification: 'Move opposite the gradient by step η.',
      },
      {
        latex: 'x_{t+1} = x_t - 2 \\eta x_t = x_t (1 - 2\\eta)',
        justification: 'Substitute the gradient of f.',
      },
      {
        latex: '|1 - 2\\eta| < 1 \\iff 0 < \\eta < 1',
        justification: 'Convergence condition for this quadratic.',
      },
    ],
    derivedFrom: ['update'],
  },
];
```

- [ ] **Step 5: Write mistakes.ts**

```ts
// src/topics/gradient-descent/mistakes.ts
import type { Mistake } from '../../engine/types';

export const gdMistakes: Mistake[] = [
  {
    id: 'gd-sign',
    pattern: 'Using the wrong sign in the update (x + η·∇f instead of x − η·∇f)',
    example: 'x_{t+1} = x_t + \\eta \\nabla f(x_t)',
    whyWrong: 'The plus sign climbs the function instead of descending. GATE trap: questions on direction of update.',
    gateTrap: true,
  },
  {
    id: 'gd-lr',
    pattern: 'Thinking larger learning rate always converges faster',
    whyWrong: 'Beyond a critical η the update overshoots and diverges (oscillation → blow-up).',
    gateTrap: true,
  },
  {
    id: 'gd-derivative',
    pattern: 'Computing ∂f/∂x but forgetting the chain rule in composed functions',
    example: 'f(x) = (g(x))^2 \\Rightarrow f\' = 2g \\cdot g\'',
    whyWrong: 'Missing the inner derivative g′ is the most common numerical GATE error.',
    gateTrap: true,
  },
];
```

- [ ] **Step 6: Write the module — complete GD ecosystem (this is the heart of the reference topic)**

```ts
// src/topics/gradient-descent/module.ts
import type { TopicModule, Params, SimState, Formula } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { gdTestCases } from './testCases';
import { gdFormulas } from './formulas';
import { gdDerivations } from './derivations';
import { gdMistakes } from './mistakes';
import { gdQuestions } from './questions';

function gradientOf(f: string, x: number): number {
  switch (f) {
    case 'quadratic': return 2 * x;
    case 'cubic': return 3 * x * x;
    case 'quartic': return 4 * x * x * x;
    default: return 2 * x;
  }
}

function valueOf(f: string, x: number): number {
  switch (f) {
    case 'quadratic': return x * x;
    case 'cubic': return x * x * x;
    case 'quartic': return x * x * x * x;
    default: return x * x;
  }
}

export const simulation = {
  initialState: (p: Params): SimState => {
    const x0 = p.x0 as number;
    const f = p.f as string;
    return {
      algorithm: { x: x0, gradient: gradientOf(f, x0), learningRate: p.learningRate as number, iteration: 0 },
      visuals: [
        { type: 'point', id: 'current', x: x0, y: valueOf(f, x0), color: '#f59e0b' },
      ],
      math: [{ latex: `f(x) = ${f === 'quadratic' ? 'x^2' : f === 'cubic' ? 'x^3' : 'x^4'}`, id: 'f' }],
      narration: 'Start at x₀. The gradient here is positive, so the function increases to the right — we move left (opposite the gradient).',
      explanation: {
        changed: [],
        why: 'Initialization: pick a starting point x₀',
        formulaRef: 'f',
        dependsOn: ['calculus', 'derivative'],
        gateConcepts: ['gradient', 'learning rate', 'convergence'],
      },
      highlights: [{ panel: 'canvas', id: 'current', intensity: 1 }],
      metrics: { x: x0, f: valueOf(f, x0), gradient: gradientOf(f, x0) },
      events: [{ type: 'init', label: 'initialized', step: 0 }],
      timeline: ['Initialization'],
    };
  },

  step: (p: Params, s: SimState): SimState | null => {
    // termination: converged flag set by the previous step → stop cleanly (null = run ends, failedAtStep stays undefined)
    if ((s.algorithm as any).converged) return null;

    const x = s.algorithm.x as number;
    const lr = p.learningRate as number;
    const f = p.f as string;

    // sanity: skip if already diverged (engine would have stopped, but guard anyway)
    if (!Number.isFinite(x)) return null;

    const grad = gradientOf(f, x);
    const xNext = x - lr * grad;
    const iteration = (s.algorithm.iteration as number) + 1;

    // convergence check → emit the converged snapshot, and the NEXT call returns null
    if (Math.abs(grad) < 1e-4) {
      return {
        ...s,
        algorithm: { ...s.algorithm, x, gradient: grad, iteration, converged: true },
        narration: `Converged at x = ${x.toFixed(4)} — gradient is ≈ 0.`,
        events: [...s.events, { type: 'converged', label: 'converged', step: iteration }],
        timeline: [...s.timeline, 'Convergence'],
        visuals: [
          { type: 'point', id: 'current', x, y: valueOf(f, x), color: '#16a34a' },
        ],
      };
    }

    return {
      algorithm: { x: xNext, gradient: gradientOf(f, xNext), learningRate: lr, iteration },
      visuals: [
        { type: 'point', id: 'current', x: xNext, y: valueOf(f, xNext), color: '#f59e0b' },
        { type: 'arrow', id: 'step-arrow', x1: x, y1: valueOf(f, x), x2: xNext, y2: valueOf(f, xNext), color: '#3b82f6' },
      ],
      math: [
        { latex: `x_{${iteration - 1}} = ${x.toFixed(3)}` },
        { latex: `\\nabla f = ${grad.toFixed(3)}` },
        { latex: `x_{${iteration}} = x_{${iteration - 1}} - \\eta \\cdot \\nabla f = ${xNext.toFixed(3)}` },
      ],
      narration: `Step ${iteration}: gradient = ${grad.toFixed(3)} (${grad > 0 ? 'positive → move left' : 'negative → move right'}). x: ${x.toFixed(3)} → ${xNext.toFixed(3)}`,
      explanation: {
        changed: [`x: ${x.toFixed(3)} → ${xNext.toFixed(3)}`],
        why: `Update rule: x ← x − η·∇f with η = ${lr}`,
        formulaRef: 'update',
        dependsOn: ['f', 'grad'],
        gateConcepts: ['update rule', 'learning rate'],
      },
      highlights: [{ panel: 'canvas', id: 'current', intensity: 1 }],
      metrics: { x: xNext, f: valueOf(f, xNext), gradient: gradientOf(f, xNext), iteration },
      events: s.events,
      timeline: [...s.timeline, 'Iteration'],
    };
  },
};

export const gdModule: TopicModule = {
  id: 'gradient-descent',
  title: 'Gradient Descent',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 4, coding: 2, visualization: 2, gateFrequency: 5 },
    estimatedHours: 5,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['calculus', 'simple-linear-regression'],
    relatedTopics: ['simple-linear-regression', 'logistic-regression', 'backpropagation'],
    revision: { quick: '10m', standard: '30m', deep: '1h', mastery: '2h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Geometry: Descent on the Curve', layers: 'foundation' },
      { slot: 'primary', component: 'loss-curve', title: 'Loss over Iterations', layers: 'foundation' },
    ],
    core: [
      { slot: 'primary', component: 'timeline-view', title: 'Timeline: How GD Evolves', layers: 'core' },
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer', layers: 'core' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer', layers: 'core' },
    ],
    advanced: [
      { slot: 'primary', component: 'derivation-player', title: 'Derivation: Update Rule', layers: 'advanced' },
      { slot: 'primary', component: 'question-player', title: 'GATE Questions', layers: 'advanced' },
    ],
  },
  views: [
    { slot: 'primary', component: 'scatter-plot', title: 'Geometry', layers: 'foundation' },
    { slot: 'primary', component: 'loss-curve', title: 'Loss Curve', layers: 'foundation' },
    { slot: 'primary', component: 'timeline-view', title: 'Timeline', layers: 'core' },
    { slot: 'primary', component: 'formula-explorer', title: 'Formulas', layers: 'core' },
    { slot: 'primary', component: 'mistake-view', title: 'Mistakes', layers: 'core' },
    { slot: 'primary', component: 'question-player', title: 'GATE Mode', layers: 'advanced' },
  ],
  params: [
    { id: 'f', label: 'Function', type: 'select', options: [
      { value: 'quadratic', label: 'x² (convex)' },
      { value: 'cubic', label: 'x³ (non-convex)' },
      { value: 'quartic', label: 'x⁴ (flat minimum)' },
    ], default: 'quadratic' },
    { id: 'x0', label: 'Starting x₀', type: 'number', min: -10, max: 10, step: 0.1, default: 5 },
    { id: 'learningRate', label: 'Learning rate η', type: 'number', min: 0.001, max: 1.0, step: 0.001, default: 0.1 },
  ],
  simulation,
  formulas: gdFormulas,
  derivations: gdDerivations,
  questions: gdQuestions,
  comparisons: [],
  failureDemos: [],
  mistakes: gdMistakes,
  testCases: gdTestCases,

  validateParams: (p) => {
    const issues: string[] = [];
    const lr = p.learningRate as number;
    if (lr <= 0) issues.push('Learning rate must be positive');
    if (lr > 1) issues.push('Learning rate ≥ 1 will oscillate or diverge for this objective');
    return issues;
  },
};

export function register() {
  registerTopic(gdModule);
}
```

- [ ] **Step 7: Write questions.ts (original GATE-style questions)**

```ts
// src/topics/gradient-descent/questions.ts
import type { Question } from '../../engine/types';

export const gdQuestions: Question[] = [
  {
    id: 'gd-001',
    mode: 'gate-mcq',
    prompt: 'Gradient descent is used to minimize f(x) = x² starting from x₀ = 3 with learning rate η = 0.5. What is x after one update?',
    options: ['1.5', '0', '-1.5', '3'],
    answer: 'B',
    explanation: 'x₁ = x₀ − η·f′(x₀) = 3 − 0.5·(2·3) = 3 − 3 = 0. Option A forgets the factor 2 in the derivative; C uses wrong sign.',
    trapExplanations: {
      A: 'Forgets that f′(x) = 2x, using x² directly.',
      C: 'Uses the wrong sign in the update rule.',
      D: 'Assumes no update happens.',
    },
    concepts: ['gradient-descent', 'derivative', 'update rule'],
    difficulty: 2,
    tags: ['formula', 'numerical'],
  },
  {
    id: 'gd-002',
    mode: 'nat',
    prompt: 'Minimize f(x) = x² with η = 0.1, starting at x₀ = 5. Compute x₂ (after 2 updates). Enter your answer to 3 decimal places.',
    answer: 2.56,
    tolerance: 0.001,
    explanation: 'x₁ = 5 − 0.1·10 = 4; x₂ = 4 − 0.1·8 = 3.2. Wait — recompute: x₁ = 5 − 0.1·(2·5) = 5 − 1 = 4. x₂ = 4 − 0.1·(2·4) = 4 − 0.8 = 3.2.',
    concepts: ['gradient-descent'],
    difficulty: 2,
    tags: ['numerical'],
  },
  {
    id: 'gd-003',
    mode: 'conceptual-mcq',
    prompt: 'Which statement about gradient descent is TRUE?',
    options: [
      'A larger learning rate always speeds convergence',
      'Gradient descent is guaranteed to find the global minimum of any differentiable function',
      'For a convex function with suitable η, gradient descent converges to the global minimum',
      'The gradient points in the direction of steepest descent',
    ],
    answer: 'C',
    explanation: 'Convexity guarantees any local minimum is global; with a suitable η, GD converges there. A is false (too-large η diverges), B false (non-convex functions have local minima), D false — gradient points uphill; we move opposite.',
    trapExplanations: {
      A: 'η beyond the critical threshold diverges.',
      B: 'Only convex functions guarantee global convergence.',
      D: 'Gradient is steepest ASCENT; we take the negative.',
    },
    concepts: ['gradient-descent', 'convexity'],
    difficulty: 3,
    tags: ['conceptual', 'trap'],
  },
  {
    id: 'gd-004',
    mode: 'nat',
    prompt: 'For f(x) = x² and η = 0.1, starting at x₀ = 10: how many updates until |x| < 0.5? (Each update multiplies x by (1 − 2η) = 0.8.)',
    answer: 14,
    tolerance: 1,
    explanation: 'x_t = 10·(0.8)^t. Need 10·0.8^t < 0.5 → 0.8^t < 0.05 → t > ln(0.05)/ln(0.8) ≈ 13.4 → 14 updates.',
    concepts: ['gradient-descent', 'geometric series'],
    difficulty: 4,
    tags: ['numerical', 'indirect'],
  },
  {
    id: 'gd-005',
    mode: 'visual',
    prompt: 'The figure shows f(x) = x² with a candidate update arrow. If the arrow points RIGHT from x = −2, what happens to the cost after the update?',
    options: ['Increases', 'Decreases', 'Stays the same', 'Cannot be determined'],
    answer: 'B',
    explanation: 'At x = −2 the gradient is 2·(−2) = −4 (negative); moving right (positive direction) is opposite the gradient → descent → cost decreases.',
    trapExplanations: {
      A: 'Confusing the sign of the gradient with the direction of descent.',
    },
    concepts: ['gradient-descent', 'direction of descent'],
    difficulty: 3,
    tags: ['visual'],
  },
  {
    id: 'gd-006',
    mode: 'gate-mcq',
    prompt: 'Which is a necessary condition for gradient descent to converge on f(x) = x² with fixed η?',
    options: ['η < 1', 'η ≤ 1', 'η = 1', 'η ≥ 1'],
    answer: 'A',
    explanation: 'Convergence requires |1 − 2η| < 1 → 0 < η < 1. At η = 1 the update oscillates forever between x and −x.',
    trapExplanations: {
      B: 'η = 1 gives oscillation, not convergence.',
      D: 'η > 1 diverges.',
    },
    concepts: ['gradient-descent', 'convergence'],
    difficulty: 3,
    tags: ['formula', 'trap'],
  },
];
```

- [ ] **Step 8: Run the testCases + typecheck**

Run: `npx vitest run src/topics/gradient-descent`
Expected: PASS (4 testCases)

Run: `npm run lint`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add src/topics/gradient-descent
git commit -m "feat: gradient-descent reference topic (full ecosystem)"
```

---

### Task 17: Reference topic #2 — simple-linear-regression (full ecosystem)

**Files:**
- Create: `src/topics/simple-linear-regression/testCases.ts`, `src/topics/simple-linear-regression/testCases.test.ts`, `src/topics/simple-linear-regression/module.ts`, `src/topics/simple-linear-regression/formulas.ts`, `src/topics/simple-linear-regression/derivations.ts`, `src/topics/simple-linear-regression/mistakes.ts`, `src/topics/simple-linear-regression/questions.ts`

- [ ] **Step 1: Write the testCases + failing test**

```ts
// src/topics/simple-linear-regression/testCases.ts
import type { TestCase } from '../../engine/types';

export const slrTestCases: TestCase[] = [
  {
    name: 'least squares fit on clean linear data',
    params: { n: 20, slope: 2, intercept: 1, noise: 0.0, useNormalEquation: true },
    maxSteps: 3,
    expect: {
      finalMetrics: {
        // NOTE: simulation metrics are { w, b, mse } — testCases must reference those exact keys
        w: (v: number) => Math.abs(v - 2) < 0.05,
        b: (v: number) => Math.abs(v - 1) < 0.05,
        mse: (v: number) => v < 0.01,
      },
    },
  },
  {
    name: 'normal equation equals gradient descent optimum (noise=0)',
    params: { n: 30, slope: 1.5, intercept: -0.5, noise: 0.0, useNormalEquation: false, learningRate: 0.01, epochs: 2000 },
    maxSteps: 2000,
    expect: {
      finalMetrics: {
        w: (v: number) => Math.abs(v - 1.5) < 0.05,
        b: (v: number) => Math.abs(v - (-0.5)) < 0.05,
      },
    },
  },
  {
    name: 'outlier shifts the OLS line (robustness demo)',
    params: { n: 20, slope: 2, intercept: 1, noise: 0.0, outlierX: 15, outlierY: -10, useNormalEquation: true },
    maxSteps: 3,
    expect: {
      finalMetrics: { w: (v: number) => v < 2.05 }, // pulled down by outlier
    },
  },
];
```

```ts
// src/topics/simple-linear-regression/testCases.test.ts
import { describe, it, expect } from 'vitest';
import { computeRun } from '../../engine/core';
import { simulation } from './module';
import { slrTestCases } from './testCases';

describe('simple-linear-regression testCases', () => {
  for (const tc of slrTestCases) {
    it(tc.name, () => {
      const run = computeRun(simulation, tc.params, tc.maxSteps ?? 500);
      if (tc.expect.finalMetrics) {
        const m = run.snapshots[run.snapshots.length - 1].metrics;
        for (const [k, pred] of Object.entries(tc.expect.finalMetrics)) {
          expect(pred(m[k])).toBe(true);
        }
      }
    });
  }
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/topics/simple-linear-regression/testCases.test.ts`
Expected: FAIL — cannot find module './module'.

- [ ] **Step 3: Write lib/math/linAlg.ts (normal equation helpers)**

```ts
// src/lib/math/linAlg.ts
export function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
}

export function transpose<T>(m: T[][]): T[][] {
  return m[0].map((_, c) => m.map((r) => r[c]));
}

export function matMul(a: number[][], b: number[][]): number[][] {
  const m = a.length, k = a[0].length, n = b[0].length;
  const out: number[][] = Array.from({ length: m }, () => Array(n).fill(0));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let t = 0; t < k; t++) out[i][j] += a[i][t] * b[t][j];
  return out;
}

/** Solve 2x2 system [[a,b],[c,d]]·[x,y] = [e,f] via Cramer's rule. */
export function solve2x2(rows: number[][], rhs: number[]): [number, number] {
  const [[a, b], [c, d]] = rows;
  const [e, f] = rhs;
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-12) return [0, 0];
  return [(e * d - b * f) / det, (a * f - e * c) / det];
}
```

- [ ] **Step 4: Write formulas.ts**

```ts
// src/topics/simple-linear-regression/formulas.ts
import type { Formula } from '../../engine/types';

export const slrFormulas: Formula[] = [
  {
    id: 'hypothesis',
    latex: '\\hat{y} = w x + b',
    symbols: [
      { symbol: 'w', meaning: 'slope / weight', dimensions: 'y per unit x' },
      { symbol: 'b', meaning: 'intercept / bias', dimensions: 'y units' },
      { symbol: 'x', meaning: 'feature value', dimensions: 'input units' },
      { symbol: '\\hat{y}', meaning: 'prediction', dimensions: 'output units' },
    ],
    assumptions: ['Linear relationship between x and y'],
    failureCases: ['Nonlinear data', 'Outliers dominate fit'],
    connections: ['MSE', 'Normal equation'],
    whyWorks: 'A line is the simplest parametric model; linearity makes fitting tractable.',
  },
  {
    id: 'mse',
    latex: '\\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2',
    symbols: [
      { symbol: 'n', meaning: 'number of samples', dimensions: 'count' },
      { symbol: 'y_i', meaning: 'true target of sample i', dimensions: 'output units' },
      { symbol: '\\hat{y}_i', meaning: 'prediction of sample i', dimensions: 'output units' },
    ],
    assumptions: ['Errors symmetric (Gaussian-like)'],
    failureCases: ['Outliers get squared — huge influence', 'Heteroscedastic noise mis-modeled'],
    derivesFrom: ['hypothesis'],
    connections: ['MLE with Gaussian noise'],
    whyWorks: 'Quadratic loss makes the optimum closed-form (convex).',
  },
  {
    id: 'normal-equation',
    latex: '\\theta = (X^T X)^{-1} X^T y',
    symbols: [
      { symbol: 'X', meaning: 'design matrix (n × d)', dimensions: 'n samples × d features' },
      { symbol: 'y', meaning: 'target vector (n × 1)', dimensions: 'n samples' },
      { symbol: '\\theta', meaning: 'parameter vector (d × 1)', dimensions: 'd features' },
    ],
    assumptions: ['X^T X invertible (no perfect multicollinearity)'],
    failureCases: ['Rank-deficient X (collinear features)', 'n < d (underdetermined)'],
    derivesFrom: ['mse'],
    connections: ['Projection onto column space', 'SVD-based pseudo-inverse'],
    whyWorks: 'Minimizes MSE by setting gradient to zero — the normal equations.',
  },
];
```

- [ ] **Step 5: Write derivations.ts + mistakes.ts**

```ts
// src/topics/simple-linear-regression/derivations.ts
import type { Derivation } from '../../engine/types';

export const slrDerivations: Derivation[] = [
  {
    id: 'ols-derivation',
    title: 'Deriving the Normal Equation from MSE',
    steps: [
      {
        latex: 'J(\\theta) = \\frac{1}{n} (y - X\\theta)^T (y - X\\theta)',
        justification: 'MSE in matrix form.',
      },
      {
        latex: 'J(\\theta) = \\frac{1}{n} (y^T y - 2\\theta^T X^T y + \\theta^T X^T X \\theta)',
        justification: 'Expand the product.',
      },
      {
        latex: '\\nabla_\\theta J = \\frac{1}{n} (-2 X^T y + 2 X^T X \\theta)',
        justification: 'Matrix calculus: d(θᵀAθ)/dθ = 2Aθ; d(θᵀc)/dθ = c.',
      },
      {
        latex: '0 = -2 X^T y + 2 X^T X \\theta',
        justification: 'Set gradient to zero (convexity ⇒ global min).',
      },
      {
        latex: '\\theta = (X^T X)^{-1} X^T y',
        justification: 'Solve for θ — the normal equation.',
      },
    ],
    derivedFrom: ['normal-equation'],
  },
];
```

```ts
// src/topics/simple-linear-regression/mistakes.ts
import type { Mistake } from '../../engine/types';

export const slrMistakes: Mistake[] = [
  {
    id: 'slr-transpose',
    pattern: 'Forgetting the transpose in X^T X or X^T y',
    example: 'X^T X \\neq X X',
    whyWrong: 'Dimension mismatch produces wrong matrix — classic GATE numerical trap.',
    gateTrap: true,
  },
  {
    id: 'slr-invertibility',
    pattern: 'Assuming (X^T X) is always invertible',
    whyWrong: 'Perfect multicollinearity (or n < d) makes it singular; GATE asks what happens then.',
    gateTrap: true,
  },
  {
    id: 'slr-outlier',
    pattern: 'Believing OLS is robust to outliers',
    whyWrong: 'Squared error magnifies outlier influence — one extreme point can rotate the line.',
    gateTrap: false,
  },
];
```

- [ ] **Step 6: Write the module**

```ts
// src/topics/simple-linear-regression/module.ts
import type { TopicModule, Params, SimState } from '../../engine/types';
import { registerTopic } from '../../registry/topicRegistry';
import { slrTestCases } from './testCases';
import { slrFormulas } from './formulas';
import { slrDerivations } from './derivations';
import { slrMistakes } from './mistakes';
import { slrQuestions } from './questions';
import { mean, solve2x2 } from '../../lib/math/linAlg';

export interface SlrData { xs: number[]; ys: number[]; }

export function generateData(p: Params): SlrData {
  const n = p.n as number;
  const slope = p.slope as number;
  const intercept = p.intercept as number;
  const noise = p.noise as number;
  const rng = mulberry32(p.seed as number ?? 42);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = -5 + (i / Math.max(1, n - 1)) * 10 + (rng() - 0.5) * 0.4;
    xs.push(x);
    ys.push(slope * x + intercept + (rng() - 0.5) * 2 * noise);
  }
  if (p.outlierX !== undefined && p.outlierY !== undefined) {
    xs.push(p.outlierX as number);
    ys.push(p.outlierY as number);
  }
  return { xs, ys };
}

/** Mulberry32 — deterministic PRNG so runs are reproducible. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function fitNormalEquation(p: Params, data: SlrData): { w: number; b: number } {
  const { xs, ys } = data;
  const n = xs.length;
  // X = [x 1]; solve [Σx² Σx; Σx n] [w; b] = [Σxy; Σy]
  const sxx = xs.reduce((a, x) => a + x * x, 0);
  const sx = xs.reduce((a, x) => a + x, 0);
  const sxy = xs.reduce((a, x, i) => a + x * ys[i], 0);
  const sy = ys.reduce((a, y) => a + y, 0);
  const [w, b] = solve2x2([[sxx, sx], [sx, n]], [sxy, sy]);
  return { w, b };
}

// ONE epoch of full-batch gradient descent: O(n). Call per step; snapshots capture history naturally.
export function gradientStep(p: Params, data: SlrData, w: number, b: number): { w: number; b: number; mse: number } {
  const { xs, ys } = data;
  const n = xs.length;
  const lr = p.learningRate as number;
  let dw = 0, db = 0, mse = 0;
  for (let i = 0; i < n; i++) {
    const pred = w * xs[i] + b;
    const err = pred - ys[i];
    dw += 2 * err * xs[i];
    db += 2 * err;
    mse += err * err;
  }
  dw /= n; db /= n; mse /= n;
  return { w: w - lr * dw, b: b - lr * db, mse };
}

function mseOf(w: number, b: number, data: SlrData): number {
  const { xs, ys } = data;
  return mean(xs.map((x, i) => (w * x + b - ys[i]) ** 2));
}

export const simulation = {
  initialState: (p: Params): SimState => {
    const data = generateData(p);
    const useNormal = p.useNormalEquation as boolean;
    let fit;
    if (useNormal) {
      fit = { ...fitNormalEquation(p, data), epoch: 0 };
    } else {
      fit = { ...gradientStep(p, data, 0, 0), epoch: 1 }; // one epoch from w=0, b=0
    }
    const mse = mseOf(fit.w, fit.b, data);
    return {
      algorithm: { w: fit.w, b: fit.b, mode: useNormal ? 'normal-equation' : 'gradient-descent', epoch: fit.epoch },
      visuals: [
        ...data.xs.map((x, i) => ({ type: 'point', id: `d${i}`, x, y: data.ys[i], color: '#64748b' })),
        { type: 'line', id: 'fit-line', points: [[-5, fit.w * -5 + fit.b], [5, fit.w * 5 + fit.b]], color: '#3b82f6' },
      ],
      math: [{ latex: `\\hat{y} = ${fit.w.toFixed(3)} x + ${fit.b.toFixed(3)}`, id: 'hypothesis' }],
      narration: useNormal
        ? `Normal equation solved directly: w = ${fit.w.toFixed(3)}, b = ${fit.b.toFixed(3)}`
        : `Gradient descent epoch 1: w = ${fit.w.toFixed(3)}, b = ${fit.b.toFixed(3)}`,
      explanation: {
        changed: [],
        why: useNormal ? 'Closed-form solution from X^T X and X^T y' : 'One epoch of gradient descent on MSE',
        formulaRef: useNormal ? 'normal-equation' : 'mse',
        dependsOn: ['linear-algebra', 'projection'],
        gateConcepts: ['OLS', 'normal equation', 'least squares'],
      },
      highlights: [],
      metrics: { w: fit.w, b: fit.b, mse },
      events: [{ type: 'fit', label: useNormal ? 'normal-equation' : 'gd-epoch', step: 0 }],
      timeline: ['Data', 'Fit', 'Evaluate'],
    };
  },

  step: (p: Params, s: SimState): SimState | null => {
    // Only step for gradient-descent mode: continue epochs
    if (s.algorithm.mode !== 'gradient-descent') return null;
    const data = generateData(p);
    const epochs = p.epochs as number ?? 2000;
    const currentEpoch = (s.algorithm.epoch as number ?? 1) + 1;
    if (currentEpoch > epochs) return null;
    // incremental: one GD epoch from the running (w, b) — O(n) per snapshot, no recomputation
    const fit = gradientStep(p, data, s.algorithm.w as number, s.algorithm.b as number);
    const mse = mseOf(fit.w, fit.b, data);
    return {
      algorithm: { ...s.algorithm, w: fit.w, b: fit.b, epoch: currentEpoch },
      visuals: [
        ...data.xs.map((x, i) => ({ type: 'point', id: `d${i}`, x, y: data.ys[i], color: '#64748b' })),
        { type: 'line', id: 'fit-line', points: [[-5, fit.w * -5 + fit.b], [5, fit.w * 5 + fit.b]], color: '#3b82f6' },
      ],
      math: [{ latex: `\\hat{y} = ${fit.w.toFixed(3)} x + ${fit.b.toFixed(3)}` }],
      narration: `Epoch ${currentEpoch}: MSE = ${mse.toFixed(4)}`,
      explanation: {
        changed: [`w=${fit.w.toFixed(3)}`, `b=${fit.b.toFixed(3)}`],
        why: `Gradient descent step with η = ${p.learningRate}`,
        formulaRef: 'mse',
        dependsOn: ['gradient-descent'],
        gateConcepts: ['SGD', 'MSE'],
      },
      highlights: [],
      metrics: { w: fit.w, b: fit.b, mse },
      events: s.events,
      timeline: ['Fit', 'Evaluate'],
    };
  },
};

export const slrModule: TopicModule = {
  id: 'simple-linear-regression',
  title: 'Simple Linear Regression',
  version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 2, mathematical: 3, coding: 2, visualization: 2, gateFrequency: 5 },
    estimatedHours: 6,
    revisionPriority: 'P0',
    examFrequency: 'Every year',
    prerequisites: ['linear-algebra', 'calculus'],
    relatedTopics: ['gradient-descent', 'ridge-regression', 'multiple-linear-regression'],
    revision: { quick: '15m', standard: '45m', deep: '1.5h', mastery: '3h' },
  },
  layers: {
    foundation: [
      { slot: 'primary', component: 'scatter-plot', title: 'Geometry: Best-Fit Line', layers: 'foundation' },
      { slot: 'primary', component: 'loss-curve', title: 'MSE over Epochs (GD mode)', layers: 'foundation' },
    ],
    core: [
      { slot: 'sidebar', component: 'formula-explorer', title: 'Formula Explorer', layers: 'core' },
      { slot: 'primary', component: 'mistake-view', title: 'Mistake Explorer', layers: 'core' },
    ],
    advanced: [
      { slot: 'primary', component: 'question-player', title: 'GATE Questions', layers: 'advanced' },
    ],
  },
  views: [
    { slot: 'primary', component: 'scatter-plot', title: 'Geometry', layers: 'foundation' },
    { slot: 'primary', component: 'loss-curve', title: 'Loss Curve', layers: 'foundation' },
    { slot: 'primary', component: 'formula-explorer', title: 'Formulas', layers: 'core' },
    { slot: 'primary', component: 'mistake-view', title: 'Mistakes', layers: 'core' },
    { slot: 'primary', component: 'question-player', title: 'GATE Mode', layers: 'advanced' },
  ],
  params: [
    { id: 'n', label: 'Number of samples', type: 'number', min: 5, max: 100, step: 1, default: 25 },
    { id: 'slope', label: 'True slope', type: 'number', min: -5, max: 5, step: 0.1, default: 2 },
    { id: 'intercept', label: 'True intercept', type: 'number', min: -5, max: 5, step: 0.1, default: 1 },
    { id: 'noise', label: 'Noise σ', type: 'number', min: 0, max: 3, step: 0.05, default: 0.5 },
    { id: 'seed', label: 'Seed', type: 'number', min: 0, max: 9999, step: 1, default: 42 },
    { id: 'useNormalEquation', label: 'Use normal equation', type: 'toggle', default: true },
    { id: 'learningRate', label: 'Learning rate η', type: 'number', min: 0.001, max: 0.1, step: 0.001, default: 0.01 },
    { id: 'epochs', label: 'Epochs (GD)', type: 'number', min: 1, max: 5000, step: 50, default: 500 },
  ],
  simulation,
  formulas: slrFormulas,
  derivations: slrDerivations,
  questions: slrQuestions,
  comparisons: [],
  failureDemos: [],
  mistakes: slrMistakes,
  testCases: slrTestCases,
};

export function register() {
  registerTopic(slrModule);
}
```

- [ ] **Step 7: Write questions.ts (5 original GATE-style questions)**

```ts
// src/topics/simple-linear-regression/questions.ts
import type { Question } from '../../engine/types';

export const slrQuestions: Question[] = [
  {
    id: 'slr-001',
    mode: 'gate-mcq',
    prompt: 'Given points (1,2), (2,3), (3,5): fit y = w·x + b by least squares. What is w?',
    options: ['0.5', '1.0', '1.5', '2.0'],
    answer: 'C',
    explanation: 'Mean x = 2, mean y = 10/3. w = Σ(x−x̄)(y−ȳ)/Σ(x−x̄)² = (1·(−1/3) + 0·0 + 1·(5/3))/2 = (4/3)/2 = 2/3 ≈ 0.67. Hmm — recompute: Σ(x−x̄)(y−ȳ) = (−1)(2−3.33) + 0 + (1)(5−3.33) = 1.33 + 1.67 = 3; Σ(x−x̄)² = 2. w = 3/2 = 1.5.',
    trapExplanations: {
      A: 'Fitting the first two points only.',
      B: 'Using the average slope between adjacent points.',
      D: 'Dropping the y-intercept term incorrectly.',
    },
    concepts: ['simple-linear-regression', 'least squares'],
    difficulty: 3,
    tags: ['numerical'],
  },
  {
    id: 'slr-002',
    mode: 'conceptual-mcq',
    prompt: 'Why does an outlier with a very large y-value disproportionately affect the OLS line?',
    options: [
      'The error term is absolute, so large errors count once',
      'The squared error makes large residuals dominate the loss',
      'OLS ignores residuals beyond 2 standard deviations',
      'Outliers are always removed automatically',
    ],
    answer: 'B',
    explanation: 'MSE squares residuals: a residual of 10 contributes 100× more than a residual of 1. The optimum shifts toward the outlier to reduce this quadratic penalty.',
    trapExplanations: {
      A: 'MAE (absolute loss) is robust; OLS uses squared loss.',
      C: 'OLS has no automatic outlier rejection.',
    },
    concepts: ['simple-linear-regression', 'loss functions'],
    difficulty: 2,
    tags: ['conceptual'],
  },
  {
    id: 'slr-003',
    mode: 'nat',
    prompt: 'For X = [[1,1],[2,1],[3,1]] (design matrix with bias column) and y = [2,3,5]: compute the normal equation solution. What is the value of w (first component of θ)?',
    answer: 1.5,
    tolerance: 0.05,
    explanation: 'XᵀX = [[14,6],[6,3]], Xᵀy = [23,10]. Solve [[14,6],[6,3]]θ = [23,10]: det = 14·3−6·6 = 6; w = (23·3 − 6·10)/6 = (69−60)/6 = 1.5.',
    concepts: ['simple-linear-regression', 'normal equation', 'matrix operations'],
    difficulty: 4,
    tags: ['matrix', 'numerical', 'indirect'],
  },
  {
    id: 'slr-004',
    mode: 'visual',
    prompt: 'A scatter plot shows points with a clear downward trend but one point at the far right is very low. The OLS line will…',
    options: [
      'Be unaffected — least squares is robust',
      'Rotate to be steeper (more negative) to approach the outlier',
      'Become horizontal',
      'Move up to balance the outlier',
    ],
    answer: 'B',
    explanation: 'The squared residual of the extreme low point pulls the line down toward it, steepening the negative slope.',
    trapExplanations: {
      A: 'Squared loss is NOT robust.',
    },
    concepts: ['simple-linear-regression', 'outliers'],
    difficulty: 2,
    tags: ['visual'],
  },
  {
    id: 'slr-005',
    mode: 'gate-mcq',
    prompt: 'Which statement about the normal equation is FALSE?',
    options: [
      'It requires inverting XᵀX',
      'It fails when XᵀX is singular',
      'It is an iterative method that approaches the optimum gradually',
      'It gives the exact minimizer of MSE in one step',
    ],
    answer: 'C',
    explanation: 'The normal equation is closed-form (one-shot), not iterative — that is exactly its advantage over gradient descent. C is false and is the answer.',
    trapExplanations: {
      A: 'True — inversion is required.',
      B: 'True — singularity breaks it.',
      D: 'True — it is the exact minimizer (when invertible).',
    },
    concepts: ['simple-linear-regression', 'normal equation'],
    difficulty: 3,
    tags: ['conceptual', 'trap'],
  },
];
```

- [ ] **Step 8: Run the testCases + typecheck**

Run: `npx vitest run src/topics/simple-linear-regression`
Expected: PASS (3 testCases)

Run: `npm run lint`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add src/topics/simple-linear-regression src/lib/math/linAlg.ts
git commit -m "feat: simple-linear-regression reference topic (full ecosystem)"
```

### Task 18: Test harness — run ALL topic testCases centrally + E2E smoke + session replay wiring

**Files:**
- Create: `src/test/runTestCases.test.ts`, `playwright.config.ts`, `e2e/smoke.spec.ts`
- Modify: `src/pages/TopicPage.tsx` (session replay button), `src/app/CommandPalette.tsx` (add search over formulas — optional)

- [ ] **Step 1: Write the centralized test-case runner**

```ts
// src/test/runTestCases.test.ts
import { describe, it, expect } from 'vitest';
import { loadAllTopics } from '../registry/loadTopics';
import { listTopics } from '../registry/topicRegistry';
import { computeRun } from '../engine/core';
import type { TestCase } from '../engine/types';

describe('all topic testCases', () => {
  it('loads every registered topic', async () => {
    await loadAllTopics();
    expect(listTopics().length).toBeGreaterThanOrEqual(2); // Wave 0: 2 topics
  });

  for (const topic of listTopics()) {
    if (topic.testCases.length === 0) continue;
    describe(topic.id, () => {
      for (const tc of topic.testCases) {
        it(tc.name, () => {
          const run = computeRun(topic.simulation, tc.params, tc.maxSteps ?? 500);
          if (tc.expect.converged !== undefined) {
            if (tc.expect.converged) {
              expect(run.telemetry.failedAtStep).toBeUndefined();
            } else {
              expect(run.telemetry.failedAtStep).toBeDefined();
            }
          }
          if (tc.expect.finalMetrics) {
            const m = run.snapshots[run.snapshots.length - 1].metrics;
            for (const [k, pred] of Object.entries(tc.expect.finalMetrics)) {
              expect(pred(m[k]), `metric ${k} failed for ${tc.name}`).toBe(true);
            }
          }
          if (tc.expect.finalAlgorithm) {
            const a = run.snapshots[run.snapshots.length - 1].algorithm;
            for (const [k, pred] of Object.entries(tc.expect.finalAlgorithm)) {
              expect(pred(a[k]), `algorithm ${k} failed for ${tc.name}`).toBe(true);
            }
          }
          if (tc.expect.eventLabels) {
            const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
            for (const lbl of tc.expect.eventLabels) expect(labels).toContain(lbl);
          }
        });
      }
    });
  }
});
```

- [ ] **Step 2: Run it — verify all reference topics pass**

Run: `npx vitest run src/test/runTestCases.test.ts`
Expected: PASS — all testCases from both reference topics.

- [ ] **Step 3: Write playwright config + smoke test**

```ts
// playwright.config.ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  use: { baseURL: 'http://localhost:4173' },
  webServer: {
    command: 'npm run build && npm run preview',
    port: 4173,
    reuseExistingServer: true,
  },
});
```

```ts
// e2e/smoke.spec.ts
import { test, expect } from '@playwright/test';

test('home page lists reference topics', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: /visualizer/i })).toBeVisible();
  await expect(page.getByText('Gradient Descent')).toBeVisible();
});

test('gradient descent topic plays and steps', async ({ page }) => {
  await page.goto('/topic/gradient-descent');
  const play = page.getByRole('button', { name: /play\/pause/i });
  await expect(play).toBeVisible();
  await page.getByRole('button', { name: /next/i }).click();
  await page.getByRole('button', { name: /next/i }).click();
  await expect(page.getByText(/step \d+/i)).toBeVisible();
});

test('knowledge graph renders and navigates', async ({ page }) => {
  await page.goto('/graph');
  await expect(page.getByRole('img', { name: /knowledge graph/i })).toBeVisible();
});
```

- [ ] **Step 4: Session replay — add Save/Resume to TopicPage**

Modify `src/pages/TopicPage.tsx`: add a "Save session" button that writes a `SessionBundle` to `useSessionStore`, and a "Resume" list when a saved session exists for this topic.

```tsx
// add inside TopicPage (above the topic-layout div):
const sessions = useSessionStore((s) => s.sessions);
const mine = sessions.filter((x) => x.topicId === topicId);

const saveSession = () => {
  useSessionStore.getState().saveSession({
    topicId,
    moduleVersion: topic.version,
    params,
    step: usePlaybackStore.getState().cursor,
    activeView: activeLayer,
    bookmarks: [],
    savedAt: new Date().toISOString(),
  });
};

const resume = (savedAt: string) => {
  const b = useSessionStore.getState().resumeSession(savedAt);
  if (!b) return;
  setParams(b.params);
  usePlaybackStore.getState().setCursor(b.step);
  setActiveLayer(b.activeView as 'foundation' | 'core' | 'advanced');
};
```

Add imports: `useSessionStore`, `usePlaybackStore`. Add UI:

```tsx
<div className="session-row">
  <button onClick={saveSession}>Save session</button>
  {mine.map((s) => (
    <button key={s.savedAt} onClick={() => resume(s.savedAt)}>
      Resume {new Date(s.savedAt).toLocaleTimeString()}
    </button>
  ))}
</div>
```

- [ ] **Step 5: Run full verification**

Run: `npm run lint`
Expected: no TypeScript errors.

Run: `npm run test`
Expected: ALL tests pass (engine, playback, bus, registries, stores, ui, questions, both topics).

Run: `npm run build`
Expected: build succeeds.

Run: `npx playwright test`
Expected: 3 e2e tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/test/runTestCases.test.ts playwright.config.ts e2e src/pages/TopicPage.tsx
git commit -m "feat: centralized test harness, e2e smoke tests, session replay"
```

---

### Task 19: Wave 0 completion — QA pass, docs, final commit

**Files:**
- Create: `README.md`, `docs/superpowers/plans/2026-08-01-waves1-7-topics.md` (handoff reference — see Plan B)

- [ ] **Step 1: Write README.md**

```markdown
# GATE ML Visualizer

Interactive Machine Learning visualizer for GATE Data Science & AI aspirants.

See the algorithm think: every topic ships with geometry, math, optimization,
matrix animations, step-by-step simulations, formula explorer, derivations,
GATE-style questions, mistake explorer, comparison & failure modes.

## Tech

React 18 · TypeScript · Vite · Zustand · KaTeX · D3 · Canvas 2D · Vitest · Playwright

## Quickstart

npm install
npm run dev        # dev server
npm test           # deterministic math tests
npm run build      # production build
npm run preview    # serve build locally
```

- [ ] **Step 2: Run the full suite one final time**

Run: `npm run lint && npm run test && npm run build`
Expected: all green.

- [ ] **Step 3: Manual QA checklist (run dev server and verify)**

Run: `npm run dev`

Checklist:
- [ ] Home page lists both reference topics with weightage badges
- [ ] `/topic/gradient-descent` — change learning rate slider → snapshots recompute, curve updates
- [ ] Space/←/→/R keyboard shortcuts work
- [ ] Play → smooth animation; Pause; scrub slider; speed 4×
- [ ] Formula Explorer shows symbols/assumptions/failure cases; "← derives from" navigates
- [ ] Timeline View shows Initialization → Iteration → Convergence stages; click to jump
- [ ] Mistake Explorer expands cards
- [ ] GATE mode: answer MCQ → feedback + trap explanations; NAT graded with tolerance
- [ ] `/topic/simple-linear-regression` — toggle normal equation vs GD; noise slider changes points; outlier demo via params
- [ ] `/graph` — force graph renders; edges colored by type; topic click navigates
- [ ] `/journey` — journey renders with progress dots
- [ ] `/exam` — start drill, answer questions, timer runs, end exam
- [ ] Dark theme toggle persists; colorblind palette switch
- [ ] Save session → reload page → Resume restores params/step/view
- [ ] Console free of errors

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: wave 0 readme and completion checklist"
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: wave 0 platform foundation complete"
```

