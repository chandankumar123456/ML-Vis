// src/visualizers/registerViews.tsx
// One-time registration of every visualizer into the view registry.
// ViewHost resolves layer components by id via getView(); without this module
// topic pages render "Unknown view: X" (browser-verified in Wave-0 QA).
// NOTE: .tsx not .ts — contains JSX and tsconfig has jsx: react-jsx.
import { registerView, type ViewProps } from '../registry/viewRegistry';
import { ScatterPlot } from './ScatterPlot';
import { LossCurve } from './LossCurve';
import { TimelineView } from './TimelineView';
import { FormulaExplorer } from './FormulaExplorer';
import { MistakeView } from './MistakeView';
import { QuestionPlayer } from './QuestionPlayer';
import { MatrixAnimator } from './MatrixAnimator';
import { DerivationPlayer } from './DerivationPlayer';
import { ExplainStep } from './ExplainStep';

export function registerAllViews(): void {
  registerView('scatter-plot', (p: ViewProps) => <ScatterPlot {...p} />);
  registerView('loss-curve', (p: ViewProps) => (
    <LossCurve run={p.run ?? null} metricKey={p.topic?.lossMetricKey ?? 'cost'} metricKey2={p.topic?.lossMetricKey2} />
  ));
  registerView('timeline-view', () => <TimelineView />);
  registerView('formula-explorer', (p: ViewProps) => <FormulaExplorer topic={p.topic} />);
  registerView('mistake-view', (p: ViewProps) => <MistakeView topic={p.topic} />);
  registerView('question-player', (p: ViewProps) => <QuestionPlayer topic={p.topic} />);
  registerView('matrix-animator', (p: ViewProps) => <MatrixAnimator snapshot={p.snapshot} />);
  registerView('derivation-player', (p: ViewProps) => <DerivationPlayer topic={p.topic} />);
  registerView('explain-step', (p: ViewProps) => <ExplainStep snapshot={p.snapshot} topic={p.topic} />);
}
