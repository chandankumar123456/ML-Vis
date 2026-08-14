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
import { DecisionBoundary } from './DecisionBoundary';
import { Eigenviewer } from './Eigenviewer';
import { DistributionView } from './DistributionView';
import { collectDistributions } from './distribution';

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
  // decision-boundary resolves its classifier internally via getClassifier(topic.id);
  // SVM topics later wire supportVectors/marginLines through their topic module.
  registerView('decision-boundary', (p: ViewProps) => (
    <DecisionBoundary snapshot={p.snapshot} params={p.params} topic={p.topic} />
  ));
  // eigenviewer is a readonly view bound to the current snapshot; the axis
  // slider is the only local state (a user override of the snapshot's axis).
  registerView('eigenviewer', (p: ViewProps) => (
    <Eigenviewer snapshot={p.snapshot} params={p.params} />
  ));
  // distribution-view resolves class densities from the snapshot's visuals
  // (topics emit {type:'distribution', label, mean, variance, color} commands
  // alongside their math); x/y ranges are fitted when the topic omits them.
  registerView('distribution-view', (p: ViewProps) => (
    <DistributionView distributions={collectDistributions(p.snapshot?.visuals ?? [])} />
  ));
}
