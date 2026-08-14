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
import { TreeBuilder } from './TreeBuilder';
import { ClusterAnimator } from './ClusterAnimator';
import { Dendrogram } from './Dendrogram';

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
  // tree-builder renders decision-tree nodes from {type:'node'} commands with
  // NORMALIZED [0,1] x/y (the topic computes layout); children linkage is
  // explicit via each node's children field, purity (0..1) drives the
  // entropy/gini bar, and canvas highlights ring a node + its ancestor path.
  registerView('tree-builder', (p: ViewProps) => (
    <TreeBuilder snapshot={p.snapshot} params={p.params} />
  ));
  // cluster-animator renders the k-means step: points, centroids, assignment
  // lines and a loss readout (metrics — topic.lossMetricKey preferred — with a
  // {type:'text'} fallback); stable centroid ids across snapshots drive the
  // faint convergence trail.
  registerView('cluster-animator', (p: ViewProps) => (
    <ClusterAnimator snapshot={p.snapshot} params={p.params} topic={p.topic} />
  ));
  // dendrogram renders agglomerative {type:'merge'} commands as an SVG
  // hierarchy (leaves bottom-up, merges at their heights) with a distance
  // axis; clicking a node selects it and highlights its subtree members.
  registerView('dendrogram', (p: ViewProps) => (
    <Dendrogram snapshot={p.snapshot} />
  ));
}
