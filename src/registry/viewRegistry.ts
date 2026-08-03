import type { ComponentType } from 'react';
import type { Params, SnapshotRun, SimState, TopicModule } from '../engine/types';
import type { BusEvent } from '../bus/eventBus';

export interface ViewProps {
  run?: SnapshotRun;
  snapshot?: SimState | null;
  params: Params;
  // Bus-shaped subscription so consumers narrow on the BusEvent union
  // (matches eventBus.subscribe; ViewHost wires it up)
  subscribe?: (fn: (e: BusEvent) => void) => () => void;
  compact?: boolean;
  // The topic module owning this view — lets topic-data visualizers
  // (FormulaExplorer, MistakeView, QuestionPlayer, DerivationPlayer, ...)
  // resolve their content when registered with ViewProps only.
  topic?: TopicModule;
}

const views = new Map<string, ComponentType<ViewProps>>();

export function registerView(id: string, comp: ComponentType<ViewProps>): void {
  views.set(id, comp);
}
export function getView(id: string): ComponentType<ViewProps> | undefined { return views.get(id); }
export function viewExists(id: string): boolean { return views.has(id); }

// ===== Classifier registry =====
// Topics may register a 2D classifier alongside their views; the
// 'decision-boundary' visualizer resolves it via getClassifier(topic.id) to
// paint class regions. The classifier maps a world-space point + the topic's
// current params to a class index (0, 1, 2, ...).
export type Classifier = (x: number, y: number, params: Params) => number;

const classifiers = new Map<string, Classifier>();

export function registerClassifier(id: string, fn: Classifier): void {
  classifiers.set(id, fn);
}
export function getClassifier(id: string): Classifier | undefined { return classifiers.get(id); }
