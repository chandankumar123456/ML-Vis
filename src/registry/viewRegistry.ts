import type { ComponentType } from 'react';
import type { Params, SnapshotRun, SimState } from '../engine/types';

export interface ViewProps {
  run?: SnapshotRun;
  snapshot?: SimState | null;
  params: Params;
  subscribe?: (fn: (e: unknown) => void) => () => void;
  compact?: boolean;
}

const views = new Map<string, ComponentType<ViewProps>>();

export function registerView(id: string, comp: ComponentType<ViewProps>): void {
  views.set(id, comp);
}
export function getView(id: string): ComponentType<ViewProps> | undefined { return views.get(id); }
export function viewExists(id: string): boolean { return views.has(id); }
