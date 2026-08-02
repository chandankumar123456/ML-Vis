import { useEffect, useMemo } from 'react';
import { getView } from '../registry/viewRegistry';
import { ensurePlaybackLoop, usePlaybackStore } from '../store/playbackStore';
import { eventBus } from '../bus/eventBus';
import type { TopicModule, Params } from '../engine/types';
import type { BusEvent } from '../bus/eventBus';
import type { ViewProps } from '../registry/viewRegistry';

export function ViewHost({ topic, component, params }: {
  topic: TopicModule; component: string; params: Params;
}) {
  const Comp = getView(component);
  const run = usePlaybackStore((s) => s.run);
  const cursor = usePlaybackStore((s) => s.cursor);

  // join the shared animation loop (idempotent): exactly one tick per frame
  // across ALL mounted hosts — per-host loops would advance speed ×N
  useEffect(() => {
    ensurePlaybackLoop();
  }, []);

  // re-compute snapshots when params change (debounced 150ms)
  useEffect(() => {
    const id = setTimeout(() => {
      usePlaybackStore.getState().computeAndSet(topic.simulation, params);
    }, 150);
    return () => clearTimeout(id);
  }, [topic, params]);

  // dispose may fire once per host (N views per topic); module teardown must be idempotent
  useEffect(() => () => topic.dispose?.(), [topic]);

  const snapshot = useMemo(() => run?.snapshots[cursor] ?? null, [run, cursor]);
  // bus-shaped subscribe matching ViewProps (Task 6 hardened to BusEvent)
  const subscribe = useMemo<ViewProps['subscribe']>(
    () => (fn: (e: BusEvent) => void) => eventBus.subscribe(fn),
    []
  );

  if (!Comp) return <div>Unknown view: {component}</div>;
  return (
    <div className="view-host">
      <Comp run={run ?? undefined} params={params} snapshot={snapshot} subscribe={subscribe} />
    </div>
  );
}
