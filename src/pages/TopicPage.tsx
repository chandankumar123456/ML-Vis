import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getTopic } from '../registry/topicRegistry';
import { usePlaybackStore } from '../store/playbackStore';
import { useProgressStore } from '../store/progressStore';
import { useAnalyticsStore } from '../store/analyticsStore';
import { ViewHost } from './ViewHost';
import { PlaybackBar } from '../ui/PlaybackBar';
import { ParamPanel } from '../ui/ParamPanel';
import { Tabs } from '../ui/Tabs';
import { defaultParams } from '../lib/params';
import type { Params } from '../engine/types';

const LAYER_ORDER = ['foundation', 'core', 'advanced'] as const;
type LoadState = 'loading' | 'ready' | 'error';

export function TopicPage({ loader }: { loader: (id: string) => Promise<boolean> }) {
  const { topicId = '' } = useParams();
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [activeLayer, setActiveLayer] = useState<(typeof LAYER_ORDER)[number]>('foundation');
  const [params, setParams] = useState<Params>({});

  const load = useCallback(() => {
    setLoadState('loading');
    // drop the previous topic's run so the scrubber never shows stale data
    usePlaybackStore.setState({ run: null, playback: null, cursor: 0, playing: false });
    loader(topicId).then((ok) => {
      if (!ok) { setLoadState('error'); return; }
      const t = getTopic(topicId);
      if (!t) { setLoadState('error'); return; }
      setParams(defaultParams(t.params));
      setLoadState('ready');
      useProgressStore.getState().setLastVisited(topicId);
      useAnalyticsStore.getState().recordVisit(topicId);
    });
  }, [topicId, loader]);

  useEffect(() => { load(); }, [load]);

  const topic = loadState === 'ready' ? getTopic(topicId) : undefined;
  const views = useMemo(
    () => (topic ? topic.layers[activeLayer] : []),
    [topic, activeLayer]
  );

  useEffect(() => {
    const t = topic;
    if (!t) return;
    for (const v of views) useProgressStore.getState().markView(t.id, v.component);
  }, [topic, views]);

  if (loadState === 'error') {
    return (
      <div>
        <h1>Topic not found</h1>
        <p>We couldn't load “{topicId}”.</p>
        <button onClick={load}>Retry</button>
      </div>
    );
  }
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
