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

export function TopicPage({ loader }: { loader: (id: string) => Promise<unknown> }) {
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
