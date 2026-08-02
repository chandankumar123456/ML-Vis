import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getTopic } from '../registry/topicRegistry';
import { usePlaybackStore } from '../store/playbackStore';
import { useSessionStore } from '../store/sessionStore';
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

  const sessions = useSessionStore((s) => s.sessions);
  const mine = sessions.filter((x) => x.topicId === topicId);

  const saveSession = () => {
    if (!topic) return;
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
    // validate activeView against LAYER_ORDER before casting (corrupted/stale bundles)
    const layer = LAYER_ORDER.includes(b.activeView as (typeof LAYER_ORDER)[number])
      ? (b.activeView as (typeof LAYER_ORDER)[number])
      : 'foundation';
    setActiveLayer(layer);
    // if the current run already holds this exact params object, ViewHost's
    // computeAndSet will dedupe and never swap the run — the one-shot below
    // would never fire; restore the cursor directly (no overwrite risk).
    if (usePlaybackStore.getState().run?.params === b.params) {
      usePlaybackStore.getState().setCursor(b.step);
      return;
    }
    // one-shot: ViewHost's debounced computeAndSet replaces the playback ~150ms
    // after setParams, resetting cursor to 0 — restore the saved step AFTER the
    // new run lands (subscribe fires when the run reference changes)
    const unsub = usePlaybackStore.subscribe((s, prev) => {
      if (prev.run !== s.run) {
        unsub();
        usePlaybackStore.getState().setCursor(b.step);
      }
    });
  };

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
      <div className="session-row">
        <button onClick={saveSession} disabled={loadState !== 'ready'}>Save session</button>
        {mine.map((s) => (
          <button key={s.savedAt} onClick={() => resume(s.savedAt)}>
            Resume {new Date(s.savedAt).toLocaleTimeString()}
          </button>
        ))}
      </div>
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
