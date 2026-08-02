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
