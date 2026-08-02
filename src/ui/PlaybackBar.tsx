import { usePlaybackStore } from '../store/playbackStore';

export function PlaybackBar() {
  const cursor = usePlaybackStore((s) => s.cursor);
  const playing = usePlaybackStore((s) => s.playing);
  const speed = usePlaybackStore((s) => s.speed);
  const run = usePlaybackStore((s) => s.run);
  // empty-run sentinel: engine mirrors cursor -1 for failed runs (0 snapshots);
  // a range input with max < min is invalid HTML, so clamp and show '—'
  const max = run ? Math.max(0, run.snapshots.length - 1) : 0;
  const hasRun = run !== null && run.snapshots.length > 0;

  return (
    <div className="playback-bar" role="toolbar" aria-label="Playback">
      <button aria-label="Previous" onClick={() => usePlaybackStore.getState().stepBackward()}>⏮</button>
      <button aria-label="Play/Pause" onClick={() => {
        const st = usePlaybackStore.getState();
        st.playing ? st.pause() : st.play();
      }}>
        {playing ? '⏸' : '▶'}
      </button>
      <button aria-label="Next" onClick={() => usePlaybackStore.getState().stepForward()}>⏭</button>
      <button aria-label="Reset" onClick={() => usePlaybackStore.getState().reset()}>⟲</button>
      <input
        type="range" min={0} max={max} value={Math.max(0, cursor)}
        aria-label="Step scrubber"
        disabled={!hasRun}
        onChange={(e) => usePlaybackStore.getState().setCursor(Number(e.target.value))}
      />
      <span>Step {hasRun ? `${cursor}/${max}` : '—'}</span>
      <select
        aria-label="Speed"
        value={speed}
        onChange={(e) => usePlaybackStore.getState().setSpeed(Number(e.target.value))}
      >
        {[0.25, 0.5, 1, 2, 4].map((v) => <option key={v} value={v}>{v}×</option>)}
      </select>
    </div>
  );
}
