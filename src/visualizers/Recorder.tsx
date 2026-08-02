// src/visualizers/Recorder.tsx
import { useRef, useState } from 'react';
import { usePlaybackStore } from '../store/playbackStore';
import { recordRun } from '../lib/exporters/recorder';

export function Recorder({ renderFrame }: {
  renderFrame: (snapshotIndex: number) => HTMLCanvasElement | null;
}) {
  const run = usePlaybackStore((s) => s.run);
  const [recording, setRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  // retained for Wave-1 GIF/MP4 packaging (no re-render cost)
  const framesRef = useRef<string[]>([]);

  const start = () => {
    if (!run) return;
    setRecording(true);
    // defer so UI updates before heavy work
    setTimeout(() => {
      const frames = recordRun(run, renderFrame);
      framesRef.current = frames;
      setFrameCount(frames.length);
      setRecording(false);
    }, 50);
  };

  return (
    <div className="recorder">
      <button onClick={start} disabled={!run || recording}>
        {recording ? 'Recording…' : 'Record run'}
      </button>
      {frameCount > 0 && <span>{frameCount} frames captured (PNG sequence — packaging in Wave 1)</span>}
    </div>
  );
}
