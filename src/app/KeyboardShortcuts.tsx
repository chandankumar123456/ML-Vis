import { useEffect } from 'react';
import { usePlaybackStore } from '../store/playbackStore';

const PREVENT = new Set(['Space', 'ArrowLeft', 'ArrowRight']);

export function KeyboardShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement
        || t instanceof HTMLSelectElement || (t !== null && t.isContentEditable)) return;
      if (PREVENT.has(e.code)) e.preventDefault();
      switch (e.code) {
        case 'Space': {
          const st = usePlaybackStore.getState();
          st.playing ? st.pause() : st.play();
          break;
        }
        case 'ArrowLeft': usePlaybackStore.getState().stepBackward(); break;
        case 'ArrowRight': usePlaybackStore.getState().stepForward(); break;
        case 'KeyR': usePlaybackStore.getState().reset(); break;
        case 'KeyS':
          document.dispatchEvent(new CustomEvent('mlv:open-palette'));
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  return null;
}
