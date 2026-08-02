import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PlaybackBar } from './PlaybackBar';
import { usePlaybackStore } from '../store/playbackStore';

describe('PlaybackBar', () => {
  it('renders controls and steps forward', () => {
    usePlaybackStore.getState().computeAndSet({
      initialState: () => ({ algorithm: {}, visuals: [], math: [], narration: '', explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] }, highlights: [], metrics: {}, events: [], timeline: [] }),
      step: () => null,
    } as any, {});
    render(<PlaybackBar />);
    const next = screen.getByRole('button', { name: /next/i });
    fireEvent.click(next);
    // cursor should not exceed 0 (single snapshot run)
    expect(usePlaybackStore.getState().cursor).toBe(0);
  });

  it('degrades gracefully on an empty (failed) run — sentinel -1', () => {
    // engine failure with 0 snapshots → store mirrors cursor -1
    usePlaybackStore.getState().computeAndSet({
      initialState: () => { throw new Error('boom'); },
      step: () => null,
    } as any, {});
    render(<PlaybackBar />);
    const scrubber = screen.getByRole('slider', { name: 'Step scrubber' }) as HTMLInputElement;
    // max clamped to >= 0 → valid range input
    expect(Number(scrubber.max)).toBe(0);
    expect(scrubber.disabled).toBe(true);
    // step label shows em dash, not -1/-1
    expect(screen.getByText(/Step —/)).toBeTruthy();
    // play must not claim playing on an empty run
    fireEvent.click(screen.getByRole('button', { name: /play\/pause/i }));
    expect(usePlaybackStore.getState().playing).toBe(false);
  });
});
