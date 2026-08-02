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
});
