import { describe, it, expect } from 'vitest';
import { fitBounds } from './CanvasStage';

describe('fitBounds', () => {
  it('scales so data fits within padded viewport', () => {
    const t = fitBounds({ x: [0, 10], y: [0, 10] }, 200, 100, 20);
    // available: w=160, h=60 → scale = min(16, 6) = 6
    expect(t.scale).toBeCloseTo(6, 5);
    // world 0 → tx = (200 - 10*6)/2 = 70 ; world 10 → 70 + 60 = 130 = 200 - 70 ✓ centered
    expect(t.tx).toBeCloseTo(70, 5);
  });

  it('handles zero-range x via the || 1 guard', () => {
    const t = fitBounds({ x: [5, 5], y: [0, 10] }, 200, 100); // default pad 40
    // sx = 120/1 = 120, sy = 20/10 = 2 → scale 2, tx = (200 - 10*2)/2 = 90
    expect(t.scale).toBeCloseTo(2, 5);
    expect(t.tx).toBeCloseTo(90, 5);
  });
});
