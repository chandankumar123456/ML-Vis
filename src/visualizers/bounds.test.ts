// Pure-helper coverage for the decision-boundary renderer (review findings 1 & 5):
// safeClassify never throws — undefined / throwing / non-finite classifiers all
// degrade to class 0 — and boundsOfVisuals includes circle extents (x±r, y±r)
// so distance rings are not clipped by the world-space domain.
import { describe, it, expect } from 'vitest';
import { boundsOfVisuals, safeClassify } from './bounds';
import type { Params } from '../engine/types';

describe('safeClassify', () => {
  it('returns class 0 when the classifier is undefined', () => {
    expect(safeClassify(undefined, 1, -2, {})).toBe(0);
  });

  it('returns class 0 when the classifier throws, without rethrowing', () => {
    const throwing = () => { throw new Error('boom'); };
    expect(() => safeClassify(throwing, 0, 0, {})).not.toThrow();
    expect(safeClassify(throwing, 0, 0, {})).toBe(0);
  });

  it('returns class 0 for non-finite classifier results', () => {
    expect(safeClassify(() => NaN, 1, 1, {})).toBe(0);
    expect(safeClassify(() => Infinity, 1, 1, {})).toBe(0);
  });

  it('returns the classifier result and forwards (x, y, params)', () => {
    const seen: unknown[] = [];
    const working = (x: number, y: number, p: Params) => { seen.push(x, y, p); return x + y; };
    const params: Params = { k: 3 };
    expect(safeClassify(working, 1, 2, params)).toBe(3);
    expect(seen).toEqual([1, 2, params]);
  });
});

describe('boundsOfVisuals', () => {
  it('returns null for an empty command list', () => {
    expect(boundsOfVisuals([])).toBeNull();
  });

  it('includes a circle extent (x±r, y±r) in the domain', () => {
    const b = boundsOfVisuals([
      { type: 'circle', x: 0, y: 0, r: 5 },
      { type: 'circle', x: 10, y: -4, r: 2 },
    ]);
    expect(b).not.toBeNull();
    // raw extents x∈[-5,12], y∈[-6,5]; pad = 0.1 * span + 0.5 per axis
    expect(b!.x[0]).toBeCloseTo(-7.2, 10);
    expect(b!.x[1]).toBeCloseTo(14.2, 10);
    expect(b!.y[0]).toBeCloseTo(-7.6, 10);
    expect(b!.y[1]).toBeCloseTo(6.6, 10);
  });

  it('keeps the domain non-null when only circles are present', () => {
    // before the fix, circles were ignored → this returned null and callers
    // fell back to a tight default domain that clipped the ring edges
    expect(boundsOfVisuals([{ type: 'circle', x: 3, y: 3, r: 1 }])).not.toBeNull();
  });

  it('merges circle extents with point/line/arrow extents', () => {
    const b = boundsOfVisuals([
      { type: 'point', x: 0, y: 0 },
      { type: 'line', points: [[4, 4], [6, 0]] },
      { type: 'circle', x: 10, y: 5, r: 3 },
    ]);
    expect(b).not.toBeNull();
    // point(0,0); line x∈[4,6] y∈[0,4]; circle x∈[7,13] y∈[2,8]
    // extents x∈[0,13] (pad 1.8), y∈[0,8] (pad 1.3)
    expect(b!.x[0]).toBeCloseTo(-1.8, 10);
    expect(b!.x[1]).toBeCloseTo(14.8, 10);
    expect(b!.y[0]).toBeCloseTo(-1.3, 10);
    expect(b!.y[1]).toBeCloseTo(9.3, 10);
  });

  it('ignores circles with a non-finite radius instead of poisoning the domain', () => {
    const b = boundsOfVisuals([
      { type: 'point', x: 1, y: 1 },
      { type: 'circle', x: 5, y: 5, r: Number.NaN },
    ]);
    expect(b).not.toBeNull();
    // only the point contributes: extent [1,1] → pad 0.5 both axes
    expect(b!.x[0]).toBeCloseTo(0.5, 10);
    expect(b!.x[1]).toBeCloseTo(1.5, 10);
    expect(b!.y[0]).toBeCloseTo(0.5, 10);
    expect(b!.y[1]).toBeCloseTo(1.5, 10);
  });
});
