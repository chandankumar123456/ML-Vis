// Pure-helper coverage for the eigenviewer registry view (Task 15): centroid,
// axis unit vector, orthogonal projection, residual length, projected
// coordinate, variance along an axis, explained-variance fractions for the pair
// of orthogonal 2-D axes (the "per-PC" bar split), axis extent, and the
// world-space domain fitted over the point cloud.
import { describe, it, expect } from 'vitest';
import {
  centroid, axisUnit, axisExtent, projectPoint, residualLength,
  axisCoordinate, varianceAlong, varianceFractions, pointsBounds,
  resolveAngleDeg, sceneProjections,
} from './eigenview';
import type { VisualCommand } from '../engine/types';

describe('centroid', () => {
  it('returns [0, 0] for an empty point list', () => {
    expect(centroid([])).toEqual([0, 0]);
  });

  it('is the arithmetic mean of the points', () => {
    expect(centroid([[0, 0], [4, 0]])).toEqual([2, 0]);
    expect(centroid([[1, 1], [3, 5]])).toEqual([2, 3]);
  });
});

describe('axisUnit', () => {
  it('is [1, 0] at angle 0', () => {
    const [ux, uy] = axisUnit(0);
    expect(ux).toBeCloseTo(1, 10);
    expect(uy).toBeCloseTo(0, 10);
  });

  it('is [0, 1] at a right angle', () => {
    const [ux, uy] = axisUnit(Math.PI / 2);
    expect(ux).toBeCloseTo(0, 10);
    expect(uy).toBeCloseTo(1, 10);
  });

  it('is [-1, 0] at pi', () => {
    const [ux, uy] = axisUnit(Math.PI);
    expect(ux).toBeCloseTo(-1, 10);
    expect(uy).toBeCloseTo(0, 10);
  });

  it('is a unit vector at an arbitrary angle', () => {
    const [ux, uy] = axisUnit(0.7);
    expect(Math.hypot(ux, uy)).toBeCloseTo(1, 10);
  });
});

describe('projectPoint', () => {
  it('projects a point onto a horizontal axis through the center', () => {
    // axis = x-axis through c=(2,0); (0,0) is 2 left of center → stays at (0,0)
    expect(projectPoint([0, 0], [2, 0], [1, 0])).toEqual([0, 0]);
    // (4,0) is on the axis → maps to itself
    expect(projectPoint([4, 0], [2, 0], [1, 0])).toEqual([4, 0]);
  });

  it('projects a point onto a vertical axis through the origin', () => {
    expect(projectPoint([2, 3], [0, 0], [0, 1])).toEqual([0, 3]);
  });

  it('keeps a point on the axis center fixed', () => {
    expect(projectPoint([1, 1], [1, 1], [1, 0])).toEqual([1, 1]);
  });
});

describe('residualLength', () => {
  it('is the perpendicular distance to the axis line', () => {
    expect(residualLength([0, 2], [0, 0], [1, 0])).toBeCloseTo(2, 10);
  });

  it('is 0 when the point lies on the axis line', () => {
    expect(residualLength([0, 0], [2, 0], [1, 0])).toBeCloseTo(0, 10);
  });
});

describe('axisCoordinate', () => {
  it('is the signed distance along the axis direction', () => {
    expect(axisCoordinate([5, 7], [0, 0], [1, 0])).toBeCloseTo(5, 10);
  });

  it('is the dot (p - center, u) for a diagonal axis', () => {
    const u: [number, number] = [Math.SQRT1_2, Math.SQRT1_2];
    expect(axisCoordinate([1, 1], [0, 0], u)).toBeCloseTo(Math.SQRT2, 6);
  });
});

describe('varianceAlong', () => {
  // data [(0,0),(4,0)] centered at (2,0): the x-direction carries all variance
  const pts: [number, number][] = [[0, 0], [4, 0]];
  const c: [number, number] = [2, 0];

  it('is the full variance along the data axis', () => {
    expect(varianceAlong(pts, c, [1, 0])).toBeCloseTo(4, 10);
  });

  it('is 0 orthogonal to the data axis', () => {
    expect(varianceAlong(pts, c, [0, 1])).toBeCloseTo(0, 10);
  });

  it('splits variance along a 45° diagonal', () => {
    const u: [number, number] = [Math.SQRT1_2, Math.SQRT1_2];
    // projections t = [−√2, √2] → var = 2
    expect(varianceAlong(pts, c, u)).toBeCloseTo(2, 10);
  });

  it('is 0 without any points', () => {
    expect(varianceAlong([], [0, 0], [1, 0])).toBeCloseTo(0, 10);
  });
});

describe('varianceFractions', () => {
  // data [(0,0),(4,0)] centered at (2,0) — all variance on the x-axis
  const pts: [number, number][] = [[0, 0], [4, 0]];
  const c: [number, number] = [2, 0];

  it('reports 1.0 / 0.0 for axes aligned with / orthogonal to the data', () => {
    const bars = varianceFractions(pts, c, 0);
    expect(bars[0].angle).toBeCloseTo(0, 10);
    expect(bars[0].fraction).toBeCloseTo(1, 9);
    expect(bars[1].angle).toBeCloseTo(Math.PI / 2, 10);
    expect(bars[1].fraction).toBeCloseTo(0, 9);
  });

  it('splits the variance evenly at 45°', () => {
    const bars = varianceFractions(pts, c, Math.PI / 4);
    expect(bars[0].fraction).toBeCloseTo(0.5, 9);
    expect(bars[1].fraction).toBeCloseTo(0.5, 9);
  });

  it('always accounts for 100% of the 2-D variance', () => {
    for (const deg of [0, 17, 45, 83, 123]) {
      const bars = varianceFractions(pts, c, (deg * Math.PI) / 180);
      expect(bars[0].fraction + bars[1].fraction).toBeCloseTo(1, 9);
    }
  });

  it('degenerates to an even split for a single point (no variance)', () => {
    const bars = varianceFractions([[2, 2]], [2, 2], 0);
    expect(bars[0].fraction).toBeCloseTo(0.5, 9);
    expect(bars[1].fraction).toBeCloseTo(0.5, 9);
  });
});

describe('pointsBounds', () => {
  it('pads the point extents by 10% + 0.5 per axis (ScatterPlot convention)', () => {
    const b = pointsBounds([[0, 0], [4, 0]]);
    expect(b).not.toBeNull();
    // x span 4 → pad 0.9; y span 0 → pad 0.5 (per-axis padding)
    expect(b!.x[0]).toBeCloseTo(-0.9, 10);
    expect(b!.x[1]).toBeCloseTo(4.9, 10);
    expect(b!.y[0]).toBeCloseTo(-0.5, 10);
    expect(b!.y[1]).toBeCloseTo(0.5, 10);
  });

  it('keeps a tight window around a single point', () => {
    const b = pointsBounds([[2, 2]]);
    expect(b!.x[0]).toBeCloseTo(1.5, 10);
    expect(b!.x[1]).toBeCloseTo(2.5, 10);
    expect(b!.y[0]).toBeCloseTo(1.5, 10);
    expect(b!.y[1]).toBeCloseTo(2.5, 10);
  });

  it('returns null for an empty point list (callers fall back to a default)', () => {
    expect(pointsBounds([])).toBeNull();
  });

  it('skips non-finite points when fitting bounds (ScatterPlot convention)', () => {
    const b = pointsBounds([[0, 0], [Number.NaN, 0], [4, 0]]);
    expect(b).not.toBeNull();
    // the NaN-coordinate point is ignored → bounds come from (0,0) and (4,0)
    expect(b!.x[0]).toBeCloseTo(-0.9, 10);
    expect(b!.x[1]).toBeCloseTo(4.9, 10);
    expect(b!.y[0]).toBeCloseTo(-0.5, 10);
    expect(b!.y[1]).toBeCloseTo(0.5, 10);
  });

  it('returns null when every point is non-finite (callers fall back to a default)', () => {
    expect(pointsBounds([[Number.NaN, 0], [0, Number.NEGATIVE_INFINITY]])).toBeNull();
  });
});

describe('axisExtent', () => {
  it('spans the point cloud around the center with a 10% margin', () => {
    expect(axisExtent([[0, 0], [4, 0]], [2, 0])).toBeCloseTo(2.2, 10);
  });

  it('is 0 without any points', () => {
    expect(axisExtent([], [0, 0])).toBeCloseTo(0, 10);
  });
});

describe('resolveAngleDeg', () => {
  const axisCmd = (angle: number): VisualCommand => ({ type: 'axis', angle });

  it('prefers a user slider override over everything else', () => {
    const visuals = [axisCmd(Math.PI / 4)]; // 45°
    expect(resolveAngleDeg(visuals, { angleDeg: 30 }, 90)).toBe(90);
  });

  it('falls back to the snapshot axis command (radians → degrees)', () => {
    expect(resolveAngleDeg([axisCmd(Math.PI / 4)], {}, null)).toBe(45);
    expect(resolveAngleDeg([axisCmd(Math.PI / 2)], {}, null)).toBe(90);
  });

  it('falls back to the params angleDeg hint without an axis command', () => {
    expect(resolveAngleDeg([], { angleDeg: 33 }, null)).toBe(33);
  });

  it('ignores a non-finite axis command angle', () => {
    expect(resolveAngleDeg([{ type: 'axis', angle: Number.NaN }], {}, null)).toBe(0);
    expect(resolveAngleDeg([axisCmd(Number.POSITIVE_INFINITY)], { angleDeg: 12 }, null)).toBe(12);
  });

  it('normalizes a negative axis angle (radians) into [0, 180)', () => {
    // -0.5 rad ≈ -28.65° — a line axis is θ ≡ θ + 180°, so -28.65° ≡ 151.35°
    expect(resolveAngleDeg([axisCmd(-0.5)], {}, null)).toBe(151);
    // -π/2 rad = -90° ≡ 90°
    expect(resolveAngleDeg([axisCmd(-Math.PI / 2)], {}, null)).toBe(90);
  });

  it('defaults to 0 when nothing provides an angle', () => {
    expect(resolveAngleDeg([], {}, null)).toBe(0);
  });
});

describe('sceneProjections', () => {
  // points [(0,0),(4,0)] centered at (2,0) — axis angle 0 is the x-axis
  const pts: [number, number][] = [[0, 0], [4, 0]];
  const c: [number, number] = [2, 0];

  it('recomputes projections from the points when no commands exist', () => {
    const pr = sceneProjections(pts, c, [], 0, null);
    expect(pr).toHaveLength(2);
    // (0,0) projects to (0,0); (4,0) projects to (4,0); residual 0 (on-axis)
    expect(pr[0].to).toEqual([0, 0]);
    expect(pr[1].to).toEqual([4, 0]);
    expect(pr[0].residual).toBeCloseTo(0, 10);
  });

  it('recomputes projections when the user overrides the angle, ignoring commands', () => {
    const pr = sceneProjections(pts, c, [
      { type: 'projection', point: [0, 0], onto: [9, 9], residual: 5 },
    ], 0, 90);
    // override → own math wins: onto = (0,0),(4,0), NOT the command's (9,9)
    expect(pr[0].to).toEqual([0, 0]);
    expect(pr[1].to).toEqual([4, 0]);
  });

  it('uses the snapshot projection commands (with residual) when there is no override', () => {
    const pr = sceneProjections(pts, c, [
      { type: 'projection', point: [0, 0], onto: [1, 1], residual: 2.5, color: '#ff0000' },
      { type: 'projection', point: [4, 0], onto: [3, 1], residual: 1.25 },
    ], 0, null);
    expect(pr[0].to).toEqual([1, 1]);
    expect(pr[0].residual).toBe(2.5);
    expect(pr[0].color).toBe('#ff0000');
    expect(pr[1].to).toEqual([3, 1]);
    expect(pr[1].residual).toBe(1.25);
  });

  it('truncates snapshot projections to the point count (colors stay indexed)', () => {
    // 2 points but 3 projection commands: the view colors by the POINTS array,
    // so the scene must emit at most points.length projections — anything more
    // would index an undefined color when Eigenviewer paints scene.colors[i].
    const pr = sceneProjections(pts, c, [
      { type: 'projection', point: [0, 0], onto: [1, 1], residual: 2.5, color: '#ff0000' },
      { type: 'projection', point: [4, 0], onto: [3, 1], residual: 1.25 },
      { type: 'projection', point: [9, 9], onto: [8, 8], residual: 0.5 },
    ], 0, null);
    expect(pr).toHaveLength(2);
    expect(pr[0].to).toEqual([1, 1]);
    expect(pr[1].to).toEqual([3, 1]);
    expect(pr[0].color).toBe('#ff0000');
  });

  it('computes the residual as the perpendicular distance to the axis', () => {
    const pr = sceneProjections([[1, 2], [3, -1]], [0, 0], [], 0, null);
    // horizontal axis: residual = |y|
    expect(pr[0].residual).toBeCloseTo(2, 10);
    expect(pr[1].residual).toBeCloseTo(1, 10);
  });

  it('returns an empty list without points', () => {
    expect(sceneProjections([], [0, 0], [], 0, null)).toEqual([]);
  });
});