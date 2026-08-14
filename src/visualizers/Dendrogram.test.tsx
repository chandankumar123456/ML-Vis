// Component coverage for the dendrogram registry view (Task 19): empty state,
// correct edge counts for a 4-leaf binary tree (3 merges → 9 tree edges + 5
// axis gridlines), heights mapped to y positions, evenly spaced leaves,
// click-to-select with subtree highlighting and the caption, non-finite height
// skipping, dangling (unknown-child) stubs, and the cyclic-reference guard.
// The pure layout helpers (layoutDendrogram / subtreeMembers / niceScaleMax)
// are unit-tested directly.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Dendrogram, layoutDendrogram, subtreeMembers, niceScaleMax } from './Dendrogram';
import type { SimState, VisualCommand } from '../engine/types';

const mkSnapshot = (visuals: VisualCommand[]): SimState => ({
  algorithm: {},
  visuals,
  math: [],
  narration: '',
  explanation: { changed: [], why: '', dependsOn: [], gateConcepts: [] },
  highlights: [],
  metrics: {},
  events: [],
  timeline: [],
});

// 4-leaf binary tree: m1 merges l1+l2 at 1, m2 merges l3+l4 at 2, m3 merges
// the two clusters at 4 (chronological emission order)
const FOUR_LEAF_TREE = mkSnapshot([
  { type: 'merge', id: 'm1', height: 1, children: ['l1', 'l2'] },
  { type: 'merge', id: 'm2', height: 2, children: ['l3', 'l4'] },
  { type: 'merge', id: 'm3', height: 4, children: ['m1', 'm2'] },
]);

describe('Dendrogram (component)', () => {
  beforeEach(() => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders an empty state when no snapshot is provided', () => {
    render(<Dendrogram />);
    expect(screen.getByRole('status')).toHaveTextContent('dendrogram: no merges');
  });

  it('renders an empty state when the snapshot has no merge commands', () => {
    render(<Dendrogram snapshot={mkSnapshot([])} />);
    expect(screen.getByRole('status')).toHaveTextContent('dendrogram: no merges');
  });

  it('draws the correct edge count for a 4-leaf tree (3 merges) plus axis gridlines', () => {
    const { container } = render(<Dendrogram snapshot={FOUR_LEAF_TREE} />);
    // 3 merges × (1 horizontal + 2 vertical drop lines) = 9 tree edges
    expect(container.querySelectorAll('[data-testid="dendro-edge"]').length).toBe(9);
    // distance axis: 5 nice ticks (0, ¼, ½, ¾, max)
    expect(container.querySelectorAll('[data-testid="dendro-gridline"]').length).toBe(5);
  });

  it('maps merge heights to y positions (higher height → smaller y) and exposes them', () => {
    const { container } = render(<Dendrogram snapshot={FOUR_LEAF_TREE} />);
    const node = (id: string) => container.querySelector(
      `[data-testid="dendro-node"][data-id="${id}"]`,
    );
    expect(node('m1')!.getAttribute('data-height')).toBe('1');
    expect(node('m2')!.getAttribute('data-height')).toBe('2');
    expect(node('m3')!.getAttribute('data-height')).toBe('4');
    const y1 = parseFloat(node('m1')!.getAttribute('data-y')!);
    const y2 = parseFloat(node('m2')!.getAttribute('data-y')!);
    const y3 = parseFloat(node('m3')!.getAttribute('data-y')!);
    expect(y1).toBeGreaterThan(y2);
    expect(y2).toBeGreaterThan(y3);
    // tallest merge sits above the leaf baseline
    expect(y3).toBeLessThan(292);
  });

  it('spaces the leaves evenly along the baseline', () => {
    const { container } = render(<Dendrogram snapshot={FOUR_LEAF_TREE} />);
    const xs = Array.from(container.querySelectorAll('[data-testid="dendro-leaf"]'))
      .map((el) => parseFloat(el.getAttribute('data-x')!));
    expect(xs).toHaveLength(4);
    const gaps = xs.slice(1).map((x, i) => x - xs[i]);
    // evenly spaced: every gap is the same slot width
    expect(new Set(gaps).size).toBe(1);
    expect(gaps[0]).toBeGreaterThan(0);
  });

  it('clicking a merge selects it, highlights its subtree members and shows a caption', () => {
    const { container } = render(<Dendrogram snapshot={FOUR_LEAF_TREE} />);
    const node = (id: string) => container.querySelector(
      `[data-testid="dendro-node"][data-id="${id}"]`,
    );
    const leaf = (id: string) => container.querySelector(
      `[data-testid="dendro-leaf"][data-id="${id}"]`,
    );
    expect(container.querySelector('[data-testid="dendro-caption"]')).toBeNull();

    // select the root merge m3: the whole tree is its subtree
    fireEvent.click(node('m3')!);
    expect(node('m3')!.getAttribute('data-selected')).toBe('true');
    expect(container.querySelector('[data-testid="dendro-caption"]')!.textContent)
      .toContain('merge m3 @ height 4.00');
    ['l1', 'l2', 'l3', 'l4'].forEach((id) => {
      expect(leaf(id)!.getAttribute('data-member')).toBe('true');
    });

    // select the left cluster m1: only l1/l2 remain highlighted
    fireEvent.click(node('m1')!);
    expect(node('m1')!.getAttribute('data-selected')).toBe('true');
    expect(node('m3')!.getAttribute('data-selected')).toBe('false');
    expect(container.querySelector('[data-testid="dendro-caption"]')!.textContent)
      .toContain('merge m1 @ height 1.00');
    expect(leaf('l1')!.getAttribute('data-member')).toBe('true');
    expect(leaf('l2')!.getAttribute('data-member')).toBe('true');
    expect(leaf('l3')!.getAttribute('data-member')).toBe('false');
    expect(leaf('l4')!.getAttribute('data-member')).toBe('false');
  });

  it('skips merges with non-finite or negative heights and warns', () => {
    const snap = mkSnapshot([
      { type: 'merge', id: 'ok', height: 1, children: ['l1', 'l2'] },
      { type: 'merge', id: 'nan', height: NaN, children: ['l3', 'l4'] },
      { type: 'merge', id: 'neg', height: -2, children: ['l5', 'l6'] },
    ]);
    const { container } = render(<Dendrogram snapshot={snap} />);
    // only the valid merge renders: 1 horizontal + 2 verticals = 3 edges
    expect(container.querySelectorAll('[data-testid="dendro-edge"]').length).toBe(3);
    expect(container.querySelector('[data-testid="dendro-node"][data-id="nan"]')).toBeNull();
    expect(container.querySelector('[data-testid="dendro-node"][data-id="neg"]')).toBeNull();
    expect(console.warn).toHaveBeenCalledTimes(2);
  });

  it('renders unknown child ids as dangling leaf stubs', () => {
    const snap = mkSnapshot([
      { type: 'merge', id: 'm1', height: 1, children: ['l1', 'ghost'] },
    ]);
    const { container } = render(<Dendrogram snapshot={snap} />);
    // 'ghost' is not a merge: it becomes a leaf marker at the baseline
    const ghost = container.querySelector('[data-testid="dendro-leaf"][data-id="ghost"]');
    expect(ghost).not.toBeNull();
    // its drop line still exists (1 horizontal + 2 verticals)
    expect(container.querySelectorAll('[data-testid="dendro-edge"]').length).toBe(3);
  });

  it('guards against cyclic child references without hanging', () => {
    const snap = mkSnapshot([
      { type: 'merge', id: 'A', height: 2, children: ['B', 'l1'] },
      { type: 'merge', id: 'B', height: 1, children: ['A', 'l2'] },
    ]);
    const { container } = render(<Dendrogram snapshot={snap} />);
    // both merges render; the cycle is resolved with dangling stubs, no hang
    expect(container.querySelectorAll('[data-testid="dendro-node"]').length).toBe(2);
    expect(screen.queryByRole('status')).toBeNull();
  });
});

describe('Dendrogram (pure layout helpers)', () => {
  it('lays out a 4-leaf binary tree with midpoints and evenly spaced leaves', () => {
    const layout = layoutDendrogram([
      { id: 'm1', height: 1, children: ['l1', 'l2'] },
      { id: 'm2', height: 2, children: ['l3', 'l4'] },
      { id: 'm3', height: 4, children: ['m1', 'm2'] },
    ]);
    expect(layout.leaves.map((l) => l.id)).toEqual(['l1', 'l2', 'l3', 'l4']);
    // leaf slots are 0,1,2,3 — evenly spaced
    expect(layout.leaves.map((l) => l.x)).toEqual([0, 1, 2, 3]);
    // merges sit at the midpoint of their children
    expect(layout.merges.find((m) => m.id === 'm1')!.x).toBe(0.5);
    expect(layout.merges.find((m) => m.id === 'm2')!.x).toBe(2.5);
    expect(layout.merges.find((m) => m.id === 'm3')!.x).toBe(1.5);
    expect(layout.niceMax).toBe(5); // nice 1-2-5 rounding of max height 4
    // child linkage keeps kind + height for edge drawing
    const m3 = layout.merges.find((m) => m.id === 'm3')!;
    expect(m3.children.map((c) => c.kind)).toEqual(['merge', 'merge']);
    expect(m3.children.map((c) => c.childHeight)).toEqual([1, 2]);
  });

  it('treats unknown children as dangling leaves and guards cycles', () => {
    const layout = layoutDendrogram([
      { id: 'A', height: 2, children: ['B', 'l1'] },
      { id: 'B', height: 1, children: ['A', 'l2'] },
    ]);
    expect(layout.merges).toHaveLength(2);
    expect(layout.niceMax).toBe(2);
    // both cycles resolved (a stub slot is assigned), nothing hangs
    expect(layout.merges[0].children.length).toBe(2);
    expect(layout.merges[1].children.length).toBe(2);
  });

  it('subtreeMembers collects the full descendant set without revisiting', () => {
    const layout = layoutDendrogram([
      { id: 'm1', height: 1, children: ['l1', 'l2'] },
      { id: 'm2', height: 2, children: ['l3', 'l4'] },
      { id: 'm3', height: 4, children: ['m1', 'm2'] },
    ]);
    expect(subtreeMembers(layout, 'm3').sort()).toEqual(['l1', 'l2', 'l3', 'l4', 'm1', 'm2', 'm3']);
    expect(subtreeMembers(layout, 'm1').sort()).toEqual(['l1', 'l2', 'm1']);
    expect(subtreeMembers(layout, 'l3')).toEqual(['l3']);
  });

  it('niceScaleMax rounds up to a 1-2-5 scale', () => {
    expect(niceScaleMax(4)).toBe(5);
    expect(niceScaleMax(10)).toBe(10);
    expect(niceScaleMax(0.03)).toBe(0.05);
    expect(niceScaleMax(0)).toBe(1);
    expect(niceScaleMax(NaN)).toBe(1);
  });
});