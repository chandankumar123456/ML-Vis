// Registry wiring coverage for the Wave-4 views (Task 15): both ids resolve
// through registerViews, empty states render through the registry, and the
// distribution-view wrapper resolves class densities from snapshot visuals.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { ComponentType } from 'react';
import { render, screen } from '@testing-library/react';
import { registerAllViews } from './registerViews';
import { getView, viewExists } from '../registry/viewRegistry';
import type { ViewProps } from '../registry/viewRegistry';
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

describe('registerAllViews (wave 4 additions)', () => {
  beforeEach(() => {
    class CtxStub {
      setTransform() {}
      clearRect() {}
      fillRect() {}
      beginPath() {}
      moveTo() {}
      lineTo() {}
      stroke() {}
      fill() {}
      arc() {}
      closePath() {}
      lineWidth = 1;
      fillStyle: string | null = null;
      strokeStyle: string | null = null;
    }
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext')
      .mockReturnValue(new CtxStub() as unknown as CanvasRenderingContext2D);
    class ResizeObserverStub {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    vi.stubGlobal('ResizeObserver', ResizeObserverStub);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('registers the eigenviewer and distribution-view ids', () => {
    registerAllViews();
    expect(viewExists('eigenviewer')).toBe(true);
    expect(viewExists('distribution-view')).toBe(true);
  });

  it('registers the tree-builder, cluster-animator and dendrogram ids', () => {
    registerAllViews();
    expect(viewExists('tree-builder')).toBe(true);
    expect(viewExists('cluster-animator')).toBe(true);
    expect(viewExists('dendrogram')).toBe(true);
  });

  it('renders the eigenviewer empty state through the registry', () => {
    registerAllViews();
    const comp = getView('eigenviewer');
    expect(comp).toBeDefined();
    const View = comp as ComponentType<ViewProps>;
    render(<View params={{}} />);
    expect(screen.getByRole('status')).toHaveTextContent('eigenviewer: no data');
  });

  it('renders the distribution-view empty state through the registry', () => {
    registerAllViews();
    const comp = getView('distribution-view');
    expect(comp).toBeDefined();
    const View = comp as ComponentType<ViewProps>;
    render(<View params={{}} />);
    expect(screen.getByRole('status')).toHaveTextContent('distribution-view: no distributions');
  });

  it('renders the tree-builder empty state through the registry', () => {
    registerAllViews();
    const comp = getView('tree-builder');
    expect(comp).toBeDefined();
    const View = comp as ComponentType<ViewProps>;
    render(<View params={{}} />);
    expect(screen.getByRole('status')).toHaveTextContent('tree-builder: no nodes');
  });

  it('renders the cluster-animator empty state through the registry', () => {
    registerAllViews();
    const comp = getView('cluster-animator');
    expect(comp).toBeDefined();
    const View = comp as ComponentType<ViewProps>;
    render(<View params={{}} />);
    expect(screen.getByRole('status')).toHaveTextContent('cluster-animator: no data');
  });

  it('renders the dendrogram empty state through the registry', () => {
    registerAllViews();
    const comp = getView('dendrogram');
    expect(comp).toBeDefined();
    const View = comp as ComponentType<ViewProps>;
    render(<View params={{}} />);
    expect(screen.getByRole('status')).toHaveTextContent('dendrogram: no merges');
  });

  it('distribution-view resolves class densities from snapshot visuals', () => {
    registerAllViews();
    const comp = getView('distribution-view');
    expect(comp).toBeDefined();
    const View = comp as ComponentType<ViewProps>;
    render(<View params={{}} snapshot={mkSnapshot([
      { type: 'distribution', label: 'class 0', mean: -1, variance: 1.5, color: '#2563eb' },
      { type: 'point', x: 1, y: 1 },
    ])} />);
    expect(screen.getByText('class 0')).toBeInTheDocument();
    expect(screen.queryByRole('status')).toBeNull();
  });
});