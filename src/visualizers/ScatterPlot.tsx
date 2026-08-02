// src/visualizers/ScatterPlot.tsx
import { useEffect, useReducer, useRef } from 'react';
import { CanvasStage, cssVar, type Bounds } from '../lib/canvas/CanvasStage';
import { useContainerSize } from '../lib/canvas/useContainerSize';
import { eventBus } from '../bus/eventBus';
import type { ViewProps } from '../registry/viewRegistry';
import type { VisualCommand } from '../engine/types';

const PALETTE = { point: '#2563eb', hl: '#f59e0b' };

export function ScatterPlot({ snapshot, params }: ViewProps) {
  const [ref, size] = useContainerSize(600, 400);
  const stageRef = useRef<CanvasStage | null>(null);
  const highlightRef = useRef<string | null>(null);
  const [, bump] = useReducer((x: number) => x + 1, 0);

  useEffect(() => {
    const unsub = eventBus.subscribe((e) => {
      if (e.type === 'highlight') highlightRef.current = e.payload.id;
      if (e.type === 'clear-highlights') highlightRef.current = null;
      bump(); // highlight state changed → redraw
    });
    return unsub;
  }, []);

  // instantiate stage + interaction once per size
  useEffect(() => {
    if (!ref.current) return;
    const stage = new CanvasStage(size.w, size.h);
    stage.canvas.style.display = 'block';
    ref.current.replaceChildren(stage.canvas);
    stageRef.current = stage;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = stage.canvas.getBoundingClientRect();
      stage.zoomAt(e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1.1 : 0.9);
    };
    let dragging = false, lastX = 0, lastY = 0;
    const onDown = (e: PointerEvent) => { dragging = true; lastX = e.clientX; lastY = e.clientY; };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      stage.panBy(e.clientX - lastX, e.clientY - lastY);
      lastX = e.clientX; lastY = e.clientY;
    };
    const onUp = () => { dragging = false; };
    stage.canvas.addEventListener('wheel', onWheel, { passive: false });
    stage.canvas.addEventListener('pointerdown', onDown);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      stage.canvas.removeEventListener('wheel', onWheel);
      stage.canvas.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [size]);

  // draw current snapshot
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !snapshot) return;
    const bounds = boundsOf(snapshot.visuals);
    stage.setBounds(bounds, size.w, size.h);
    stage.clear(size.w, size.h, 'transparent');

    const fg = cssVar('--fg', '#0f172a');
    const { visuals, highlights } = snapshot;
    for (const cmd of visuals) {
      const hl = highlights.find((h) => h.id === cmd.id);
      const isHl = highlightRef.current === cmd.id;
      switch (cmd.type) {
        case 'point': {
          stage.drawCircle(cmd.x as number, cmd.y as number, isHl || hl ? 7 : 4.5,
            isHl ? PALETTE.hl : ((cmd.color as string) ?? PALETTE.point));
          break;
        }
        case 'line': {
          stage.drawPath(cmd.points as [number, number][], (cmd.color as string) ?? fg, isHl ? 4 : 2);
          break;
        }
        case 'arrow': {
          stage.drawArrow(cmd.x1 as number, cmd.y1 as number, cmd.x2 as number, cmd.y2 as number,
            (cmd.color as string) ?? fg);
          break;
        }
      }
    }
  }, [snapshot, size, params, bump]);

  return <div ref={ref} style={{ width: '100%', height: 400 }} />;
}

function boundsOf(cmds: VisualCommand[]): Bounds {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  const touch = (x: number, y: number) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  };
  for (const c of cmds) {
    if (c.type === 'point') touch(c.x as number, c.y as number);
    if (c.type === 'line' && Array.isArray(c.points)) {
      for (const [x, y] of c.points as [number, number][]) touch(x, y);
    }
    if (c.type === 'arrow') { touch(c.x1 as number, c.y1 as number); touch(c.x2 as number, c.y2 as number); }
  }
  if (!Number.isFinite(x0)) return { x: [0, 1], y: [0, 1] };
  const padX = (x1 - x0) * 0.1 + 0.5, padY = (y1 - y0) * 0.1 + 0.5;
  return { x: [x0 - padX, x1 + padX], y: [y0 - padY, y1 + padY] };
}
