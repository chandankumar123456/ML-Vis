// src/visualizers/TreeBuilder.tsx
// Decision-tree renderer (registered as 'tree-builder'). Reads {type:'node'}
// commands from the snapshot: x/y are NORMALIZED [0,1] fractions of the
// drawing area — the topic computes the layout, the view scales it to its
// container (so the tree resizes with the panel). Node linkage is explicit:
// each node's `children: string[]` field lists child node ids — topics emit
// no separate edge command. Entropy/gini bars derive from the optional
// `purity` field (0..1, topic-computed); `splitInfo` is free-form split text
// under the label. Snapshot highlights (panel 'canvas', id = node id) draw a
// ring on that node and every node on its ancestor path (found via the
// children map). Defensive: NaN/∞ coordinates are skipped, duplicate ids
// resolve last-wins, and child references to missing nodes are tolerated
// (the edge is skipped, the tree still renders).
import { useEffect, useMemo, useRef } from 'react';
import { useContainerSize } from '../lib/canvas/useContainerSize';
import type { Params, SimState } from '../engine/types';

const PAD = 30;
const NODE_R = 6;
const RING_R = 10;
const BAR_W = 40;
const BAR_H = 4;
const BAR_Y_OFF = 14;   // bar top = node y + this
const EDGE_COLOR = '#94a3b8';
const HL_COLOR = '#f59e0b';
const LABEL_COLOR = '#0f172a';
const BAR_BG = '#e2e8f0';
// Nodes without an explicit color cycle this fixed palette.
const FALLBACK_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#f59e0b', '#0891b2'];

export interface TreeSceneNode {
  id: string;
  x: number;               // normalized [0,1] of the drawing area width
  y: number;               // normalized [0,1] of the drawing area height
  label: string;
  splitInfo?: string;
  color?: string;
  className?: string;
  purity?: number;         // 0..1 — entropy/gini bar fill fraction
  children: string[];      // child node ids (explicit linkage)
  parentId: string | null; // reverse edge for the highlight path walk
}

export interface TreeScene {
  nodes: TreeSceneNode[];
  highlighted: Set<string>; // ids on any highlight path (incl. the node itself)
}

/**
 * Pure scene builder: filters node commands (non-finite coords dropped),
 * resolves duplicate ids last-wins, links children → parents (dangling child
 * references tolerated), and expands canvas highlights up their ancestor
 * chains. Deterministic and DOM-free so it can be unit-tested directly.
 */
export function buildTreeScene(snapshot?: SimState | null): TreeScene {
  if (!snapshot) return { nodes: [], highlighted: new Set() };
  const byId = new Map<string, TreeSceneNode>();
  for (const cmd of snapshot.visuals) {
    if (cmd.type !== 'node') continue;
    const x = cmd.x as number;
    const y = cmd.y as number;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const id = typeof cmd.id === 'string' ? cmd.id : `node-${byId.size}`;
    // duplicate ids: last command wins
    byId.set(id, {
      id,
      x, y,
      label: typeof cmd.label === 'string' ? cmd.label : id,
      splitInfo: typeof cmd.splitInfo === 'string' ? cmd.splitInfo : undefined,
      color: typeof cmd.color === 'string' ? cmd.color : undefined,
      className: typeof cmd.className === 'string' ? cmd.className : undefined,
      purity: typeof cmd.purity === 'number' && Number.isFinite(cmd.purity)
        ? Math.min(1, Math.max(0, cmd.purity))
        : undefined,
      children: Array.isArray(cmd.children)
        ? cmd.children.filter((c): c is string => typeof c === 'string')
        : [],
      parentId: null,
    });
  }
  // child → parent linkage (last children-list wins)
  for (const node of byId.values()) {
    for (const child of node.children) {
      const childNode = byId.get(child);
      // degenerate references to a missing node are tolerated: parentId stays
      // null, no edge is drawn, rendering continues.
      if (childNode) childNode.parentId = node.id;
    }
  }
  // canvas highlights expand to the full ancestor path
  const highlighted = new Set<string>();
  for (const h of snapshot.highlights) {
    if (h.panel !== 'canvas' || !byId.has(h.id)) continue;
    let cur: string | null = h.id;
    while (cur) {
      highlighted.add(cur);
      cur = byId.get(cur)?.parentId ?? null;
    }
  }
  return { nodes: [...byId.values()], highlighted };
}

export function TreeBuilder({ snapshot, params }: {
  snapshot?: SimState | null;
  params: Params;
}) {
  const [ref, size] = useContainerSize(600, 400);
  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<{ host: HTMLDivElement; canvas: HTMLCanvasElement } | null>(null);

  // Scene is a pure memo of the snapshot; highlights and layout are resolved
  // once per snapshot (scrubbing or panel re-renders reuse the cached scene).
  const scene = useMemo(() => buildTreeScene(snapshot), [snapshot]);

  // Paint lazily like DecisionBoundary: create the canvas on first use, then
  // resize + redraw whenever the scene or the observed size changes. params is
  // in the deps by ScatterPlot convention (redraw on param change).
  useEffect(() => {
    const host = hostRef.current;
    if (!host || scene.nodes.length === 0) return;
    const dpr = window.devicePixelRatio || 1;
    let canvas = canvasRef.current?.host === host ? canvasRef.current.canvas : null;
    if (!canvas) {
      const created = document.createElement('canvas');
      created.className = 'tree-canvas';
      created.setAttribute('data-testid', 'tree-canvas');
      created.setAttribute('aria-label',
        'decision tree with node labels, split-purity bars and highlight rings');
      created.style.display = 'block';
      host.replaceChildren(created);
      canvasRef.current = { host, canvas: created };
      canvas = created;
    }
    canvas.width = Math.max(1, size.w * dpr);
    canvas.height = Math.max(1, size.h * dpr);
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${size.h}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    draw(ctx, scene, size, dpr);
  }, [scene, size, params]);

  if (scene.nodes.length === 0) {
    return (
      <div className="tree-builder tree-empty" role="status">
        tree-builder: no nodes
      </div>
    );
  }

  return (
    <div ref={ref} className="tree-builder" data-testid="tree-builder"
      data-node-count={scene.nodes.length}
      data-classes={[...new Set(
        scene.nodes.map((n) => n.className).filter((c): c is string => Boolean(c)),
      )].join(' ')}
      style={{ width: '100%', height: 400 }}>
      <div ref={hostRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}

function draw(ctx: CanvasRenderingContext2D, scene: TreeScene,
  size: { w: number; h: number }, dpr: number): void {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size.w, size.h);
  // normalized [0,1] → pixels inside the padded drawing area
  const w2s = (x: number, y: number): [number, number] => [
    PAD + x * (size.w - 2 * PAD),
    PAD + y * (size.h - 2 * PAD),
  ];
  const byId = new Map(scene.nodes.map((n) => [n.id, n]));

  // edges first (under the nodes): parent → child; stroked in the highlight
  // color when both endpoints sit on a highlighted path
  for (const node of scene.nodes) {
    if (node.children.length === 0) continue;
    const [px, py] = w2s(node.x, node.y);
    for (const cid of node.children) {
      const child = byId.get(cid);
      if (!child) continue; // dangling reference — skip the edge
      const [cx, cy] = w2s(child.x, child.y);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(cx, cy);
      ctx.strokeStyle =
        scene.highlighted.has(node.id) && scene.highlighted.has(cid)
          ? HL_COLOR : EDGE_COLOR;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  scene.nodes.forEach((node, i) => {
    const [px, py] = w2s(node.x, node.y);
    const color = node.color ?? FALLBACK_COLORS[i % FALLBACK_COLORS.length];
    const isHl = scene.highlighted.has(node.id);

    // node circle
    ctx.beginPath();
    ctx.arc(px, py, NODE_R, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // highlight ring on the node itself and its ancestor path
    if (isHl) {
      ctx.beginPath();
      ctx.arc(px, py, RING_R, 0, Math.PI * 2);
      ctx.strokeStyle = HL_COLOR;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // label + split info
    ctx.fillStyle = LABEL_COLOR;
    ctx.font = '11px sans-serif';
    ctx.fillText(node.label, px + 8, py - 8);
    if (node.splitInfo) {
      ctx.font = '10px sans-serif';
      ctx.fillText(node.splitInfo, px + 8, py + 8);
    }

    // purity bar: light track + fill proportional to purity (clamped 0..1)
    if (typeof node.purity === 'number') {
      const bx = px - BAR_W / 2;
      const by = py + BAR_Y_OFF;
      ctx.fillStyle = BAR_BG;
      ctx.fillRect(bx, by, BAR_W, BAR_H);
      ctx.fillStyle = color;
      ctx.fillRect(bx, by, BAR_W * node.purity, BAR_H);
    }
  });
}
