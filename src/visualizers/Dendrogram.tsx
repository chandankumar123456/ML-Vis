// src/visualizers/Dendrogram.tsx
// Hierarchical-clustering dendrogram (registered as 'dendrogram'). Reads
// {type:'merge'} commands from the snapshot: `height` is in distance units
// (non-decreasing across the topic's emission order) and `children` reference
// leaf ids OR ids of previously-emitted merges. Layout: leaves sit at height 0
// along the bottom, evenly spaced; each merge sits at its height with a
// horizontal elbow over its subtree span and vertical drop lines to its
// children (a classic bottom-up dendrogram). A distance axis on the left
// auto-ranges from the max height with nice 1-2-5 ticks. Clicking a node
// selects it, draws a ring, highlights every member of its subtree and shows a
// caption ("merge m1 @ height 1.00"). Rendered as SVG (crisp lines, simple
// tree geometry) scaled via viewBox — no container sizing hook needed.
//
// Defensive: non-finite/negative heights are skipped (with a console warning),
// children referencing unknown ids render as dangling leaf stubs, and cyclic
// child references are resolved via slot assignment so layout can never hang
// (subtree member collection is additionally visited-set guarded).
import { useMemo, useState } from 'react';
import type { SimState } from '../engine/types';

// ---- fixed geometry (viewBox units; the SVG scales to its container) ----
const AXIS_W = 56;       // left distance-axis gutter
const TOP = 8;
const BOTTOM = 28;       // leaf label band
const SLOT = 64;         // px per leaf slot
const H = 320;
const LEAF_R = 4;
const TICK_FRACTIONS = [0, 0.25, 0.5, 0.75, 1];

const EDGE_COLOR = '#475569';
const GRID_COLOR = '#e2e8f0';
const LABEL_COLOR = '#0f172a';
const LEAF_COLOR = '#2563eb';
const MERGE_COLOR = '#64748b';
const HL_COLOR = '#f59e0b';

export interface DendroMergeInput {
  id: string;
  height: number;
  children: string[];
}

export interface DendroMergeLayout {
  id: string;
  height: number;
  x: number;                    // layout x in slot units (may be fractional)
  children: { id: string; kind: 'leaf' | 'merge'; x: number; childHeight: number }[];
}

export interface DendroLayout {
  leaves: { id: string; x: number }[];  // x in slot units (0, 1, 2, ...)
  merges: DendroMergeLayout[];          // emission order (bottom-up)
  maxHeight: number;
  niceMax: number;                      // 1-2-5 rounded axis ceiling
}

/**
 * Pure layout: resolves leaves + merges into slot-space geometry. Non-finite /
 * negative heights are skipped (warned). Duplicate merge ids resolve
 * last-wins. Unknown child ids become dangling leaf stubs; a child merge that
 * is not yet laid out (forward reference / cycle) is given a stub slot so the
 * pass is strictly iterative — it cannot hang.
 */
export function layoutDendrogram(merges: DendroMergeInput[]): DendroLayout {
  const valid = merges.filter((m) => {
    const ok = Number.isFinite(m.height) && m.height >= 0;
    if (!ok) console.warn(`dendrogram: skipping merge ${m.id} (non-finite/negative height)`);
    return ok;
  });
  const byId = new Map(valid.map((m) => [m.id, m])); // duplicate ids: last wins

  // leaves = ids referenced as children but never defined as merges; unknown
  // ids (dangling stubs) are collected here too, in first-appearance order.
  const leaves: { id: string; x: number }[] = [];
  const leafX = new Map<string, number>();
  const slotFor = (id: string): number => {
    const existing = leafX.get(id);
    if (existing !== undefined) return existing;
    const slot = leaves.length;
    leaves.push({ id, x: slot });
    leafX.set(id, slot);
    return slot;
  };
  for (const m of valid) {
    for (const c of m.children) if (!byId.has(c)) slotFor(c);
  }

  const nodeX = new Map<string, number>(); // merge id → x (slot units)
  const mergesOut: DendroMergeLayout[] = [];
  for (const m of valid) {
    const children = m.children.map((cid) => {
      const childMerge = byId.get(cid);
      if (childMerge) {
        // known merge: use its laid-out x; a forward reference (cycle) falls
        // back to a dangling stub slot — documented defensive behavior.
        const cx = nodeX.get(cid) ?? slotFor(cid);
        return { id: cid, kind: 'merge' as const, x: cx, childHeight: childMerge.height };
      }
      return { id: cid, kind: 'leaf' as const, x: slotFor(cid), childHeight: 0 };
    });
    const xs = children.map((c) => c.x);
    const x = xs.length > 0 ? (Math.min(...xs) + Math.max(...xs)) / 2 : 0;
    nodeX.set(m.id, x);
    mergesOut.push({ id: m.id, height: m.height, x, children });
  }

  const maxHeight = valid.length > 0 ? Math.max(...valid.map((m) => m.height)) : 0;
  return { leaves, merges: mergesOut, maxHeight, niceMax: niceScaleMax(maxHeight) };
}

/** Collect every descendant id of `id` (itself included), cycle-safe. */
export function subtreeMembers(layout: DendroLayout, id: string): string[] {
  const out: string[] = [];
  const visited = new Set<string>();
  const byId = new Map(layout.merges.map((m) => [m.id, m]));
  const walk = (nodeId: string) => {
    if (visited.has(nodeId)) return;
    visited.add(nodeId);
    out.push(nodeId);
    const m = byId.get(nodeId);
    if (m) for (const c of m.children) walk(c.id);
  };
  walk(id);
  return out;
}

/** Ceiling of v on a 1-2-5 × 10^n scale (1 when v is not a positive number). */
export function niceScaleMax(v: number): number {
  if (!Number.isFinite(v) || v <= 0) return 1;
  const exp = Math.floor(Math.log10(v));
  const f = v / 10 ** exp;
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return nice * 10 ** exp;
}

function formatTick(v: number): string {
  return String(Number(v.toFixed(2)));
}

export function Dendrogram({ snapshot }: { snapshot?: SimState | null }) {
  const merges = useMemo(
    () => (snapshot?.visuals ?? [])
      .filter((v): v is Extract<typeof v, { type: 'merge' }> => v.type === 'merge'),
    [snapshot],
  );
  const layout = useMemo(() => layoutDendrogram(merges), [merges]);
  const [selected, setSelected] = useState<string | null>(null);
  const members = useMemo(
    () => (selected ? new Set(subtreeMembers(layout, selected)) : new Set<string>()),
    [layout, selected],
  );

  if (merges.length === 0) {
    return (
      <div className="dendrogram dendrogram-empty" role="status">
        dendrogram: no merges
      </div>
    );
  }

  const plotBottom = H - BOTTOM;
  const plotHeight = H - TOP - BOTTOM;
  const yFor = (h: number) => plotBottom - (h / layout.niceMax) * plotHeight;
  const xFor = (slotX: number) => AXIS_W + slotX * SLOT + SLOT / 2;
  const W = Math.max(AXIS_W + Math.max(layout.leaves.length, 1) * SLOT + 40, 260);

  const selectedMerge = selected
    ? layout.merges.find((m) => m.id === selected) ?? null
    : null;
  const caption = selectedMerge
    ? `merge ${selectedMerge.id} @ height ${selectedMerge.height.toFixed(2)}`
    : selected
      ? `leaf ${selected}`
      : null;

  return (
    <div className="dendrogram" data-testid="dendrogram" data-merge-count={merges.length}
      style={{ width: '100%', height: 320 }}>
      {caption && (
        <div className="dendrogram-caption" data-testid="dendro-caption" data-selected-id={selected}>
          {caption}
        </div>
      )}
      <svg data-testid="dendro-svg" viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label="hierarchical clustering dendrogram with a distance axis"
        style={{ width: '100%', height: '100%', display: 'block' }}>
        {/* distance axis gridlines at nice tick heights */}
        {TICK_FRACTIONS.map((f) => {
          const h = f * layout.niceMax;
          const y = yFor(h);
          return (
            <g key={f}>
              <line data-testid="dendro-gridline" x1={AXIS_W} x2={W - 12} y1={y} y2={y}
                stroke={GRID_COLOR} strokeWidth={1} />
              <text x={AXIS_W - 6} y={y + 3} textAnchor="end" fontSize={10} fill={LABEL_COLOR}>
                {formatTick(h)}
              </text>
            </g>
          );
        })}

        {/* tree edges: horizontal elbow at each merge + vertical drops to children */}
        {layout.merges.map((m) => {
          const xs = m.children.map((c) => c.x);
          const minX = Math.min(...xs);
          const maxX = Math.max(...xs);
          const yM = yFor(m.height);
          return (
            <g key={`edges-${m.id}`}>
              <line data-testid="dendro-edge" x1={xFor(minX)} x2={xFor(maxX)} y1={yM} y2={yM}
                stroke={EDGE_COLOR} strokeWidth={1.5} />
              {m.children.map((c) => (
                <line key={`${m.id}-${c.id}`} data-testid="dendro-edge"
                  x1={xFor(c.x)} x2={xFor(c.x)} y1={yM} y2={yFor(c.childHeight)}
                  stroke={EDGE_COLOR} strokeWidth={1.5} />
              ))}
            </g>
          );
        })}

        {/* leaves: markers on the baseline with labels below */}
        {layout.leaves.map((leaf) => {
          const isMember = members.has(leaf.id);
          const x = xFor(leaf.x);
          return (
            <g key={`leaf-${leaf.id}`} data-testid="dendro-leaf" data-id={leaf.id}
              data-x={String(x)} data-member={isMember ? 'true' : 'false'}
              onClick={() => setSelected(leaf.id)} style={{ cursor: 'pointer' }}>
              <circle cx={x} cy={plotBottom} r={LEAF_R} fill={isMember ? HL_COLOR : LEAF_COLOR} />
              <text x={x} y={plotBottom + 14} textAnchor="middle" fontSize={10} fill={LABEL_COLOR}>
                {leaf.id}
              </text>
            </g>
          );
        })}

        {/* merge nodes: clickable, ring + accent when selected */}
        {layout.merges.map((m) => {
          const isSelected = selected === m.id;
          const isMember = members.has(m.id);
          const x = xFor(m.x);
          const y = yFor(m.height);
          return (
            <g key={`node-${m.id}`} data-testid="dendro-node" data-id={m.id}
              data-kind="merge" data-height={String(m.height)}
              data-x={String(x)} data-y={y.toFixed(1)}
              data-selected={isSelected ? 'true' : 'false'}
              data-member={isMember ? 'true' : 'false'}
              onClick={() => setSelected((cur) => (cur === m.id ? null : m.id))}
              style={{ cursor: 'pointer' }}>
              {isSelected && (
                <circle cx={x} cy={y} r={9} fill="none" stroke={HL_COLOR} strokeWidth={2} />
              )}
              <circle cx={x} cy={y} r={5} fill={isMember ? HL_COLOR : MERGE_COLOR} />
            </g>
          );
        })}
      </svg>
    </div>
  );
}