// src/visualizers/knowledgeGraph/KnowledgeGraph.tsx
import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { graphNodes, graphEdges, edgeTypeColor, nodeColor, type EdgeType } from './graphData';
import { useNavigate } from 'react-router-dom';

export const EDGE_TYPES: EdgeType[] = ['requires', 'related', 'extends', 'derives-from', 'contrasts-with', 'frequently-confused', 'hidden-gate-link'];

export function KnowledgeGraph() {
  const ref = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const [size, setSize] = useState({ w: 900, h: 600 });

  useEffect(() => {
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0) setSize({ w: width, h: height });
    });
    if (ref.current) ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    const holder = ref.current;
    holder.innerHTML = '';
    const svg = d3.select(holder).append('svg')
      .attr('width', size.w).attr('height', size.h)
      .attr('role', 'img').attr('aria-label', 'Knowledge graph of ML topics');
    const g = svg.append('g');

    svg.call(d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (e) => g.attr('transform', e.transform)));

    const nodes = graphNodes.map((d) => ({ ...d }));
    const links = graphEdges.map((e) => ({ ...e }));

    const simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links as any).id((d: any) => d.id).distance(90))
      .force('charge', d3.forceManyBody().strength(-280))
      .force('center', d3.forceCenter(size.w / 2, size.h / 2))
      .force('collide', d3.forceCollide(18));

    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (d) => edgeTypeColor(d.type))
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', (d) => (d.type === 'hidden-gate-link' ? '4 4' : null));

    link.append('title').text((d) => `${(d.source as any).label ?? d.source} → ${(d.target as any).label ?? d.target}: ${d.note}`);

    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .style('cursor', 'pointer')
      .on('click', (_e, d) => {
        if (d.kind === 'topic') navigate(`/topic/${d.id}`);
      })
      .call((d3.drag<SVGGElement, any, any>() as any)
        .on('start', (e: any, d: any) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (e: any, d: any) => { d.fx = e.x; d.fy = e.y; })
        .on('end', (e: any, d: any) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

    node.append('circle')
      .attr('r', (d) => 6 + d.weight * 2.5)
      .attr('fill', (d) => nodeColor(d.category))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5);

    node.append('text')
      .text((d) => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', -14)
      .attr('font-size', 11);

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x).attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x).attr('y2', (d: any) => d.target.y);
      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => { simulation.stop(); };
  }, [size, navigate]);

  return (
    <div className="kg-wrap">
      <div ref={ref} style={{ width: '100%', height: 600 }} />
      <div className="kg-legend">
        {EDGE_TYPES.map((t) => (
          <span key={t}><i style={{ background: edgeTypeColor(t) }} /> {t}</span>
        ))}
      </div>
    </div>
  );
}
