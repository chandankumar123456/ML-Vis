export function MetricGrid({ metrics }: { metrics: Record<string, number> }) {
  return (
    <div className="metric-grid">
      {Object.entries(metrics).map(([k, v]) => (
        <div key={k} className="metric" data-testid={`metric-${k}`}>
          <span>{k}</span>
          <b>{Number.isFinite(v) ? v.toFixed(4) : '—'}</b>
        </div>
      ))}
    </div>
  );
}
