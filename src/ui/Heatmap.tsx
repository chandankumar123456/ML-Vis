import type { CSSProperties } from 'react';

export function Heatmap({ dimensions }: { dimensions: [string, number][] }) {
  return (
    <div className="heatmap">
      {dimensions.map(([label, v]) => {
        // floor for visibility, ceiling for sanity, guard NaN
        const width = Number.isFinite(v) ? Math.min(100, Math.max(4, v * 20)) : 4;
        return (
          <div key={label} className="heatmap-row">
            <span>{label}</span>
            <div className="heatmap-bar">
              <div style={{ width: `${width}%` } as CSSProperties} className="heatmap-fill" />
            </div>
          </div>
        );
      })}
    </div>
  );
}
