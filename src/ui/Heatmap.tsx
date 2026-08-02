import type { CSSProperties } from 'react';

export function Heatmap({ dimensions }: { dimensions: [string, number][] }) {
  return (
    <div className="heatmap">
      {dimensions.map(([label, v]) => (
        <div key={label} className="heatmap-row">
          <span>{label}</span>
          <div className="heatmap-bar">
            <div style={{ width: `${Math.max(4, v * 20)}%` } as CSSProperties} className="heatmap-fill" />
          </div>
        </div>
      ))}
    </div>
  );
}
