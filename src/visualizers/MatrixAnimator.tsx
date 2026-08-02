// src/visualizers/MatrixAnimator.tsx
import { useEffect, useState } from 'react';
import { eventBus } from '../bus/eventBus';
import type { VisualCommand } from '../engine/types';

export function MatrixAnimator({ commands }: { commands: VisualCommand[] }) {
  const [hl, setHl] = useState<string | null>(null);

  useEffect(() => {
    const unsub = eventBus.subscribe((e) => {
      if (e.type === 'highlight') setHl(e.payload.id);
      if (e.type === 'clear-highlights') setHl(null);
    });
    return unsub;
  }, []);

  const matrices = commands.filter((c) => c.type === 'matrix') as (VisualCommand & {
    rows: number; cols: number; cells: (number | string)[][];
  })[];

  return (
    <div className="matrix-animator">
      {matrices.map((m, i) => (
        <div key={i} className="matrix-wrap">
          <table className="matrix">
            <tbody>
              {m.cells.map((row, r) => (
                <tr key={r}>
                  {row.map((cell, c) => {
                    const cellId = `${m.id}:${r},${c}`;
                    const active = hl === cellId;
                    return (
                      <td key={c} data-testid={cellId}
                        className={active ? 'cell active' : 'cell'}
                        onMouseEnter={() => eventBus.emit({ type: 'highlight', payload: { panel: 'matrix', id: cellId, intensity: 1 } })}
                        onMouseLeave={() => eventBus.emit({ type: 'clear-highlights' })}>
                        {typeof cell === 'number' ? cell.toFixed(2) : cell}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          {m.id && <small>{m.id}</small>}
        </div>
      ))}
    </div>
  );
}
