// src/visualizers/FormulaExplorer.tsx
import { useState } from 'react';
import { Latex } from '../ui/Latex';
import type { Formula } from '../engine/types';

export function FormulaExplorer({ formulas }: { formulas: Formula[] }) {
  const [selectedId, setSelectedId] = useState(formulas[0]?.id);

  const f = formulas.find((x) => x.id === selectedId);
  if (!f) return null;

  return (
    <div className="formula-explorer">
      <div className="formula-list">
        {formulas.map((x) => (
          <button key={x.id} className={x.id === selectedId ? 'pill active' : 'pill'}
            onClick={() => setSelectedId(x.id)}>
            {x.id}
          </button>
        ))}
      </div>
      <div className="formula-detail">
        <Latex tex={f.latex} block />
        <h4>Symbols</h4>
        <table>
          <thead><tr><th>Symbol</th><th>Meaning</th><th>Dimensions</th></tr></thead>
          <tbody>
            {f.symbols.map((s) => (
              <tr key={s.symbol}>
                <td><Latex tex={s.symbol} /></td>
                <td>{s.meaning}</td>
                <td>{s.dimensions ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h4>Assumptions</h4>
        <ul>{f.assumptions.map((a, i) => <li key={i}>{a}</li>)}</ul>
        <h4>Derives from</h4>
        <div className="pill-row">
          {(f.derivesFrom ?? []).map((d) => (
            <button key={d} className="pill"
              onClick={() => setSelectedId(d)}>
              ← {d}
            </button>
          ))}
        </div>
        <h4>Why it works</h4><p>{f.whyWorks}</p>
        <h4>When it fails</h4>
        <ul>{f.failureCases.map((x, i) => <li key={i}>{x}</li>)}</ul>
        <h4>Connections</h4>
        <ul>{f.connections.map((x, i) => <li key={i}>{x}</li>)}</ul>
      </div>
    </div>
  );
}
