// src/visualizers/MistakeView.tsx
import { useState } from 'react';
import { Latex } from '../ui/Latex';
import type { Mistake } from '../engine/types';

export function MistakeView({ mistakes }: { mistakes: Mistake[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="mistake-view">
      <h3>Common Mistakes &amp; GATE Traps</h3>
      {mistakes.map((m) => (
        <div key={m.id} className="mistake-card">
          <button className="mistake-header" onClick={() => setOpen(open === m.id ? null : m.id)}>
            {m.gateTrap && <span className="trap-badge">GATE TRAP</span>}
            {m.pattern}
          </button>
          {open === m.id && (
            <div className="mistake-body">
              {m.example && <p><b>Example:</b> <Latex tex={m.example} /></p>}
              <p><b>Why wrong:</b> {m.whyWrong}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
