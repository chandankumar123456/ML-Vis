// src/visualizers/DerivationPlayer.tsx
import { useState } from 'react';
import { Latex } from '../ui/Latex';
import type { Derivation, TopicModule } from '../engine/types';

export function DerivationPlayer({ derivations, topic }: {
  derivations?: Derivation[]; topic?: TopicModule;
}) {
  const list = derivations ?? topic?.derivations ?? [];
  const [idx, setIdx] = useState(0);
  const [revealed, setRevealed] = useState(1);
  const d = list[idx];
  if (!d) return <div className="derivation-player">No derivations yet.</div>;
  return (
    <div className="derivation-player">
      <h3>{d.title}</h3>
      <div className="derivation-nav">
        <button disabled={idx === 0}
          onClick={() => { setIdx(idx - 1); setRevealed(1); }}>← Prev derivation</button>
        <span>{idx + 1}/{list.length}</span>
        <button disabled={idx === list.length - 1}
          onClick={() => { setIdx(idx + 1); setRevealed(1); }}>Next derivation →</button>
      </div>
      <ol className="derivation-steps">
        {d.steps.slice(0, revealed).map((s, i) => (
          <li key={i}>
            <Latex tex={s.latex} block />
            <p className="derivation-why">{s.justification}</p>
          </li>
        ))}
      </ol>
      {revealed < d.steps.length && (
        <button onClick={() => setRevealed(revealed + 1)}>Reveal next step</button>
      )}
    </div>
  );
}
