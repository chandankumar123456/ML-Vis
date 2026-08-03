// src/visualizers/ExplainStep.tsx
// Expandable panel rendering snapshot.explanation (changed[], why, formulaRef,
// dependsOn, gateConcepts) — Wave-1 registry component, registered as 'explain-step'.
import { useState } from 'react';
import type { SimState, TopicModule } from '../engine/types';

export function ExplainStep({ snapshot, topic }: {
  snapshot?: SimState | null; topic?: TopicModule;
}) {
  const [open, setOpen] = useState(false);
  const exp = snapshot?.explanation;
  if (!exp) return null;

  // formulaRef is looked up in the topic's formulas when the topic is provided;
  // an unknown id still renders as a bare chip.
  const formula = exp.formulaRef
    ? (topic?.formulas ?? []).find((f) => f.id === exp.formulaRef)
    : undefined;

  return (
    <div className="explain-step">
      <button className="explain-toggle" aria-expanded={open} aria-controls="explain-body"
        onClick={() => setOpen((v) => !v)}>
        <span className="explain-chevron" aria-hidden="true">{open ? '▾' : '▸'}</span>
        Why did this step change?
      </button>
      {open && (
        <div className="explain-body" id="explain-body">
          {exp.changed.length > 0 && (
            <>
              <h4 className="explain-label">Changed</h4>
              <div className="explain-chips">
                {exp.changed.map((c, i) => (
                  <span key={i} className="explain-chip">{c}</span>
                ))}
              </div>
            </>
          )}
          {exp.why && (
            <>
              <h4 className="explain-label">Why</h4>
              <p className="explain-why">{exp.why}</p>
            </>
          )}
          {exp.dependsOn.length > 0 && (
            <>
              <h4 className="explain-label">Depends on</h4>
              <div className="explain-chips">
                {exp.dependsOn.map((d, i) => (
                  <span key={i} className="explain-chip">{d}</span>
                ))}
              </div>
            </>
          )}
          {exp.gateConcepts.length > 0 && (
            <>
              <h4 className="explain-label">GATE concepts</h4>
              <div className="explain-chips">
                {exp.gateConcepts.map((g, i) => (
                  <span key={i} className="explain-chip">{g}</span>
                ))}
              </div>
            </>
          )}
          {exp.formulaRef && (
            <div className="explain-formula">
              <h4 className="explain-label">Formula</h4>
              <span className="explain-chip">{exp.formulaRef}</span>
              {formula && <p className="explain-latex">{formula.latex}</p>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
