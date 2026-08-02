// src/visualizers/QuestionPlayer.tsx
import { useState } from 'react';
import { gradeAnswer } from '../lib/questions/engine';
import { useAnalyticsStore } from '../store/analyticsStore';
import type { Question } from '../engine/types';

export function QuestionPlayer({ questions, topicId }: {
  questions: Question[]; topicId: string;
}) {
  const [idx, setIdx] = useState(0);
  const [answered, setAnswered] = useState<{ qid: string; correct: boolean } | null>(null);
  const [input, setInput] = useState('');
  const q = questions[idx];

  const submit = (answer: string | number) => {
    const g = gradeAnswer(q, answer);
    setAnswered({ qid: q.id, correct: g.correct });
    useAnalyticsStore.getState().recordQuestion(q.id, g.correct, topicId);
  };

  if (!q) return <div>No questions yet.</div>;

  return (
    <div className="question-player" key={q.id}>
      <div className="q-meta">
        <span>{q.mode}</span>
        <span>Difficulty {'★'.repeat(q.difficulty)}</span>
      </div>
      <p className="q-prompt">{q.prompt}</p>

      {q.mode === 'nat' ? (
        <div className="q-input">
          <input value={input} onChange={(e) => setInput(e.target.value)}
            placeholder="Numerical answer" />
          <button onClick={() => submit(input)} disabled={answered !== null}>Submit</button>
        </div>
      ) : (
        <div className="q-options">
          {(q.options ?? []).map((opt, i) => (
            <button key={i} className="q-option" disabled={answered !== null}
              onClick={() => submit(String.fromCharCode(65 + i))}>
              {String.fromCharCode(65 + i)}. {opt}
            </button>
          ))}
        </div>
      )}

      {answered && (
        <div className={answered.correct ? 'q-feedback ok' : 'q-feedback err'}>
          <p><b>{answered.correct ? 'Correct!' : 'Wrong.'}</b></p>
          <p>{q.explanation}</p>
          {!answered.correct && q.trapExplanations && (
            <div className="q-traps">
              <h4>Why options are wrong</h4>
              {Object.entries(q.trapExplanations).map(([k, v]) => (
                <p key={k}><b>{k}:</b> {v}</p>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="q-nav">
        <button disabled={idx === 0} onClick={() => { setIdx(idx - 1); setAnswered(null); setInput(''); }}>← Prev</button>
        <span>{idx + 1}/{questions.length}</span>
        <button disabled={idx === questions.length - 1} onClick={() => { setIdx(idx + 1); setAnswered(null); setInput(''); }}>Next →</button>
      </div>
    </div>
  );
}
