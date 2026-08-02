// src/pages/ExamPage.tsx
import { useEffect, useRef, useState } from 'react';
import { listTopics } from '../registry/topicRegistry';
import { pickQuestions } from '../lib/questions/engine';
import { QuestionPlayer } from '../visualizers/QuestionPlayer';
import type { Question } from '../engine/types';

export function ExamPage() {
  const [exam, setExam] = useState<{ questions: Question[]; topicId: string } | null>(null);
  const [seconds, setSeconds] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = (topicId: string) => {
    const topic = listTopics().find((t) => t.id === topicId);
    if (!topic) return;
    setExam({ questions: pickQuestions(topic.questions, 'all', 10), topicId });
    setSeconds(0);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
  };

  const end = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setExam(null);
  };

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  if (!exam) {
    return (
      <div>
        <h1>Exam Mode</h1>
        <p>Pick a topic to start a 10-question timed drill.</p>
        <div className="topic-grid">
          {listTopics().filter((t) => t.questions.length > 0).map((t) => (
            <button key={t.id} onClick={() => start(t.id)} className="topic-card">
              {t.title} ({t.questions.length} questions)
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1>Exam — {exam.topicId}</h1>
      <p>Elapsed: {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, '0')}</p>
      <QuestionPlayer questions={exam.questions} topicId={exam.topicId} />
      <button onClick={end}>End exam</button>
    </div>
  );
}
