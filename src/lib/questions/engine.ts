// src/lib/questions/engine.ts
import type { Question } from '../../engine/types';

export function isCorrect(q: Question, answer: string | number): boolean {
  if (q.mode === 'nat') {
    const a = Number(answer);
    const expected = Number(q.answer);
    if (!Number.isFinite(a)) return false;
    // epsilon guards against float artifacts (e.g. 0.43 - 0.42 = 0.010000000000000009)
    return Math.abs(a - expected) <= (q.tolerance ?? 0) + Number.EPSILON;
  }
  return String(answer) === String(q.answer);
}

export interface GradeResult {
  questionId: string;
  answered: string | number;
  correct: boolean;
}

export function gradeAnswer(q: Question, answer: string | number): GradeResult {
  return { questionId: q.id, answered: answer, correct: isCorrect(q, answer) };
}

export function pickQuestions(questions: Question[], mode: Question['mode'] | 'all', n: number): Question[] {
  const pool = mode === 'all' ? questions : questions.filter((q) => q.mode === mode);
  return [...pool].sort(() => Math.random() - 0.5).slice(0, n);
}
