import { describe, it, expect } from 'vitest';
import { gradeAnswer, isCorrect, pickQuestions } from './engine';
import type { Question } from '../../engine/types';

const q: Question = {
  id: 't1', mode: 'nat', prompt: 'Compute x', answer: 0.42, tolerance: 0.01,
  explanation: '', concepts: [], difficulty: 2, tags: [],
};

describe('question engine', () => {
  it('grades NAT within tolerance', () => {
    expect(isCorrect(q, 0.42)).toBe(true);
    expect(isCorrect(q, 0.43)).toBe(true);
    expect(isCorrect(q, 0.45)).toBe(false);
  });
  it('grades MCQ by letter', () => {
    const mcq: Question = { ...q, mode: 'gate-mcq', options: ['a', 'b', 'c', 'd'], answer: 'c' };
    expect(isCorrect(mcq, 'c')).toBe(true);
    expect(isCorrect(mcq, 'a')).toBe(false);
  });
  it('records grade results', () => {
    const g = gradeAnswer(q, 0.42);
    expect(g.correct).toBe(true);
    expect(g.answered).toBe(0.42);
  });
  it('grades MCQ case-insensitively (player submits uppercase letters)', () => {
    const mcq: Question = { ...q, mode: 'gate-mcq', options: ['a', 'b', 'c', 'd'], answer: 'c' };
    expect(isCorrect(mcq, 'C')).toBe(true);
    expect(isCorrect(mcq, 'A')).toBe(false);
  });
  it('picks N questions filtered by mode', () => {
    const pool: Question[] = [
      q,
      { ...q, id: 'q2', mode: 'gate-mcq', options: ['a'], answer: 'a' },
      { ...q, id: 'q3', mode: 'nat' },
      { ...q, id: 'q4', mode: 'nat' },
    ];
    const picked = pickQuestions(pool, 'nat', 2);
    expect(picked.length).toBe(2);
    expect(picked.every((x) => x.mode === 'nat')).toBe(true);
  });
});
