import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface QuestionAttempt {
  questionId: string;
  correct: boolean;
  topicId: string;
  at: number;
}

interface AnalyticsState {
  questionsAttempted: Record<string, QuestionAttempt>;
  timePerTopic: Record<string, number>;
  topicVisits: Record<string, number>;
  recordQuestion(questionId: string, correct: boolean, topicId: string): void;
  addTime(topicId: string, seconds: number): void;
  recordVisit(topicId: string): void;
  getWeakestTopics(): string[];
  reset(): void;
}

export const useAnalyticsStore = create<AnalyticsState>()(
  persist(
    (set, get) => ({
      questionsAttempted: {},
      timePerTopic: {},
      topicVisits: {},
      recordQuestion: (questionId, correct, topicId) =>
        set((s) => ({
          questionsAttempted: {
            ...s.questionsAttempted,
            [questionId]: { questionId, correct, topicId, at: Date.now() },
          },
        })),
      addTime: (topicId, seconds) =>
        set((s) => ({ timePerTopic: { ...s.timePerTopic, [topicId]: (s.timePerTopic[topicId] ?? 0) + seconds } })),
      recordVisit: (topicId) =>
        set((s) => ({ topicVisits: { ...s.topicVisits, [topicId]: (s.topicVisits[topicId] ?? 0) + 1 } })),
      getWeakestTopics: () => {
        const wrong: Record<string, number> = {};
        for (const q of Object.values(get().questionsAttempted)) {
          if (!q.correct) wrong[q.topicId] = (wrong[q.topicId] ?? 0) + 1;
        }
        return Object.entries(wrong)
          .sort((a, b) => b[1] - a[1])
          .map(([t]) => t)
          .slice(0, 5);
      },
      reset: () => set({ questionsAttempted: {}, timePerTopic: {}, topicVisits: {} }),
    }),
    { name: 'mlv-analytics' }
  )
);
