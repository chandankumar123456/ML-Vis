import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface TopicProgress {
  viewsDone: Record<string, boolean>;
  mastered?: boolean;
}

interface ProgressState {
  completed: Record<string, TopicProgress>;
  bookmarks: string[];
  lastVisited?: string;
  markView(topicId: string, view: string): void;
  toggleBookmark(topicId: string): void;
  isTopicComplete(topicId: string): boolean;
  setLastVisited(topicId: string): void;
  reset(): void;
}

export const useProgressStore = create<ProgressState>()(
  persist(
    (set, get) => ({
      completed: {},
      bookmarks: [],
      markView: (topicId, view) =>
        set((s) => ({
          completed: {
            ...s.completed,
            [topicId]: {
              viewsDone: { ...(s.completed[topicId]?.viewsDone ?? {}), [view]: true },
              mastered: s.completed[topicId]?.mastered,
            },
          },
        })),
      toggleBookmark: (topicId) =>
        set((s) => ({
          bookmarks: s.bookmarks.includes(topicId)
            ? s.bookmarks.filter((b) => b !== topicId)
            : [...s.bookmarks, topicId],
        })),
      isTopicComplete: (topicId) => {
        const p = get().completed[topicId];
        if (!p) return false;
        return Object.keys(p.viewsDone).length >= 3; // ≥3 views engaged
      },
      setLastVisited: (lastVisited) => set({ lastVisited }),
      reset: () => set({ completed: {}, bookmarks: [], lastVisited: undefined }),
    }),
    { name: 'mlv-progress' }
  )
);
