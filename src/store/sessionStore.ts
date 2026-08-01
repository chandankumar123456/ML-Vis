import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { SessionBundle } from '../engine/types';

interface SessionState {
  sessions: SessionBundle[];
  saveSession(bundle: SessionBundle): void;
  deleteSession(savedAt: string): void;
  resumeSession(savedAt: string): SessionBundle | undefined;
  reset(): void;
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      sessions: [],
      saveSession: (bundle) =>
        set((s) => {
          const rest = s.sessions.filter((x) => x.savedAt !== bundle.savedAt);
          return { sessions: [bundle, ...rest].slice(0, 20) };
        }),
      deleteSession: (savedAt) =>
        set((s) => ({ sessions: s.sessions.filter((x) => x.savedAt !== savedAt) })),
      resumeSession: (savedAt) => get().sessions.find((x) => x.savedAt === savedAt),
      reset: () => set({ sessions: [] }),
    }),
    { name: 'mlv-sessions' }
  )
);
