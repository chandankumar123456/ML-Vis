import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Theme = 'light' | 'dark';
export type Palette = 'default' | 'deuteranopia' | 'protanopia' | 'tritanopia';

interface SettingsState {
  theme: Theme;
  palette: Palette;
  reducedMotion: boolean;
  showTelemetry: boolean;
  setTheme(t: Theme): void;
  setPalette(p: Palette): void;
  setReducedMotion(v: boolean): void;
  setShowTelemetry(v: boolean): void;
  reset(): void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'light',
      palette: 'default',
      reducedMotion: false,
      showTelemetry: false,
      setTheme: (theme) => set({ theme }),
      setPalette: (palette) => set({ palette }),
      setReducedMotion: (reducedMotion) => set({ reducedMotion }),
      setShowTelemetry: (showTelemetry) => set({ showTelemetry }),
      reset: () => set({ theme: 'light', palette: 'default', reducedMotion: false, showTelemetry: false }),
    }),
    { name: 'mlv-settings', version: 1 }
  )
);
