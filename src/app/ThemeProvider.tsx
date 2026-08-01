import { useEffect, type ReactNode } from 'react';
import { useSettingsStore } from '../store/settingsStore';

export function ThemeProvider({ children }: { children: ReactNode }) {
  const theme = useSettingsStore((s) => s.theme);
  const palette = useSettingsStore((s) => s.palette);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.palette = palette;
  }, [theme, palette]);
  return <>{children}</>;
}
