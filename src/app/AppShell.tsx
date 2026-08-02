import { NavLink, Outlet } from 'react-router-dom';
import { listTopics } from '../registry/topicRegistry';
import { ThemeProvider } from './ThemeProvider';
import { KeyboardShortcuts } from './KeyboardShortcuts';
import { CommandPalette } from './CommandPalette';
import { useSettingsStore, type Palette } from '../store/settingsStore';

export function AppShell() {
  const topics = listTopics();
  const theme = useSettingsStore((s) => s.theme);
  const palette = useSettingsStore((s) => s.palette);
  return (
    <ThemeProvider>
      <KeyboardShortcuts />
      <div className="shell">
        <aside className="sidebar">
          <h1>GATE ML Visualizer</h1>
          <nav>
            <NavLink to="/">Home</NavLink>
            <NavLink to="/graph">Knowledge Graph</NavLink>
            <NavLink to="/journey">Learning Journey</NavLink>
            <NavLink to="/exam">Exam Mode</NavLink>
            <div className="topic-list">
              {topics.map((t) => (
                <NavLink key={t.id} to={`/topic/${t.id}`}>{t.title}</NavLink>
              ))}
            </div>
          </nav>
          <div className="settings-row">
            <button onClick={() => useSettingsStore.getState().setTheme(theme === 'dark' ? 'light' : 'dark')}>
              {theme === 'dark' ? '☀ Light' : '🌙 Dark'}
            </button>
            <select aria-label="Colorblind palette" value={palette}
              onChange={(e) => useSettingsStore.getState().setPalette(e.target.value as Palette)}>
              <option value="default">Palette: default</option>
              <option value="deuteranopia">deuteranopia</option>
              <option value="protanopia">protanopia</option>
              <option value="tritanopia">tritanopia</option>
            </select>
          </div>
        </aside>
        <main className="content">
          <Outlet />
        </main>
        <CommandPalette />
      </div>
    </ThemeProvider>
  );
}
