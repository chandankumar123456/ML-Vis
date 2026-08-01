import { NavLink, Outlet } from 'react-router-dom';
import { listTopics } from '../registry/topicRegistry';
import { ThemeProvider } from './ThemeProvider';
import { KeyboardShortcuts } from './KeyboardShortcuts';
import { CommandPalette } from './CommandPalette';

export function AppShell() {
  const topics = listTopics();
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
        </aside>
        <main className="content">
          <Outlet />
        </main>
        <CommandPalette />
      </div>
    </ThemeProvider>
  );
}
