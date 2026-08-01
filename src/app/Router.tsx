import { Routes, Route } from 'react-router-dom';
import { AppShell } from './AppShell';
import { HomePage } from '../pages/HomePage';
import { TopicPage } from '../pages/TopicPage';
import { GraphPage } from '../pages/GraphPage';
import { JourneyPage } from '../pages/JourneyPage';
import { ExamPage } from '../pages/ExamPage';
import { loadTopic } from '../registry/loadTopics';

export function Router() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="/journey" element={<JourneyPage />} />
        <Route path="/exam" element={<ExamPage />} />
        <Route path="/topic/:topicId" element={<TopicPage loader={loadTopic} />} />
        <Route path="*" element={<HomePage />} />
      </Route>
    </Routes>
  );
}
