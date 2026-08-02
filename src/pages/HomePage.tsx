import { Link } from 'react-router-dom';
import { listTopics } from '../registry/topicRegistry';
import { useProgressStore } from '../store/progressStore';

export function HomePage() {
  const topics = listTopics();
  const completed = useProgressStore((s) => s.completed);
  return (
    <div>
      <h1>GATE DA — Machine Learning Visualizer</h1>
      <p>See the algorithm think. 31 topics, 30+ interactive views, GATE exam mode.</p>
      <div className="topic-grid">
        {topics.map((t) => (
          <Link key={t.id} to={`/topic/${t.id}`} className="topic-card">
            <h3>{t.title}</h3>
            <small>Weightage: {t.metadata.gateWeightage} · Priority {t.metadata.revisionPriority}</small>
            <div>{completed[t.id] ? '✓ started' : 'not started'}</div>
          </Link>
        ))}
      </div>
    </div>
  );
}
