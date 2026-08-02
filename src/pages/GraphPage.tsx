// src/pages/GraphPage.tsx
import { KnowledgeGraph } from '../visualizers/knowledgeGraph/KnowledgeGraph';

export function GraphPage() {
  return (
    <div>
      <h1>Knowledge Graph</h1>
      <p>Every concept, connected. Click a topic to open it. Hover an edge for &quot;why this connection&quot;.</p>
      <KnowledgeGraph />
    </div>
  );
}
