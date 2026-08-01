import type { TopicModule, SessionBundle } from '../engine/types';

const topics = new Map<string, TopicModule>();

export function registerTopic(m: TopicModule): void {
  if (topics.has(m.id)) {
    const existing = topics.get(m.id)!;
    if (existing.version >= m.version) return; // keep newest
  }
  topics.set(m.id, m);
}

export function getTopic(id: string): TopicModule | undefined { return topics.get(id); }
export function listTopics(): TopicModule[] { return [...topics.values()]; }
export function getTopicCount(): number { return topics.size; }

export function migrateBundle(topic: TopicModule, bundle: SessionBundle): SessionBundle {
  let b = bundle;
  const migrations = topic.migrations ?? {};
  for (const v of Object.keys(migrations).map(Number).sort((a, b) => a - b)) {
    if (b.moduleVersion < v) b = migrations[v](b);
  }
  return { ...b, moduleVersion: topic.version };
}
