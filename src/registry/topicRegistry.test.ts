import { describe, it, expect } from 'vitest';
import { registerTopic, getTopic, listTopics, getTopicCount, migrateBundle } from './topicRegistry';
import { registerView, getView, viewExists, type ViewProps } from './viewRegistry';
import type { TopicModule, SessionBundle } from '../engine/types';

const fakeTopic = {
  id: 'fake', title: 'Fake', version: 1,
  metadata: {
    gateWeightage: 'High',
    difficultyHeatmap: { conceptual: 3, mathematical: 3, coding: 2, visualization: 2, gateFrequency: 4 },
    estimatedHours: 2, revisionPriority: 'P0', examFrequency: 'Frequent',
    prerequisites: [], relatedTopics: [],
    revision: { quick: '5m', standard: '15m', deep: '30m', mastery: '60m' },
  },
  layers: { foundation: [], core: [], advanced: [] },
  params: [], simulation: null as any, formulas: [], derivations: [],
  questions: [], comparisons: [], failureDemos: [], mistakes: [], testCases: [],
} as TopicModule;

describe('topicRegistry', () => {
  it('registers and retrieves topics', () => {
    registerTopic(fakeTopic);
    expect(getTopic('fake')?.title).toBe('Fake');
    expect(listTopics().length).toBe(1);
    expect(getTopicCount()).toBe(1);
  });
  it('migrates old bundles forward', () => {
    const topic = {
      ...fakeTopic, version: 2,
      migrations: { 1: (b: SessionBundle) => ({ ...b, params: { migrated: true } }) },
    } as TopicModule;
    const old: SessionBundle = { topicId: 'fake', moduleVersion: 0, params: {}, step: 0, activeView: 'x', bookmarks: [], savedAt: 't' };
    const migrated = migrateBundle(topic, old);
    expect(migrated.moduleVersion).toBe(2);
    expect(migrated.params.migrated).toBe(true);
  });
});

describe('viewRegistry', () => {
  it('registers views idempotently', () => {
    const C = (_p: ViewProps) => null;
    registerView('scatter-plot', C);
    expect(viewExists('scatter-plot')).toBe(true);
    expect(getView('scatter-plot')).toBe(C);
    registerView('scatter-plot', C); // no throw on duplicate
  });
});
