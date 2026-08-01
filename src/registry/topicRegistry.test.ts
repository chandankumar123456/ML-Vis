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

const mkBundle = (moduleVersion: number, params = {}): SessionBundle =>
  ({ topicId: 'fake', moduleVersion, params, step: 0, activeView: 'x', bookmarks: [], savedAt: 't' });

describe('topicRegistry', () => {
  it('registers and retrieves topics', () => {
    registerTopic({ ...fakeTopic, id: 'fake-a', title: 'Fake A' } as TopicModule);
    expect(getTopic('fake-a')?.title).toBe('Fake A');
    expect(listTopics().map((t) => t.id)).toContain('fake-a');
    expect(getTopicCount()).toBeGreaterThanOrEqual(1);
  });
  it('keeps newest version on re-registration', () => {
    registerTopic({ ...fakeTopic, id: 'fake-b', title: 'B v0', version: 0 } as TopicModule);
    registerTopic({ ...fakeTopic, id: 'fake-b', title: 'B v2', version: 2 } as TopicModule);
    registerTopic({ ...fakeTopic, id: 'fake-b', title: 'B v1', version: 1 } as TopicModule);
    expect(getTopic('fake-b')?.title).toBe('B v2'); // older must NOT overwrite newer
    expect(getTopic('fake-b')?.version).toBe(2);
  });
  it('migrates old bundles forward', () => {
    const topic = {
      ...fakeTopic, id: 'fake-c', version: 2,
      migrations: { 1: (b: SessionBundle) => ({ ...b, params: { migrated: true } }) },
    } as TopicModule;
    const migrated = migrateBundle(topic, mkBundle(0));
    expect(migrated.moduleVersion).toBe(2);
    expect(migrated.params.migrated).toBe(true);
  });
  it('does not re-run migrations on bundles already at the key version', () => {
    const topic = {
      ...fakeTopic, id: 'fake-d', version: 2,
      migrations: { 1: (b: SessionBundle) => ({ ...b, params: { migrated: true } }) },
    } as TopicModule;
    const atV1 = mkBundle(1, { migrated: false });
    const migrated = migrateBundle(topic, atV1);
    expect(migrated.moduleVersion).toBe(2);
    expect(migrated.params.migrated).toBe(false); // untouched
  });
  it('stamps version even without migrations', () => {
    const migrated = migrateBundle(fakeTopic, mkBundle(0));
    expect(migrated.moduleVersion).toBe(1);
  });
  it('refuses to downgrade newer bundles', () => {
    const migrated = migrateBundle(fakeTopic, mkBundle(5));
    expect(migrated.moduleVersion).toBe(5); // untouched
    expect(migrated.params).toEqual({});
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
