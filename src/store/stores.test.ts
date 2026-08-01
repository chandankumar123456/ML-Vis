import { describe, it, expect, beforeEach } from 'vitest';
import { useSettingsStore } from './settingsStore';
import { useProgressStore } from './progressStore';
import { useAnalyticsStore } from './analyticsStore';
import { useSessionStore } from './sessionStore';

beforeEach(() => {
  localStorage.clear();
  useSettingsStore.getState().reset();
  useProgressStore.getState().reset();
  useAnalyticsStore.getState().reset();
  useSessionStore.getState().reset();
});

describe('settingsStore', () => {
  it('toggles theme and palette', () => {
    useSettingsStore.getState().setTheme('dark');
    expect(useSettingsStore.getState().theme).toBe('dark');
    useSettingsStore.getState().setPalette('deuteranopia');
    expect(useSettingsStore.getState().palette).toBe('deuteranopia');
  });
});

describe('progressStore', () => {
  it('marks views complete and bookmarks topics', () => {
    useProgressStore.getState().markView('gd', 'geometry');
    useProgressStore.getState().toggleBookmark('gd');
    const s = useProgressStore.getState();
    expect(s.completed['gd']?.viewsDone['geometry']).toBe(true);
    expect(s.isTopicComplete('gd')).toBe(false); // not enough views done
  });
});

describe('analyticsStore', () => {
  it('records question attempts and time', () => {
    useAnalyticsStore.getState().recordQuestion('gd-004', true, 'gd');
    useAnalyticsStore.getState().addTime('gd', 30);
    const s = useAnalyticsStore.getState();
    expect(s.questionsAttempted['gd-004']).toBeDefined();
    expect(s.timePerTopic['gd']).toBe(30);
  });
});

describe('sessionStore', () => {
  it('saves and lists sessions', () => {
    useSessionStore.getState().saveSession({ topicId: 'gd', moduleVersion: 1, params: {}, step: 3, activeView: 'geometry', bookmarks: [], savedAt: 'x' });
    expect(useSessionStore.getState().sessions.length).toBe(1);
  });
});
