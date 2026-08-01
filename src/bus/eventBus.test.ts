import { describe, it, expect, vi } from 'vitest';
import { eventBus } from './eventBus';

describe('eventBus', () => {
  it('delivers events to subscribers and supports unsubscribe', () => {
    const fn = vi.fn();
    const unsub = eventBus.subscribe(fn);
    eventBus.emit({ type: 'highlight', payload: { panel: 'matrix', id: 'w23', intensity: 1 } });
    expect(fn).toHaveBeenCalledTimes(1);
    unsub();
    eventBus.emit({ type: 'highlight', payload: { panel: 'matrix', id: 'w23', intensity: 1 } });
    expect(fn).toHaveBeenCalledTimes(1);
  });
  it('does not crash with zero subscribers', () => {
    eventBus.emit({ type: 'highlight', payload: { panel: 'x', id: 'y', intensity: 0.5 } });
    expect(true).toBe(true);
  });
});
