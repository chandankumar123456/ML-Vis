import { describe, it, expect } from 'vitest';
import { loadAllTopics } from '../registry/loadTopics';
import { listTopics } from '../registry/topicRegistry';
import { computeRun } from '../engine/core';

// top-level await: topic modules register() during loadAllTopics, so describe
// generation below sees the real registry (module-scope sync listTopics() would
// be empty — the plan's original runner generated zero topic tests)
await loadAllTopics();

describe('all topic testCases', () => {
  it('registers at least the Wave-0 reference topics', () => {
    expect(listTopics().length).toBeGreaterThanOrEqual(2); // Wave 0: 2 topics
  });

  for (const topic of listTopics()) {
    if (topic.testCases.length === 0) continue;
    describe(topic.id, () => {
      for (const tc of topic.testCases) {
        it(tc.name, () => {
          const run = computeRun(topic.simulation, tc.params, tc.maxSteps ?? 500);
          if (tc.expect.converged !== undefined) {
            if (tc.expect.converged) {
              expect(run.telemetry.failedAtStep).toBeUndefined();
            } else {
              expect(run.telemetry.failedAtStep).toBeDefined();
            }
          }
          if (tc.expect.finalMetrics) {
            const m = run.snapshots[run.snapshots.length - 1].metrics;
            for (const [k, pred] of Object.entries(tc.expect.finalMetrics)) {
              // union type: number | ((v: number) => boolean) — dispatch, don't call
              if (typeof pred === 'function') expect(pred(m[k]), `metric ${k} failed for ${tc.name}`).toBe(true);
              else expect(m[k]).toBeCloseTo(pred, 6);
            }
          }
          if (tc.expect.finalAlgorithm) {
            const a = run.snapshots[run.snapshots.length - 1].algorithm;
            for (const [k, pred] of Object.entries(tc.expect.finalAlgorithm)) {
              // union type: ParamValue | ((v: ParamValue) => boolean) — dispatch, don't call
              if (typeof pred === 'function') expect(pred(a[k]), `algorithm ${k} failed for ${tc.name}`).toBe(true);
              else expect(a[k]).toBe(pred);
            }
          }
          if (tc.expect.eventLabels) {
            const labels = run.snapshots.flatMap((s) => s.events.map((e) => e.label));
            for (const lbl of tc.expect.eventLabels) expect(labels).toContain(lbl);
          }
        });
      }
    });
  }
});
