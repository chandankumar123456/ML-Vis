import type { Params, SimState, SimulationDef, SnapshotRun, RunTelemetry } from './types';

export function isFiniteState(s: SimState): boolean {
  for (const v of Object.values(s.algorithm)) {
    if (typeof v === 'number' && !Number.isFinite(v)) return false;
  }
  for (const v of Object.values(s.metrics)) {
    if (!Number.isFinite(v)) return false;
  }
  return true;
}

const DEFAULT_MAX_STEPS = 2000;

/**
 * Deterministic: same params → identical snapshot array.
 * Sandboxed: non-finite states or exceptions terminate with telemetry.
 */
export function computeRun(
  sim: SimulationDef,
  params: Params,
  maxSteps = DEFAULT_MAX_STEPS
): SnapshotRun {
  const t0 = performance.now();
  const snapshots: SimState[] = [];
  let state: SimState | null = null;
  let failedAtStep: number | undefined;
  let failureReason: string | undefined;

  try {
    state = sim.initialState(params);
    if (!isFiniteState(state)) {
      return {
        params, snapshots: [state], telemetry: {
          snapshotCount: 1, genMs: performance.now() - t0, memBytes: 0,
          failedAtStep: 0, failureReason: 'initial state non-finite',
        },
      };
    }
    snapshots.push(state);
    for (let i = 1; i < maxSteps; i++) {
      const next = sim.step(params, state);
      if (next === null) break;
      if (!isFiniteState(next)) {
        failedAtStep = i;
        failureReason = 'non-finite value (NaN/Infinity) at step ' + i;
        snapshots.push(next);
        break;
      }
      snapshots.push(next);
      state = next;
    }
    if (failedAtStep === undefined && snapshots.length >= maxSteps) {
      failedAtStep = maxSteps;
      failureReason = 'step budget exceeded (no convergence)';
    }
  } catch (e) {
    failedAtStep = snapshots.length;
    failureReason = e instanceof Error ? e.message : String(e);
  }

  // estBytes must NOT escape the sandbox: a cyclic VisualCommand (index signature
  // allows arbitrary payloads) would otherwise throw out of computeRun itself.
  let estBytes = 0;
  try {
    estBytes = snapshots.reduce((acc, s) => acc + JSON.stringify(s).length, 0);
  } catch {
    estBytes = 0;
  }

  const telemetry: RunTelemetry = {
    snapshotCount: snapshots.length,
    genMs: performance.now() - t0,
    memBytes: estBytes,
    failedAtStep,
    failureReason,
  };
  return { params, snapshots, telemetry };
}

/** Convert a snapshot array into a timeline of stage labels (deduped). */
export function timelineStages(run: SnapshotRun): { label: string; step: number }[] {
  const seen = new Set<string>();
  const stages: { label: string; step: number }[] = [];
  run.snapshots.forEach((s, i) => {
    const label = s.timeline[s.timeline.length - 1];
    if (label && !seen.has(label)) {
      seen.add(label);
      stages.push({ label, step: i });
    }
  });
  return stages;
}
