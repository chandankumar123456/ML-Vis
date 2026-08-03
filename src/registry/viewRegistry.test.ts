// Classifier registry coverage (Task 5): register/get round-trip, unknown id,
// overwrite semantics, and the literal (x, y, params) => number call contract.
import { describe, it, expect } from 'vitest';
import { registerClassifier, getClassifier } from './viewRegistry';
import type { Params } from '../engine/types';

describe('viewRegistry classifiers', () => {
  it('round-trips a registered classifier', () => {
    const fn = (_x: number, _y: number, _p: Params) => 0;
    registerClassifier('c-roundtrip', fn);
    expect(getClassifier('c-roundtrip')).toBe(fn);
  });

  it('returns undefined for an unknown id', () => {
    expect(getClassifier('c-unknown')).toBeUndefined();
  });

  it('overwrites an existing registration with the same id', () => {
    const first = (_x: number, _y: number, _p: Params) => 1;
    const second = (_x: number, _y: number, _p: Params) => 2;
    registerClassifier('c-overwrite', first);
    expect(getClassifier('c-overwrite')).toBe(first);
    registerClassifier('c-overwrite', second);
    expect(getClassifier('c-overwrite')).toBe(second);
  });

  it('passes (x, y, params) through and returns the fn result', () => {
    registerClassifier('c-call', (x, y, p) => x + y + (p.k as number));
    const got = getClassifier('c-call');
    expect(got).toBeDefined();
    expect(got?.(1, 2, { k: 3 })).toBe(6);
  });
});
