import type { Params, ParamSchema } from '../engine/types';

export function defaultParams(schema: ParamSchema[]): Params {
  const p: Params = {};
  for (const s of schema) p[s.id] = s.default;
  return p;
}
