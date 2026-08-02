import type { Params, ParamSchema, ParamValue } from '../engine/types';
import { Slider } from './Slider';
import { Toggle } from './Toggle';
import { Select } from './Select';

export function ParamPanel({ schema, values, onChange }: {
  schema: ParamSchema[]; values: Params; onChange: (p: Params) => void;
}) {
  return (
    <div className="param-panel">
      <h3>Parameters</h3>
      {schema.map((s) => {
        const set = (v: ParamValue) => onChange({ ...values, [s.id]: v });
        // fall back to schema default when a key is missing (partial params)
        const v = values[s.id] ?? s.default;
        switch (s.type) {
          case 'number':
            return <Slider key={s.id} schema={s} value={v as number} onChange={set} />;
          case 'toggle':
            return <Toggle key={s.id} label={s.label} value={v as boolean} onChange={set} />;
          case 'select':
            return <Select key={s.id} schema={s} value={v as string} onChange={set} />;
          default:
            return null;
        }
      })}
    </div>
  );
}
