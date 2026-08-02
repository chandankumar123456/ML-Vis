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
        switch (s.type) {
          case 'number':
            return <Slider key={s.id} schema={s} value={values[s.id] as number} onChange={set} />;
          case 'toggle':
            return <Toggle key={s.id} label={s.label} value={values[s.id] as boolean} onChange={set} />;
          case 'select':
            return <Select key={s.id} schema={s} value={values[s.id] as string} onChange={set} />;
          default:
            return null;
        }
      })}
    </div>
  );
}
