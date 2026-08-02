import type { ParamSchema } from '../engine/types';

export function Select({ schema, value, onChange }: {
  schema: ParamSchema; value: string; onChange: (v: string) => void;
}) {
  return (
    <label className="select-row">
      <span>{schema.label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {(schema.options ?? []).map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}
