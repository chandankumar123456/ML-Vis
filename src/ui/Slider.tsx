import type { ParamSchema } from '../engine/types';

export function Slider({ schema, value, onChange }: {
  schema: ParamSchema; value: number; onChange: (v: number) => void;
}) {
  return (
    <label className="slider-row">
      <span>{schema.label}: <b>{value}</b></span>
      <input
        type="range" min={schema.min} max={schema.max} step={schema.step ?? 0.01}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
