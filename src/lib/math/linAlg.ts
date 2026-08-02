// src/lib/math/linAlg.ts
export function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
}

export function transpose<T>(m: T[][]): T[][] {
  return m[0].map((_, c) => m.map((r) => r[c]));
}

export function matMul(a: number[][], b: number[][]): number[][] {
  const m = a.length, k = a[0].length, n = b[0].length;
  const out: number[][] = Array.from({ length: m }, () => Array(n).fill(0));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let t = 0; t < k; t++) out[i][j] += a[i][t] * b[t][j];
  return out;
}

/** Solve 2x2 system [[a,b],[c,d]]·[x,y] = [e,f] via Cramer's rule. */
export function solve2x2(rows: number[][], rhs: number[]): [number, number] {
  const [[a, b], [c, d]] = rows;
  const [e, f] = rhs;
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-12) return [0, 0];
  return [(e * d - b * f) / det, (a * f - e * c) / det];
}
