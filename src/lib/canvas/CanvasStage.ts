export interface Bounds { x: [number, number]; y: [number, number] }
export interface Transform { scale: number; tx: number; ty: number }

export function fitBounds(b: Bounds, w: number, h: number, pad = 40): Transform {
  const sx = (w - 2 * pad) / (b.x[1] - b.x[0] || 1);
  const sy = (h - 2 * pad) / (b.y[1] - b.y[0] || 1);
  const scale = Math.min(sx, sy);
  const tx = (w - (b.x[0] + b.x[1]) * scale) / 2;
  const ty = (h - (b.y[0] + b.y[1]) * scale) / 2;
  return { scale, tx, ty };
}

/** Resolve a CSS custom property — canvas colors cannot use var() directly. */
export function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Hi-DPI canvas wrapper with pan/zoom and world→screen transform. */
export class CanvasStage {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private dpr: number;
  private t: Transform = { scale: 1, tx: 0, ty: 0 };
  private onRepaint: (() => void) | null = null;

  constructor(width: number, height: number) {
    this.canvas = document.createElement('canvas');
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, width * this.dpr);
    this.canvas.height = Math.max(1, height * this.dpr);
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.ctx = this.canvas.getContext('2d')!;
  }

  setRepaint(fn: () => void) { this.onRepaint = fn; }
  requestRepaint() { this.onRepaint?.(); }

  get transform() { return this.t; }

  setBounds(b: Bounds, w: number, h: number) {
    this.t = fitBounds(b, w, h);
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, w * this.dpr);
    this.canvas.height = Math.max(1, h * this.dpr);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
  }

  worldToScreen(x: number, y: number): [number, number] {
    return [x * this.t.scale + this.t.tx, y * this.t.scale + this.t.ty];
  }

  screenToWorld(px: number, py: number): [number, number] {
    return [(px - this.t.tx) / this.t.scale, (py - this.t.ty) / this.t.scale];
  }

  zoomAt(px: number, py: number, factor: number) {
    const [wx, wy] = this.screenToWorld(px, py);
    this.t.scale *= factor;
    this.t.tx = px - wx * this.t.scale;
    this.t.ty = py - wy * this.t.scale;
    this.requestRepaint();
  }

  panBy(dx: number, dy: number) {
    this.t.tx += dx;
    this.t.ty += dy;
    this.requestRepaint();
  }

  clear(w: number, h: number, bg: string) {
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.ctx.fillStyle = bg;
    this.ctx.fillRect(0, 0, w, h);
    // draw methods apply worldToScreen() themselves — keep ctx at DPR scale only
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  drawPath(points: [number, number][], stroke: string, width: number, fill?: string) {
    if (points.length < 2) return;
    const ctx = this.ctx;
    ctx.beginPath();
    const [x0, y0] = this.worldToScreen(points[0][0], points[0][1]);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < points.length; i++) {
      const [x, y] = this.worldToScreen(points[i][0], points[i][1]);
      ctx.lineTo(x, y);
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    ctx.stroke();
  }

  drawCircle(x: number, y: number, r: number, fill: string, stroke?: string) {
    const ctx = this.ctx;
    const [sx, sy] = this.worldToScreen(x, y);
    ctx.beginPath();
    ctx.arc(sx, sy, r * this.t.scale, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
  }

  drawArrow(x1: number, y1: number, x2: number, y2: number, color: string) {
    const ctx = this.ctx;
    const [a, b] = this.worldToScreen(x1, y1);
    const [c, d] = this.worldToScreen(x2, y2);
    const angle = Math.atan2(d - b, c - a);
    const headLen = 8;
    ctx.beginPath();
    ctx.moveTo(a, b);
    ctx.lineTo(c, d);
    ctx.lineTo(c - headLen * Math.cos(angle - Math.PI / 6), d - headLen * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(c, d);
    ctx.lineTo(c - headLen * Math.cos(angle + Math.PI / 6), d - headLen * Math.sin(angle + Math.PI / 6));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}
