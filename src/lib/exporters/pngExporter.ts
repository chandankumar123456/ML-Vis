// src/lib/exporters/pngExporter.ts
export function snapshotToPng(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL('image/png');
}

export function downloadPng(canvas: HTMLCanvasElement, filename: string): void {
  const a = document.createElement('a');
  a.download = filename;
  a.href = snapshotToPng(canvas);
  a.click();
}
