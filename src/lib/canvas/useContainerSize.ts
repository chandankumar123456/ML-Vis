import { useEffect, useRef, useState } from 'react';

/** Observe an element's content size; returns [ref, size]. */
export function useContainerSize(initialW: number, initialH: number) {
  const [size, setSize] = useState({ w: initialW, h: initialH });
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setSize({ w: width, h: height });
    });
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  return [ref, size] as const;
}
