import { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

export function Latex({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, { throwOnError: false, displayMode: block });
    } catch {
      return null; // trust boundary: fall back to escaped text, never raw HTML
    }
  }, [tex, block]);
  return html === null
    ? <span aria-label={tex}>{tex}</span>
    : <span dangerouslySetInnerHTML={{ __html: html }} aria-label={tex} />;
}
