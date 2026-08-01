export type BusEvent =
  | { type: 'highlight'; payload: { panel: string; id: string; intensity: number } }
  | { type: 'clear-highlights' }
  | { type: 'open-topic'; payload: { topicId: string } }
  | { type: 'navigate-view'; payload: { view: string } }
  | { type: 'playback-cursor'; payload: { step: number } }
  | { type: 'question-answered'; payload: { questionId: string; correct: boolean } }
  | { type: 'explain-step'; payload: { step: number } };

type Handler = (e: BusEvent) => void;

class EventBus {
  private handlers = new Set<Handler>();

  subscribe(h: Handler): () => void {
    this.handlers.add(h);
    return () => { this.handlers.delete(h); };
  }

  emit(e: BusEvent): void {
    for (const h of this.handlers) {
      try { h(e); } catch { /* subscriber errors never break the bus */ }
    }
  }
}

export const eventBus = new EventBus();
