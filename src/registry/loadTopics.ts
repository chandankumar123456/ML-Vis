/**
 * Lazy-loads all topic modules. Each module file calls register() which
 * self-registers. import.meta.glob gives per-chunk code splitting.
 */
const topicLoaders = import.meta.glob('../topics/*/module.ts');

export async function loadTopic(topicId: string): Promise<void> {
  const path = `../topics/${topicId}/module.ts`;
  const loader = topicLoaders[path];
  if (!loader) return;
  const mod = (await loader()) as Record<string, unknown>;
  if (typeof mod.register === 'function') {
    (mod.register as () => void)();
  }
}

export async function loadAllTopics(): Promise<number> {
  const keys = Object.keys(topicLoaders);
  await Promise.all(keys.map(async (k) => {
    const mod = (await topicLoaders[k]()) as Record<string, unknown>;
    if (typeof mod.register === 'function') (mod.register as () => void)();
  }));
  return keys.length;
}
