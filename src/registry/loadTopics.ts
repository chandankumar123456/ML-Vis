/**
 * Lazy-loads all topic modules. Each module file calls register() which
 * self-registers. import.meta.glob gives per-chunk code splitting.
 */
const topicLoaders = import.meta.glob('../topics/*/module.ts');

async function registerModule(mod: Record<string, unknown>): Promise<void> {
  if (typeof mod.register === 'function') (mod.register as () => void)();
}

export async function loadTopic(topicId: string): Promise<boolean> {
  const path = `../topics/${topicId}/module.ts`;
  const loader = topicLoaders[path];
  if (!loader) return false;
  try {
    await registerModule((await loader()) as Record<string, unknown>);
    return true;
  } catch (e) {
    console.error(`[registry] failed to load topic module ${topicId}`, e);
    return false;
  }
}

export async function loadAllTopics(): Promise<number> {
  const keys = Object.keys(topicLoaders);
  let ok = 0;
  await Promise.all(keys.map(async (k) => {
    try {
      await registerModule((await topicLoaders[k]()) as Record<string, unknown>);
      ok++;
    } catch (e) {
      // one broken module must not brick the whole topic graph
      console.error(`[registry] failed to load topic module ${k}`, e);
    }
  }));
  return ok;
}
