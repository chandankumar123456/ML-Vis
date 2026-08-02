/// <reference types="vitest/config" />
import { defineConfig, defaultExclude } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 900 },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/testSetup.ts'],
    globals: true,
    // e2e/smoke.spec.ts is Playwright's, not vitest's — keep it out of the unit runner
    exclude: [...defaultExclude, 'e2e/**'],
  },
});
