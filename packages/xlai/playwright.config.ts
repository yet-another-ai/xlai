import { existsSync, readFileSync } from 'node:fs';
import { defineConfig, devices } from '@playwright/test';

function loadLocalEnvFile(): void {
  const envFilePath = new URL('../../.env', import.meta.url);
  if (!existsSync(envFilePath)) {
    return;
  }

  const fileContents = readFileSync(envFilePath, 'utf8');
  for (const rawLine of fileContents.split(/\r?\n/u)) {
    const line = rawLine.trim();
    if (line === '' || line.startsWith('#')) {
      continue;
    }

    const separatorIndex = line.indexOf('=');
    if (separatorIndex <= 0) {
      continue;
    }

    const key = line.slice(0, separatorIndex).trim();
    let value = line.slice(separatorIndex + 1).trim();

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    process.env[key] ??= value;
  }
}

loadLocalEnvFile();

const isCI = Boolean(
  (
    globalThis as typeof globalThis & {
      process?: { env?: Record<string, string | undefined> };
    }
  ).process?.env?.CI,
);
const reuseExistingServer =
  process.env.PLAYWRIGHT_REUSE_WEB_SERVER === '1';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  retries: isCI ? 2 : 0,
  reporter: isCI ? [['github'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
  ],
  webServer: {
    command:
      'pnpm run build:wasm && pnpm exec vite --host 127.0.0.1 --port 4173 --strictPort',
    port: 4173,
    // Prefer a fresh Vite server for E2E to avoid stale optimized-deps caches.
    reuseExistingServer: !isCI && reuseExistingServer,
    timeout: 120_000,
  },
});
