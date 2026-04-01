import { afterEach, describe, expect, it, vi } from 'vitest';

import { initXlai, packageVersion } from '../src/index';

describe('xlai package bootstrap', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  it('initializes the wasm module', async () => {
    await expect(initXlai()).resolves.toBeUndefined();
  });

  it('exposes the Rust package version', async () => {
    await expect(packageVersion()).resolves.toBe('0.1.0');
  });
});
