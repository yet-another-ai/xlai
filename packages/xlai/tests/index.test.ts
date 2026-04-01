import { describe, expect, it } from 'vitest';

import { initXlai, packageVersion } from '../src/index';

describe('xlai wasm package', () => {
  it('initializes the wasm module', async () => {
    await expect(initXlai()).resolves.toBeUndefined();
  });

  it('exposes the Rust package version', async () => {
    await expect(packageVersion()).resolves.toBe('0.1.0');
  });
});
