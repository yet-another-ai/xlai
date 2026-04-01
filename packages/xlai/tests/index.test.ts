import { afterEach, describe, expect, it, vi } from 'vitest';

import * as wasmModule from '../pkg/xlai_wasm.js';
import { chat, initXlai, packageVersion } from '../src/index';

describe('xlai wasm package', () => {
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

  it('delegates chat requests to the wasm API', async () => {
    const chatSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          chat: (options: unknown) => Promise<unknown>;
        },
        'chat',
      )
      .mockResolvedValue({
        message: {
          role: 'assistant',
          content: 'xlai test reply',
        },
        finishReason: 'stopped',
        usage: {
          inputTokens: 3,
          outputTokens: 4,
          totalTokens: 7,
        },
      });

    vi.stubEnv('OPENAI_API_KEY', 'test-key');
    vi.stubEnv('OPENAI_BASE_URL', 'https://example.com/v1/');
    vi.stubEnv('OPENAI_MODEL', 'test-model');

    await expect(
      chat({
        prompt: 'Say hi.',
        systemPrompt: 'Be concise.',
        temperature: 0.2,
        maxOutputTokens: 64,
      }),
    ).resolves.toEqual({
      message: {
        role: 'assistant',
        content: 'xlai test reply',
      },
      finishReason: 'stopped',
      usage: {
        inputTokens: 3,
        outputTokens: 4,
        totalTokens: 7,
      },
    });

    expect(chatSpy).toHaveBeenCalledWith({
      prompt: 'Say hi.',
      apiKey: 'test-key',
      baseUrl: 'https://example.com/v1/',
      model: 'test-model',
      systemPrompt: 'Be concise.',
      temperature: 0.2,
      maxOutputTokens: 64,
    });
  });
});
