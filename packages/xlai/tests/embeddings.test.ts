import { afterEach, describe, expect, it, vi } from 'vitest';

import * as wasmModule from '../pkg/xlai_wasm.js';
import { embed } from '../src/index';

describe('xlai embeddings api', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  it('delegates embedding requests to the wasm API with env fallback', async () => {
    const embedSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          embed: (options: unknown) => Promise<unknown>;
        },
        'embed',
      )
      .mockResolvedValue({
        vectors: [[0.1, 0.2, 0.3]],
        usage: {
          inputTokens: 3,
          outputTokens: 0,
          totalTokens: 3,
        },
      });

    vi.stubEnv('OPENAI_API_KEY', 'embed-key');
    vi.stubEnv('OPENAI_BASE_URL', 'https://example.com/v1/');
    vi.stubEnv('OPENAI_MODEL', 'gpt-fallback');
    vi.stubEnv('OPENAI_EMBEDDING_MODEL', 'text-embedding-env');

    await expect(
      embed({
        inputs: ['hello world'],
        dimensions: 256,
      }),
    ).resolves.toEqual({
      vectors: [[0.1, 0.2, 0.3]],
      usage: {
        inputTokens: 3,
        outputTokens: 0,
        totalTokens: 3,
      },
    });

    expect(embedSpy).toHaveBeenCalledWith({
      inputs: ['hello world'],
      apiKey: 'embed-key',
      baseUrl: 'https://example.com/v1/',
      embeddingModel: 'text-embedding-env',
      model: 'gpt-fallback',
      dimensions: 256,
    });
  });

  it('prefers explicit embeddingModel over model and env', async () => {
    const embedSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          embed: (options: unknown) => Promise<unknown>;
        },
        'embed',
      )
      .mockResolvedValue({ vectors: [[1, 2, 3]] });

    vi.stubEnv('OPENAI_API_KEY', 'embed-key');
    vi.stubEnv('OPENAI_MODEL', 'gpt-fallback');
    vi.stubEnv('OPENAI_EMBEDDING_MODEL', 'text-embedding-env');

    await embed({
      inputs: ['hello world'],
      model: 'text-embedding-model',
      embeddingModel: 'text-embedding-explicit',
    });

    expect(embedSpy).toHaveBeenCalledWith({
      inputs: ['hello world'],
      apiKey: 'embed-key',
      baseUrl: undefined,
      embeddingModel: 'text-embedding-explicit',
      model: 'text-embedding-model',
    });
  });
});
