import { afterEach, describe, expect, it, vi } from 'vitest';

import * as wasmModule from '../pkg/xlai_wasm.js';
import { generateImage } from '../src/index';

describe('xlai image generation api', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  it('delegates image generation requests to the wasm API', async () => {
    const generateImageSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          generateImage: (options: unknown) => Promise<unknown>;
        },
        'generateImage',
      )
      .mockResolvedValue({
        images: [
          {
            image: {
              kind: 'inline_data',
              mime_type: 'image/png',
              data: 'iVBORw0KGgo=',
            },
            mime_type: 'image/png',
            revised_prompt: 'A fox leaping through glowing flowers',
          },
        ],
      });

    vi.stubEnv('OPENAI_API_KEY', 'image-key');
    vi.stubEnv('OPENAI_BASE_URL', 'https://example.com/v1/');
    vi.stubEnv('OPENAI_IMAGE_MODEL', 'gpt-image-env');

    await expect(
      generateImage({
        prompt: 'A fox in a flower field',
        size: '1024x1024',
        quality: 'high',
        background: 'transparent',
        outputFormat: 'png',
        count: 2,
      }),
    ).resolves.toEqual({
      images: [
        {
          image: {
            kind: 'inline_data',
            mime_type: 'image/png',
            data: 'iVBORw0KGgo=',
          },
          mime_type: 'image/png',
          revised_prompt: 'A fox leaping through glowing flowers',
        },
      ],
    });

    expect(generateImageSpy).toHaveBeenCalledWith({
      prompt: 'A fox in a flower field',
      apiKey: 'image-key',
      baseUrl: 'https://example.com/v1/',
      imageModel: 'gpt-image-env',
      size: '1024x1024',
      quality: 'high',
      background: 'transparent',
      outputFormat: 'png',
      count: 2,
    });
  });
});
