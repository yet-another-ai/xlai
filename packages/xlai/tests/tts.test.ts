import { afterEach, describe, expect, it, vi } from 'vitest';

import * as wasmModule from '../pkg/xlai_wasm.js';
import { tts, ttsStream } from '../src/index';

describe('xlai tts api', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  it('normalizes inline audio payloads and responses', async () => {
    const ttsSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          tts: (options: unknown) => Promise<unknown>;
        },
        'tts',
      )
      .mockResolvedValue({
        audio: {
          kind: 'inline_data',
          mime_type: 'audio/mpeg',
          data: 'SUQzBA==',
        },
        mime_type: 'audio/mpeg',
      });

    vi.stubEnv('OPENAI_API_KEY', 'tts-key');

    await expect(
      tts({
        input: 'say hi',
        voice: {
          kind: 'clone',
          references: [
            {
              audio: {
                kind: 'inline_data',
                mime_type: 'audio/mpeg',
                data_base64: 'SUQzBA==',
              },
            },
          ],
        },
      }),
    ).resolves.toEqual({
      audio: {
        kind: 'inline_data',
        mime_type: 'audio/mpeg',
        data: 'SUQzBA==',
      },
      mime_type: 'audio/mpeg',
    });

    expect(ttsSpy).toHaveBeenCalledWith({
      input: 'say hi',
      apiKey: 'tts-key',
      voice: {
        kind: 'clone',
        references: [
          {
            audio: {
              kind: 'inline_data',
              mime_type: 'audio/mpeg',
              data: 'SUQzBA==',
            },
          },
        ],
      },
    });
  });

  it('normalizes streamed audio deltas to the data field', async () => {
    const ttsStreamSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          ttsStream: (options: unknown) => Promise<unknown>;
        },
        'ttsStream',
      )
      .mockResolvedValue([
        { type: 'started', mime_type: 'audio/mpeg' },
        { type: 'audio_delta', data: 'SUQzBA==' },
        {
          type: 'finished',
          response: {
            audio: {
              kind: 'inline_data',
              mime_type: 'audio/mpeg',
              data: 'SUQzBA==',
            },
            mime_type: 'audio/mpeg',
          },
        },
      ]);

    vi.stubEnv('OPENAI_API_KEY', 'tts-key');

    await expect(
      ttsStream({
        input: 'say hi',
        voice: { kind: 'preset', name: 'alloy' },
      }),
    ).resolves.toEqual([
      { type: 'started', mime_type: 'audio/mpeg' },
      { type: 'audio_delta', data: 'SUQzBA==' },
      {
        type: 'finished',
        response: {
          audio: {
            kind: 'inline_data',
            mime_type: 'audio/mpeg',
            data: 'SUQzBA==',
          },
          mime_type: 'audio/mpeg',
        },
      },
    ]);

    expect(ttsStreamSpy).toHaveBeenCalledWith({
      input: 'say hi',
      apiKey: 'tts-key',
      voice: { kind: 'preset', name: 'alloy' },
    });
  });
});
