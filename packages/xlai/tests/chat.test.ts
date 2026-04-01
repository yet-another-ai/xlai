import { afterEach, describe, expect, it, vi } from 'vitest';

import * as wasmModule from '../pkg/xlai_wasm.js';
import { chat, createChatSession } from '../src/index';

describe('xlai chat api', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
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

  it('creates chat sessions and normalizes registered tools', async () => {
    const registerTool = vi.fn();
    const prompt = vi.fn().mockResolvedValue({
      message: {
        role: 'assistant',
        content: 'tool session reply',
      },
      finishReason: 'completed',
    });

    const createChatSessionSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          createChatSession: (options: unknown) => {
            registerTool: typeof registerTool;
            prompt: typeof prompt;
          };
        },
        'createChatSession',
      )
      .mockReturnValue({
        registerTool,
        prompt,
      });

    vi.stubEnv('OPENAI_API_KEY', 'test-key');
    vi.stubEnv('OPENAI_BASE_URL', 'https://example.com/v1/');
    vi.stubEnv('OPENAI_MODEL', 'test-model');

    const session = await createChatSession({
      systemPrompt: 'Use tools when needed.',
      temperature: 0.1,
      maxOutputTokens: 80,
    });

    expect(createChatSessionSpy).toHaveBeenCalledWith({
      apiKey: 'test-key',
      baseUrl: 'https://example.com/v1/',
      model: 'test-model',
      systemPrompt: 'Use tools when needed.',
      temperature: 0.1,
      maxOutputTokens: 80,
    });

    session.registerTool(
      {
        name: 'lookup_weather',
        description: 'Lookup current weather.',
        parameters: [
          {
            name: 'city',
            description: 'The city to query.',
            kind: 'string',
            required: true,
          },
        ],
      },
      async (argumentsValue) => ({
        content: `weather for ${(argumentsValue as { city: string }).city}: sunny`,
        isError: false,
        metadata: {
          source: 'test',
        },
      }),
    );

    expect(registerTool).toHaveBeenCalledTimes(1);
    const [definition, callback] = registerTool.mock.calls[0] as [
      {
        parameters: Array<{
          kind: string;
        }>;
      },
      (argumentsValue: unknown) => Promise<unknown>,
    ];
    expect(definition.parameters[0]?.kind).toBe('String');
    await expect(callback({ city: 'Paris' })).resolves.toEqual({
      tool_name: 'lookup_weather',
      content: 'weather for Paris: sunny',
      is_error: false,
      metadata: {
        source: 'test',
      },
    });

    await expect(
      session.prompt('What is the weather in Paris?'),
    ).resolves.toEqual({
      message: {
        role: 'assistant',
        content: 'tool session reply',
      },
      finishReason: 'completed',
    });
    expect(prompt).toHaveBeenCalledWith('What is the weather in Paris?');
  });
});
