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

  it('delegates multimodal chat requests to the wasm API', async () => {
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
          content: {
            parts: [{ type: 'text', text: 'looks like a cat' }],
          },
        },
        finishReason: 'completed',
      });

    vi.stubEnv('OPENAI_API_KEY', 'test-key');

    await expect(
      chat({
        content: {
          parts: [
            { type: 'text', text: 'Describe this audio clip.' },
            {
              type: 'audio',
              source: {
                kind: 'inline_data',
                mime_type: 'audio/wav',
                data_base64: 'UklGRg==',
              },
              mime_type: 'audio/wav',
            },
          ],
        },
      }),
    ).resolves.toEqual({
      message: {
        role: 'assistant',
        content: {
          parts: [{ type: 'text', text: 'looks like a cat' }],
        },
      },
      finishReason: 'completed',
    });

    expect(chatSpy).toHaveBeenCalledWith({
      prompt: '',
      content: {
        parts: [
          { type: 'text', text: 'Describe this audio clip.' },
          {
            type: 'audio',
            source: {
              kind: 'inline_data',
              mime_type: 'audio/wav',
              data: 'UklGRg==',
            },
            mime_type: 'audio/wav',
          },
        ],
      },
      apiKey: 'test-key',
      baseUrl: undefined,
      model: undefined,
      systemPrompt: undefined,
      temperature: undefined,
      maxOutputTokens: undefined,
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
    const promptWithContent = vi.fn().mockResolvedValue({
      message: {
        role: 'assistant',
        content: {
          parts: [{ type: 'text', text: 'multimodal reply' }],
        },
      },
      finishReason: 'completed',
    });

    const createChatSessionSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          createChatSession: (options: unknown) => {
            registerTool: typeof registerTool;
            prompt: typeof prompt;
            promptWithContent: typeof promptWithContent;
          };
        },
        'createChatSession',
      )
      .mockReturnValue({
        registerTool,
        prompt,
        promptWithContent,
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
    expect((definition as { execution_mode?: string }).execution_mode).toBe(
      'Concurrent',
    );
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

    await expect(
      session.promptContent({
        parts: [
          { type: 'text', text: 'Describe this image.' },
          {
            type: 'image',
            source: { kind: 'url', url: 'https://example.com/cat.png' },
          },
        ],
      }),
    ).resolves.toEqual({
      message: {
        role: 'assistant',
        content: {
          parts: [{ type: 'text', text: 'multimodal reply' }],
        },
      },
      finishReason: 'completed',
    });
    expect(promptWithContent).toHaveBeenCalledWith({
      parts: [
        { type: 'text', text: 'Describe this image.' },
        {
          type: 'image',
          source: { kind: 'url', url: 'https://example.com/cat.png' },
        },
      ],
    });
  });

  it('passes sequential execution_mode to wasm tool definitions', async () => {
    const registerTool = vi.fn();
    const prompt = vi.fn().mockResolvedValue({
      message: { role: 'assistant', content: 'ok' },
      finishReason: 'completed',
    });

    vi.spyOn(
      wasmModule as typeof wasmModule & {
        createChatSession: (options: unknown) => {
          registerTool: typeof registerTool;
          prompt: typeof prompt;
        };
      },
      'createChatSession',
    ).mockReturnValue({ registerTool, prompt });

    vi.stubEnv('OPENAI_API_KEY', 'test-key');

    const session = await createChatSession({});
    session.registerTool(
      {
        name: 'slow_tool',
        description: 'Must not overlap.',
        parameters: [],
        executionMode: 'sequential',
      },
      async () => ({ content: 'done' }),
    );

    const [definition] = registerTool.mock.calls[0] as [
      { execution_mode?: string },
      unknown,
    ];
    expect(definition.execution_mode).toBe('Sequential');
  });
});
