import { afterEach, describe, expect, it, vi } from 'vitest';

import * as wasmModule from '../pkg/xlai_wasm.js';
import { AgentSession, agent, createAgentSession } from '../src/index';

describe('xlai agent api', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  it('delegates agent requests to the wasm API', async () => {
    const agentSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          agent: (options: unknown) => Promise<unknown>;
        },
        'agent',
      )
      .mockResolvedValue({
        message: {
          role: 'assistant',
          content: 'xlai agent reply',
        },
        finishReason: 'completed',
      });

    vi.stubEnv('OPENAI_API_KEY', 'agent-key');
    vi.stubEnv('OPENAI_BASE_URL', 'https://agents.example.com/v1/');
    vi.stubEnv('OPENAI_MODEL', 'agent-model');

    await expect(
      agent({
        prompt: 'Help me plan this task.',
        systemPrompt: 'Use tools when helpful.',
        temperature: 0.3,
        maxOutputTokens: 96,
      }),
    ).resolves.toEqual({
      message: {
        role: 'assistant',
        content: 'xlai agent reply',
      },
      finishReason: 'completed',
    });

    expect(agentSpy).toHaveBeenCalledWith({
      prompt: 'Help me plan this task.',
      apiKey: 'agent-key',
      baseUrl: 'https://agents.example.com/v1/',
      model: 'agent-model',
      systemPrompt: 'Use tools when helpful.',
      temperature: 0.3,
      maxOutputTokens: 96,
    });
  });

  it('delegates multimodal agent requests to the wasm API', async () => {
    const agentSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          agent: (options: unknown) => Promise<unknown>;
        },
        'agent',
      )
      .mockResolvedValue({
        message: {
          role: 'assistant',
          content: {
            parts: [{ type: 'text', text: 'transcription ready' }],
          },
        },
        finishReason: 'completed',
      });

    vi.stubEnv('OPENAI_API_KEY', 'agent-key');

    await expect(
      agent({
        content: {
          parts: [
            { type: 'text', text: 'Transcribe this.' },
            {
              type: 'audio',
              source: {
                kind: 'inline_data',
                mime_type: 'audio/mpeg',
                data_base64: 'SUQzBA==',
              },
              mime_type: 'audio/mpeg',
            },
          ],
        },
      }),
    ).resolves.toEqual({
      message: {
        role: 'assistant',
        content: {
          parts: [{ type: 'text', text: 'transcription ready' }],
        },
      },
      finishReason: 'completed',
    });

    expect(agentSpy).toHaveBeenCalledWith({
      prompt: '',
      content: {
        parts: [
          { type: 'text', text: 'Transcribe this.' },
          {
            type: 'audio',
            source: {
              kind: 'inline_data',
              mime_type: 'audio/mpeg',
              data: 'SUQzBA==',
            },
            mime_type: 'audio/mpeg',
          },
        ],
      },
      apiKey: 'agent-key',
      baseUrl: undefined,
      model: undefined,
      systemPrompt: undefined,
      temperature: undefined,
      maxOutputTokens: undefined,
    });
  });

  it('passes agentLoop false to the wasm agent API', async () => {
    const agentSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          agent: (options: unknown) => Promise<unknown>;
        },
        'agent',
      )
      .mockResolvedValue({
        message: { role: 'assistant', content: 'ok' },
        finishReason: 'completed',
      });

    vi.stubEnv('OPENAI_API_KEY', 'k');

    await agent({ prompt: 'x', agentLoop: false });

    expect(agentSpy).toHaveBeenCalledWith(
      expect.objectContaining({ agentLoop: false }),
    );
  });

  it('creates agent sessions and normalizes registered tools', async () => {
    const registerTool = vi.fn();
    const prompt = vi.fn().mockResolvedValue({
      message: {
        role: 'assistant',
        content: 'agent session reply',
      },
      finishReason: 'completed',
    });
    const promptWithContent = vi.fn().mockResolvedValue({
      message: {
        role: 'assistant',
        content: {
          parts: [{ type: 'text', text: 'agent multimodal reply' }],
        },
      },
      finishReason: 'completed',
    });

    const createAgentSessionSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          createAgentSession: (options: unknown) => {
            registerTool: typeof registerTool;
            prompt: typeof prompt;
            promptWithContent: typeof promptWithContent;
          };
        },
        'createAgentSession',
      )
      .mockReturnValue({
        registerTool,
        prompt,
        promptWithContent,
      });

    vi.stubEnv('OPENAI_API_KEY', 'test-key');
    vi.stubEnv('OPENAI_BASE_URL', 'https://example.com/v1/');
    vi.stubEnv('OPENAI_MODEL', 'test-model');

    const session = await createAgentSession({
      systemPrompt: 'Use tools and skills when needed.',
      temperature: 0.15,
      maxOutputTokens: 120,
    });

    expect(createAgentSessionSpy).toHaveBeenCalledWith({
      apiKey: 'test-key',
      baseUrl: 'https://example.com/v1/',
      model: 'test-model',
      systemPrompt: 'Use tools and skills when needed.',
      temperature: 0.15,
      maxOutputTokens: 120,
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
        metadata: {
          source: 'agent-test',
        },
      }),
    );

    expect(registerTool).toHaveBeenCalledTimes(1);
    const [definition, callback] = registerTool.mock.calls[0] as [
      {
        input_schema: {
          type: string;
          properties?: Record<string, { type: string }>;
          required?: string[];
        };
      },
      (argumentsValue: unknown) => Promise<unknown>,
    ];
    expect(definition.input_schema.type).toBe('object');
    expect(definition.input_schema.properties?.city?.type).toBe('string');
    expect(definition.input_schema.required).toEqual(['city']);
    await expect(callback({ city: 'Paris' })).resolves.toEqual({
      tool_name: 'lookup_weather',
      content: 'weather for Paris: sunny',
      is_error: false,
      metadata: {
        source: 'agent-test',
      },
    });

    await expect(
      session.prompt('What is the weather in Paris?'),
    ).resolves.toEqual({
      message: {
        role: 'assistant',
        content: 'agent session reply',
      },
      finishReason: 'completed',
    });
    expect(prompt).toHaveBeenCalledWith('What is the weather in Paris?');

    await expect(
      session.promptParts([
        { type: 'text', text: 'Listen to this clip.' },
        {
          type: 'audio',
          source: { kind: 'url', url: 'https://example.com/clip.mp3' },
        },
      ]),
    ).resolves.toEqual({
      message: {
        role: 'assistant',
        content: {
          parts: [{ type: 'text', text: 'agent multimodal reply' }],
        },
      },
      finishReason: 'completed',
    });
    expect(promptWithContent).toHaveBeenCalledWith({
      parts: [
        { type: 'text', text: 'Listen to this clip.' },
        {
          type: 'audio',
          source: { kind: 'url', url: 'https://example.com/clip.mp3' },
        },
      ],
    });
  });

  it('passes agentLoop to createAgentSession wasm options', async () => {
    const createAgentSessionSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          createAgentSession: (options: unknown) => {
            registerTool: () => void;
            prompt: () => Promise<unknown>;
            promptWithContent?: () => Promise<unknown>;
          };
        },
        'createAgentSession',
      )
      .mockReturnValue({
        registerTool: vi.fn(),
        prompt: vi.fn(),
        promptWithContent: vi.fn(),
      });

    vi.stubEnv('OPENAI_API_KEY', 'test-key');

    await createAgentSession({ agentLoop: false });

    expect(createAgentSessionSpy).toHaveBeenCalledWith(
      expect.objectContaining({ agentLoop: false }),
    );
  });

  it('wires registerSystemReminder on AgentSession', async () => {
    const registerSystemReminder = vi.fn();
    const prompt = vi.fn().mockResolvedValue({
      message: { role: 'assistant', content: 'ok' },
      finishReason: 'completed',
    });

    vi.spyOn(
      wasmModule as typeof wasmModule & {
        createAgentSession: (options: unknown) => {
          registerTool: () => void;
          prompt: typeof prompt;
          registerSystemReminder: typeof registerSystemReminder;
        };
      },
      'createAgentSession',
    ).mockReturnValue({
      registerTool: vi.fn(),
      prompt,
      registerSystemReminder,
    });

    vi.stubEnv('OPENAI_API_KEY', 'test-key');

    const session = await createAgentSession({});
    session.registerSystemReminder(async () => 'custom reminder');

    expect(registerSystemReminder).toHaveBeenCalledTimes(1);
    const [[cb]] = registerSystemReminder.mock.calls as [
      (messages: unknown) => Promise<unknown>,
    ][];
    await expect(cb([{ role: 'user', content: 'hi' }])).resolves.toBe(
      'custom reminder',
    );
  });

  it('wires registerContextCompressor and streamPrompt on AgentSession', async () => {
    const registerContextCompressor = vi.fn();
    const streamPrompt = vi.fn().mockResolvedValue([
      {
        kind: 'model',
        data: { MessageStart: { role: 'assistant', message_index: 0 } },
      },
    ]);

    vi.spyOn(
      wasmModule as typeof wasmModule & {
        createAgentSession: (options: unknown) => {
          registerTool: () => void;
          prompt: () => Promise<unknown>;
          registerContextCompressor: typeof registerContextCompressor;
          streamPrompt: typeof streamPrompt;
        };
      },
      'createAgentSession',
    ).mockReturnValue({
      registerTool: vi.fn(),
      prompt: vi.fn(),
      registerContextCompressor,
      streamPrompt,
    });

    vi.stubEnv('OPENAI_API_KEY', 'test-key');

    const session = await createAgentSession({});
    expect(session).toBeInstanceOf(AgentSession);

    session.registerContextCompressor(async (messages) => messages);
    expect(registerContextCompressor).toHaveBeenCalledTimes(1);

    await expect(session.streamPrompt('hello')).resolves.toEqual([
      {
        kind: 'model',
        data: { MessageStart: { role: 'assistant', message_index: 0 } },
      },
    ]);
    expect(streamPrompt).toHaveBeenCalledWith('hello');
  });
});
