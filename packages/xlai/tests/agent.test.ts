import { afterEach, describe, expect, it, vi } from 'vitest';

import * as wasmModule from '../pkg/xlai_wasm.js';
import { agent, createAgentSession } from '../src/index';

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

  it('creates agent sessions and normalizes registered tools', async () => {
    const registerTool = vi.fn();
    const prompt = vi.fn().mockResolvedValue({
      message: {
        role: 'assistant',
        content: 'agent session reply',
      },
      finishReason: 'completed',
    });

    const createAgentSessionSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          createAgentSession: (options: unknown) => {
            registerTool: typeof registerTool;
            prompt: typeof prompt;
          };
        },
        'createAgentSession',
      )
      .mockReturnValue({
        registerTool,
        prompt,
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
  });
});
