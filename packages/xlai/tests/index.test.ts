import { afterEach, describe, expect, it, vi } from 'vitest';

import * as wasmModule from '../pkg/xlai_wasm.js';
import {
  chat,
  createChatSession,
  createMemoryFileSystem,
  initXlai,
  packageVersion,
} from '../src/index';

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

  it('provides the memory filesystem interface through wasm', async () => {
    const fileSystem = await createMemoryFileSystem();

    await expect(
      fileSystem.createDirAll('/docs/guides'),
    ).resolves.toBeUndefined();
    await expect(
      fileSystem.write(
        '/docs/readme.txt',
        new TextEncoder().encode('hello wasm fs'),
      ),
    ).resolves.toBeUndefined();

    await expect(fileSystem.exists('/docs/readme.txt')).resolves.toBe(true);
    await expect(fileSystem.read('/docs/readme.txt')).resolves.toEqual(
      new TextEncoder().encode('hello wasm fs'),
    );

    await expect(fileSystem.list('/docs')).resolves.toEqual([
      {
        path: '/docs/guides',
        kind: 'directory',
      },
      {
        path: '/docs/readme.txt',
        kind: 'file',
      },
    ]);

    await expect(
      fileSystem.deletePath('/docs/readme.txt'),
    ).resolves.toBeUndefined();
    await expect(fileSystem.exists('/docs/readme.txt')).resolves.toBe(false);
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

  it('bridges generic filesystem implementations into the wasm chat session API', async () => {
    const registerTool = vi.fn();
    const prompt = vi.fn().mockResolvedValue({
      message: {
        role: 'assistant',
        content: 'generic filesystem reply',
      },
      finishReason: 'completed',
    });

    const createChatSessionWithFileSystemSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          createChatSessionWithFileSystem: (
            options: unknown,
            fileSystem: {
              read: (path: string) => Promise<Uint8Array>;
              exists: (path: string) => Promise<boolean>;
              write: (path: string, data: Uint8Array) => Promise<void>;
              createDirAll: (path: string) => Promise<void>;
              list: (
                path: string,
              ) => Promise<Array<{ path: string; kind: string }>>;
              deletePath: (path: string) => Promise<void>;
            },
          ) => {
            registerTool: typeof registerTool;
            prompt: typeof prompt;
          };
        },
        'createChatSessionWithFileSystem',
      )
      .mockReturnValue({
        registerTool,
        prompt,
      });

    const fileSystem = {
      read: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
      exists: vi.fn().mockResolvedValue(true),
      write: vi.fn().mockResolvedValue(undefined),
      createDirAll: vi.fn().mockResolvedValue(undefined),
      list: vi.fn().mockResolvedValue([
        {
          path: '/docs/readme.txt',
          kind: 'file',
        },
      ]),
      deletePath: vi.fn().mockResolvedValue(undefined),
    };

    const session = await createChatSession({
      apiKey: 'generic-key',
      fileSystem,
    });

    expect(createChatSessionWithFileSystemSpy).toHaveBeenCalledTimes(1);
    const [, bridgedFileSystem] = createChatSessionWithFileSystemSpy.mock
      .calls[0] as [
      unknown,
      {
        read: (path: string) => Promise<Uint8Array>;
        exists: (path: string) => Promise<boolean>;
        write: (path: string, data: Uint8Array) => Promise<void>;
        createDirAll: (path: string) => Promise<void>;
        list: (path: string) => Promise<Array<{ path: string; kind: string }>>;
        deletePath: (path: string) => Promise<void>;
      },
    ];

    await expect(bridgedFileSystem.read('/docs/readme.txt')).resolves.toEqual(
      new Uint8Array([1, 2, 3]),
    );
    await expect(bridgedFileSystem.exists('/docs/readme.txt')).resolves.toBe(
      true,
    );
    await expect(
      bridgedFileSystem.write('/docs/readme.txt', new Uint8Array([4, 5])),
    ).resolves.toBeUndefined();
    await expect(
      bridgedFileSystem.createDirAll('/docs'),
    ).resolves.toBeUndefined();
    await expect(bridgedFileSystem.list('/docs')).resolves.toEqual([
      {
        path: '/docs/readme.txt',
        kind: 'file',
      },
    ]);
    await expect(
      bridgedFileSystem.deletePath('/docs/readme.txt'),
    ).resolves.toBeUndefined();

    expect(fileSystem.read).toHaveBeenCalledWith('/docs/readme.txt');
    expect(fileSystem.exists).toHaveBeenCalledWith('/docs/readme.txt');
    expect(fileSystem.write).toHaveBeenCalledWith(
      '/docs/readme.txt',
      new Uint8Array([4, 5]),
    );
    expect(fileSystem.createDirAll).toHaveBeenCalledWith('/docs');
    expect(fileSystem.list).toHaveBeenCalledWith('/docs');
    expect(fileSystem.deletePath).toHaveBeenCalledWith('/docs/readme.txt');

    await expect(
      session.prompt('Check the generic filesystem.'),
    ).resolves.toEqual({
      message: {
        role: 'assistant',
        content: 'generic filesystem reply',
      },
      finishReason: 'completed',
    });
  });
});
