import { afterEach, describe, expect, it, vi } from 'vitest';

import * as wasmModule from '../pkg/xlai_wasm.js';
import {
  createAgentSession,
  createChatSession,
  createMemoryFileSystem,
} from '../src/index';

describe('xlai filesystem integration', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
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

  it('bridges generic filesystem implementations into the wasm agent session API', async () => {
    const registerTool = vi.fn();
    const prompt = vi.fn().mockResolvedValue({
      message: {
        role: 'assistant',
        content: 'generic agent filesystem reply',
      },
      finishReason: 'completed',
    });

    const createAgentSessionWithFileSystemSpy = vi
      .spyOn(
        wasmModule as typeof wasmModule & {
          createAgentSessionWithFileSystem: (
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
        'createAgentSessionWithFileSystem',
      )
      .mockReturnValue({
        registerTool,
        prompt,
      });

    const fileSystem = {
      read: vi.fn().mockResolvedValue(new Uint8Array([7, 8, 9])),
      exists: vi.fn().mockResolvedValue(true),
      write: vi.fn().mockResolvedValue(undefined),
      createDirAll: vi.fn().mockResolvedValue(undefined),
      list: vi.fn().mockResolvedValue([
        {
          path: '/agents/context.txt',
          kind: 'file',
        },
      ]),
      deletePath: vi.fn().mockResolvedValue(undefined),
    };

    const session = await createAgentSession({
      apiKey: 'generic-agent-key',
      fileSystem,
    });

    expect(createAgentSessionWithFileSystemSpy).toHaveBeenCalledTimes(1);
    const [, bridgedFileSystem] = createAgentSessionWithFileSystemSpy.mock
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

    await expect(
      bridgedFileSystem.read('/agents/context.txt'),
    ).resolves.toEqual(new Uint8Array([7, 8, 9]));
    await expect(bridgedFileSystem.exists('/agents/context.txt')).resolves.toBe(
      true,
    );
    await expect(
      bridgedFileSystem.write('/agents/context.txt', new Uint8Array([4, 5])),
    ).resolves.toBeUndefined();
    await expect(
      bridgedFileSystem.createDirAll('/agents'),
    ).resolves.toBeUndefined();
    await expect(bridgedFileSystem.list('/agents')).resolves.toEqual([
      {
        path: '/agents/context.txt',
        kind: 'file',
      },
    ]);
    await expect(
      bridgedFileSystem.deletePath('/agents/context.txt'),
    ).resolves.toBeUndefined();

    expect(fileSystem.read).toHaveBeenCalledWith('/agents/context.txt');
    expect(fileSystem.exists).toHaveBeenCalledWith('/agents/context.txt');
    expect(fileSystem.write).toHaveBeenCalledWith(
      '/agents/context.txt',
      new Uint8Array([4, 5]),
    );
    expect(fileSystem.createDirAll).toHaveBeenCalledWith('/agents');
    expect(fileSystem.list).toHaveBeenCalledWith('/agents');
    expect(fileSystem.deletePath).toHaveBeenCalledWith('/agents/context.txt');

    await expect(
      session.prompt('Check the agent filesystem.'),
    ).resolves.toEqual({
      message: {
        role: 'assistant',
        content: 'generic agent filesystem reply',
      },
      finishReason: 'completed',
    });
  });
});
