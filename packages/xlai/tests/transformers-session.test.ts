import { afterEach, describe, expect, it, vi } from 'vitest';

const mockState = vi.hoisted(() => {
  const chatInner = {
    registerTool: vi.fn(),
    prompt: vi.fn(async () => ({
      message: { role: 'assistant' as const, content: 'chat reply' },
      finishReason: 'completed' as const,
    })),
  };

  const agentInner = {
    registerTool: vi.fn(),
    prompt: vi.fn(async () => ({
      message: { role: 'assistant' as const, content: 'agent reply' },
      finishReason: 'completed' as const,
    })),
  };

  const createTransformersChatSession = vi.fn(() => chatInner);
  const createTransformersChatSessionWithMemoryFileSystem = vi.fn(
    () => chatInner,
  );
  const createTransformersChatSessionWithFileSystem = vi.fn(() => chatInner);
  const createTransformersAgentSession = vi.fn(() => agentInner);
  const createTransformersAgentSessionWithMemoryFileSystem = vi.fn(
    () => agentInner,
  );
  const createTransformersAgentSessionWithFileSystem = vi.fn(() => agentInner);

  class MockMemoryFileSystem {
    async read(): Promise<Uint8Array> {
      return new Uint8Array();
    }
    async exists(): Promise<boolean> {
      return false;
    }
    async write(): Promise<void> {}
    async createDirAll(): Promise<void> {}
    async list(): Promise<Array<{ path: string; kind: 'file' | 'directory' }>> {
      return [];
    }
    async deletePath(): Promise<void> {}
  }

  return {
    chatInner,
    agentInner,
    createTransformersChatSession,
    createTransformersChatSessionWithMemoryFileSystem,
    createTransformersChatSessionWithFileSystem,
    createTransformersAgentSession,
    createTransformersAgentSessionWithMemoryFileSystem,
    createTransformersAgentSessionWithFileSystem,
    adapter: { generate: vi.fn() },
    initXlai: vi.fn(async () => undefined),
    MockMemoryFileSystem,
  };
});

vi.mock('../src/wasm', () => ({
  initXlai: mockState.initXlai,
  getWasmModule: () => ({
    createTransformersChatSession: mockState.createTransformersChatSession,
    createTransformersChatSessionWithMemoryFileSystem:
      mockState.createTransformersChatSessionWithMemoryFileSystem,
    createTransformersChatSessionWithFileSystem:
      mockState.createTransformersChatSessionWithFileSystem,
    createTransformersAgentSession: mockState.createTransformersAgentSession,
    createTransformersAgentSessionWithMemoryFileSystem:
      mockState.createTransformersAgentSessionWithMemoryFileSystem,
    createTransformersAgentSessionWithFileSystem:
      mockState.createTransformersAgentSessionWithFileSystem,
    MemoryFileSystem: mockState.MockMemoryFileSystem,
  }),
}));

vi.mock('../src/transformers/adapter', () => ({
  createXlaiTransformersJsAdapter: vi.fn(async () => mockState.adapter),
}));

import { createMemoryFileSystem } from '../src/filesystem';
import {
  createTransformersAgentSession,
  createTransformersChatSession,
} from '../src/transformers';
import type { FileSystemApi } from '../src/filesystem';

describe('transformers sessions', () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it('creates a chat session through the dedicated wasm constructor', async () => {
    const session = await createTransformersChatSession({
      modelId: 'hf/model',
      systemPrompt: 'Be helpful.',
      temperature: 0.2,
      maxOutputTokens: 42,
    });

    expect(mockState.initXlai).toHaveBeenCalled();
    expect(mockState.createTransformersChatSession).toHaveBeenCalledWith({
      modelId: 'hf/model',
      adapter: mockState.adapter,
      systemPrompt: 'Be helpful.',
      temperature: 0.2,
      maxOutputTokens: 42,
    });
    await expect(session.prompt('hello')).resolves.toEqual({
      message: { role: 'assistant', content: 'chat reply' },
      finishReason: 'completed',
    });
  });

  it('routes generic file systems through the JS bridge constructor', async () => {
    const fileSystem: FileSystemApi = {
      read: vi.fn(async () => new Uint8Array([1])),
      exists: vi.fn(async () => true),
      write: vi.fn(async () => undefined),
      createDirAll: vi.fn(async () => undefined),
      list: vi.fn(async () => []),
      deletePath: vi.fn(async () => undefined),
    };

    await createTransformersChatSession({
      modelId: 'hf/model',
      fileSystem,
    });

    expect(
      mockState.createTransformersChatSessionWithFileSystem,
    ).toHaveBeenCalledTimes(1);
    const [, bridge] =
      mockState.createTransformersChatSessionWithFileSystem.mock.calls[0] ?? [];
    expect(bridge).toMatchObject({
      read: expect.any(Function),
      exists: expect.any(Function),
      write: expect.any(Function),
      createDirAll: expect.any(Function),
      list: expect.any(Function),
      deletePath: expect.any(Function),
    });
  });

  it('routes memory file systems through the dedicated memory constructor for agents', async () => {
    const fileSystem = await createMemoryFileSystem();

    const session = await createTransformersAgentSession({
      modelId: 'hf/agent-model',
      fileSystem,
    });

    expect(
      mockState.createTransformersAgentSessionWithMemoryFileSystem,
    ).toHaveBeenCalledTimes(1);
    await expect(session.prompt('plan')).resolves.toEqual({
      message: { role: 'assistant', content: 'agent reply' },
      finishReason: 'completed',
    });
  });
});
