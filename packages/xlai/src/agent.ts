import type {
  AgentContextCompressor,
  AgentOptions,
  AgentSessionOptions,
  ChatContent,
  ChatExecutionEvent,
  ChatMessage,
  ChatResponse,
} from './types';
import {
  ToolSession,
  createToolSession,
  normalizeInlineData,
  resolveAgentRequestOptions,
  type WasmChatFunction,
  type WasmCreateSessionFunction,
  type WasmCreateSessionWithFileSystemFunction,
  type WasmCreateSessionWithMemoryFileSystemFunction,
  type WasmToolSessionInstance,
} from './shared';
import { getWasmModule, initXlai } from './wasm';

function wasmAgentFunction(): WasmChatFunction {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      agent: WasmChatFunction;
    }
  ).agent;
}

function wasmCreateAgentSessionFunction(): WasmCreateSessionFunction {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      createAgentSession: WasmCreateSessionFunction;
    }
  ).createAgentSession;
}

function wasmCreateAgentSessionWithMemoryFileSystemFunction(): WasmCreateSessionWithMemoryFileSystemFunction {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      createAgentSessionWithMemoryFileSystem: WasmCreateSessionWithMemoryFileSystemFunction;
    }
  ).createAgentSessionWithMemoryFileSystem;
}

function wasmCreateAgentSessionWithFileSystemFunction(): WasmCreateSessionWithFileSystemFunction {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      createAgentSessionWithFileSystem: WasmCreateSessionWithFileSystemFunction;
    }
  ).createAgentSessionWithFileSystem;
}

export class AgentSession extends ToolSession {
  private constructor(inner: WasmToolSessionInstance) {
    super(inner);
  }

  static fromWasm(inner: WasmToolSessionInstance): AgentSession {
    return new AgentSession(inner);
  }

  /**
   * Registers a JS async callback run before each **streamed** agent-loop model call
   * (`streamPrompt` / `streamPromptWithContent`). Receives full message history and a best-effort
   * input token count (`null` when the runtime could not estimate).
   */
  registerContextCompressor(callback: AgentContextCompressor): void {
    if (this.inner.registerContextCompressor === undefined) {
      throw new Error(
        'registerContextCompressor is not available in this xlai WASM build',
      );
    }
    this.inner.registerContextCompressor(
      async (messages: unknown, estimatedInputTokens: unknown) => {
        let est: number | null = null;
        if (
          estimatedInputTokens !== null &&
          estimatedInputTokens !== undefined
        ) {
          const n = Number(estimatedInputTokens);
          est = Number.isFinite(n) ? n : null;
        }
        return callback(normalizeInlineData(messages) as ChatMessage[], est);
      },
    );
  }

  /** Agent streaming loop; collects all execution events (model chunks, tool call/result). */
  async streamPrompt(content: string): Promise<ChatExecutionEvent[]> {
    if (this.inner.streamPrompt === undefined) {
      throw new Error('streamPrompt is not available in this xlai WASM build');
    }
    return normalizeInlineData(
      await this.inner.streamPrompt(content),
    ) as ChatExecutionEvent[];
  }

  async streamPromptWithContent(
    content: ChatContent,
  ): Promise<ChatExecutionEvent[]> {
    if (this.inner.streamPromptWithContent === undefined) {
      throw new Error(
        'streamPromptWithContent is not available in this xlai WASM build',
      );
    }
    return normalizeInlineData(
      await this.inner.streamPromptWithContent(normalizeInlineData(content)),
    ) as ChatExecutionEvent[];
  }
}

export async function createAgentSession(
  options: AgentSessionOptions = {},
): Promise<AgentSession> {
  return createToolSession(
    options,
    wasmCreateAgentSessionFunction(),
    wasmCreateAgentSessionWithMemoryFileSystemFunction(),
    wasmCreateAgentSessionWithFileSystemFunction(),
    AgentSession.fromWasm,
  );
}

export async function agent(options: AgentOptions): Promise<ChatResponse> {
  await initXlai();
  return normalizeInlineData(
    await wasmAgentFunction()(resolveAgentRequestOptions(options)),
  );
}
