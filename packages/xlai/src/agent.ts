import type { AgentOptions, AgentSessionOptions, ChatResponse } from './types';
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
