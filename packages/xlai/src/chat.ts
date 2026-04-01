import type { ChatOptions, ChatResponse, ChatSessionOptions } from './types';
import {
  ToolSession,
  createToolSession,
  resolveRequestOptions,
  type WasmChatFunction,
  type WasmCreateSessionFunction,
  type WasmCreateSessionWithFileSystemFunction,
  type WasmCreateSessionWithMemoryFileSystemFunction,
  type WasmToolSessionInstance,
} from './shared';
import { getWasmModule, initXlai } from './wasm';

function wasmChatFunction(): WasmChatFunction {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      chat: WasmChatFunction;
    }
  ).chat;
}

function wasmCreateChatSessionFunction(): WasmCreateSessionFunction {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      createChatSession: WasmCreateSessionFunction;
    }
  ).createChatSession;
}

function wasmCreateChatSessionWithMemoryFileSystemFunction(): WasmCreateSessionWithMemoryFileSystemFunction {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      createChatSessionWithMemoryFileSystem: WasmCreateSessionWithMemoryFileSystemFunction;
    }
  ).createChatSessionWithMemoryFileSystem;
}

function wasmCreateChatSessionWithFileSystemFunction(): WasmCreateSessionWithFileSystemFunction {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      createChatSessionWithFileSystem: WasmCreateSessionWithFileSystemFunction;
    }
  ).createChatSessionWithFileSystem;
}

export class ChatSession extends ToolSession {
  private constructor(inner: WasmToolSessionInstance) {
    super(inner);
  }

  static fromWasm(inner: WasmToolSessionInstance): ChatSession {
    return new ChatSession(inner);
  }
}

export async function createChatSession(
  options: ChatSessionOptions = {},
): Promise<ChatSession> {
  return createToolSession(
    options,
    wasmCreateChatSessionFunction(),
    wasmCreateChatSessionWithMemoryFileSystemFunction(),
    wasmCreateChatSessionWithFileSystemFunction(),
    ChatSession.fromWasm,
  );
}

export async function chat(options: ChatOptions): Promise<ChatResponse> {
  await initXlai();
  return wasmChatFunction()(resolveRequestOptions(options));
}
