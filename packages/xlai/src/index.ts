import type { FileSystemApi } from './filesystem';
import { MemoryFileSystem } from './filesystem';
export {
  MemoryFileSystem,
  OpfsFileSystem,
  createMemoryFileSystem,
  createOpfsFileSystem,
} from './filesystem';
import { toWasmFileSystemBridge } from './filesystem/bridge';
import type {
  WasmFileSystemBridge,
  WasmMemoryFileSystemInstance,
} from './filesystem/types';
export type {
  FileSystemApi,
  FileSystemEntry,
  FileSystemEntryKind,
} from './filesystem';
import { envValue, getWasmModule, initXlai } from './wasm';
export { initXlai } from './wasm';

export interface ChatOptions {
  prompt: string;
  systemPrompt?: string;
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  maxOutputTokens?: number;
}

export interface ChatUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export type ChatFinishReason =
  | 'completed'
  | 'tool_calls'
  | 'length'
  | 'stopped';

export interface ChatResponse {
  message: {
    role: 'assistant';
    content: string;
  };
  finishReason: ChatFinishReason;
  usage?: ChatUsage;
}

export type ToolParameterType =
  | 'string'
  | 'number'
  | 'integer'
  | 'boolean'
  | 'array'
  | 'object';

export interface ToolParameter {
  name: string;
  description: string;
  kind: ToolParameterType;
  required: boolean;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: ToolParameter[];
}

export interface ToolResult {
  toolName?: string;
  content: string;
  isError?: boolean;
  metadata?: Record<string, string>;
}

export interface ChatSessionOptions {
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  systemPrompt?: string;
  temperature?: number;
  maxOutputTokens?: number;
  fileSystem?: FileSystemApi;
}

type WasmChatFunction = (options: {
  prompt: string;
  apiKey: string;
  systemPrompt?: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  maxOutputTokens?: number;
}) => Promise<ChatResponse>;

type WasmToolParameterKind =
  | 'String'
  | 'Number'
  | 'Integer'
  | 'Boolean'
  | 'Array'
  | 'Object';

type WasmChatSessionInstance = {
  registerTool: (
    definition: unknown,
    callback: (argumentsValue: unknown) => unknown,
  ) => void;
  prompt: (content: string) => Promise<ChatResponse>;
};

type WasmCreateChatSessionFunction = (options: {
  apiKey: string;
  systemPrompt?: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  maxOutputTokens?: number;
}) => WasmChatSessionInstance;

type WasmCreateChatSessionWithMemoryFileSystemFunction = (
  options: {
    apiKey: string;
    systemPrompt?: string;
    baseUrl?: string;
    model?: string;
    temperature?: number;
    maxOutputTokens?: number;
  },
  fileSystem: WasmMemoryFileSystemInstance,
) => WasmChatSessionInstance;

type WasmCreateChatSessionWithFileSystemFunction = (
  options: {
    apiKey: string;
    systemPrompt?: string;
    baseUrl?: string;
    model?: string;
    temperature?: number;
    maxOutputTokens?: number;
  },
  fileSystem: WasmFileSystemBridge,
) => WasmChatSessionInstance;

function wasmChatFunction(): WasmChatFunction {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      chat: WasmChatFunction;
    }
  ).chat;
}

function wasmCreateChatSessionFunction(): WasmCreateChatSessionFunction {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      createChatSession: WasmCreateChatSessionFunction;
    }
  ).createChatSession;
}

function wasmCreateChatSessionWithMemoryFileSystemFunction(): WasmCreateChatSessionWithMemoryFileSystemFunction {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      createChatSessionWithMemoryFileSystem: WasmCreateChatSessionWithMemoryFileSystemFunction;
    }
  ).createChatSessionWithMemoryFileSystem;
}

function wasmCreateChatSessionWithFileSystemFunction(): WasmCreateChatSessionWithFileSystemFunction {
  return (
    getWasmModule() as ReturnType<typeof getWasmModule> & {
      createChatSessionWithFileSystem: WasmCreateChatSessionWithFileSystemFunction;
    }
  ).createChatSessionWithFileSystem;
}

export async function packageVersion(): Promise<string> {
  await initXlai();
  return getWasmModule().package_version();
}

function requireApiKey(apiKey?: string): string {
  const resolvedApiKey = apiKey ?? envValue('OPENAI_API_KEY');
  if (resolvedApiKey === undefined || resolvedApiKey.trim() === '') {
    throw new Error('OPENAI_API_KEY must be set or passed explicitly');
  }

  return resolvedApiKey;
}

function resolveChatSessionOptions(
  options: Omit<ChatSessionOptions, 'fileSystem'>,
): {
  apiKey: string;
  systemPrompt?: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  maxOutputTokens?: number;
} {
  return {
    apiKey: requireApiKey(options.apiKey),
    systemPrompt: options.systemPrompt,
    baseUrl: options.baseUrl ?? envValue('OPENAI_BASE_URL'),
    model: options.model ?? envValue('OPENAI_MODEL'),
    temperature: options.temperature,
    maxOutputTokens: options.maxOutputTokens,
  };
}

function toolParameterKindToWasm(
  kind: ToolParameterType,
): WasmToolParameterKind {
  switch (kind) {
    case 'string':
      return 'String';
    case 'number':
      return 'Number';
    case 'integer':
      return 'Integer';
    case 'boolean':
      return 'Boolean';
    case 'array':
      return 'Array';
    case 'object':
      return 'Object';
  }
}

function toWasmToolDefinition(definition: ToolDefinition): {
  name: string;
  description: string;
  parameters: Array<{
    name: string;
    description: string;
    kind: WasmToolParameterKind;
    required: boolean;
  }>;
} {
  return {
    name: definition.name,
    description: definition.description,
    parameters: definition.parameters.map((parameter) => ({
      name: parameter.name,
      description: parameter.description,
      kind: toolParameterKindToWasm(parameter.kind),
      required: parameter.required,
    })),
  };
}

function toWasmToolResult(
  result: ToolResult,
  toolName: string,
): {
  tool_name: string;
  content: string;
  is_error: boolean;
  metadata: Record<string, string>;
} {
  return {
    tool_name: result.toolName ?? toolName,
    content: result.content,
    is_error: result.isError ?? false,
    metadata: result.metadata ?? {},
  };
}

export class ChatSession {
  private constructor(private readonly inner: WasmChatSessionInstance) {}

  static fromWasm(inner: WasmChatSessionInstance): ChatSession {
    return new ChatSession(inner);
  }

  registerTool(
    definition: ToolDefinition,
    callback: (argumentsValue: unknown) => ToolResult | Promise<ToolResult>,
  ): void {
    this.inner.registerTool(
      toWasmToolDefinition(definition),
      async (argumentsValue: unknown) =>
        toWasmToolResult(await callback(argumentsValue), definition.name),
    );
  }

  async prompt(content: string): Promise<ChatResponse> {
    return this.inner.prompt(content);
  }
}

export async function createChatSession(
  options: ChatSessionOptions = {},
): Promise<ChatSession> {
  await initXlai();

  const resolvedOptions = resolveChatSessionOptions(options);
  const inner =
    options.fileSystem === undefined
      ? wasmCreateChatSessionFunction()(resolvedOptions)
      : options.fileSystem instanceof MemoryFileSystem
        ? wasmCreateChatSessionWithMemoryFileSystemFunction()(
            resolvedOptions,
            options.fileSystem.toWasmFileSystem(),
          )
        : wasmCreateChatSessionWithFileSystemFunction()(
            resolvedOptions,
            toWasmFileSystemBridge(options.fileSystem),
          );

  return ChatSession.fromWasm(inner);
}

export async function chat(options: ChatOptions): Promise<ChatResponse> {
  await initXlai();

  return wasmChatFunction()({
    prompt: options.prompt,
    apiKey: requireApiKey(options.apiKey),
    systemPrompt: options.systemPrompt,
    baseUrl: options.baseUrl ?? envValue('OPENAI_BASE_URL'),
    model: options.model ?? envValue('OPENAI_MODEL'),
    temperature: options.temperature,
    maxOutputTokens: options.maxOutputTokens,
  });
}
