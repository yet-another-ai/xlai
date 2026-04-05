import { MemoryFileSystem } from './filesystem';
import { toWasmFileSystemBridge } from './filesystem/bridge';
import type {
  WasmFileSystemBridge,
  WasmMemoryFileSystemInstance,
} from './filesystem/types';
import type {
  AgentOptions,
  AgentSessionOptions,
  ChatContent,
  ChatOptions,
  ChatResponse,
  ChatSessionOptions,
  ContentPart,
  QtsSessionConfig,
  ToolDefinition,
  ToolParameterType,
  ToolResult,
} from './types';
import { envValue, getWasmModule, initXlai } from './wasm';

export type ResolvedRequestOptions = {
  prompt: string;
  content?: ChatContent;
  apiKey: string;
  systemPrompt?: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  maxOutputTokens?: number;
  /** Streaming agent loop toggle when the WASM layer supports it; unary `agent()` is always one model call. */
  agentLoop?: boolean;
};

export type ResolvedSessionOptions = Omit<ResolvedRequestOptions, 'prompt'> & {
  qts?: QtsSessionConfig;
};

export type WasmChatFunction = (
  options: ResolvedRequestOptions,
) => Promise<ChatResponse>;

export type WasmToolParameterKind =
  | 'String'
  | 'Number'
  | 'Integer'
  | 'Boolean'
  | 'Array'
  | 'Object';

export type WasmToolSessionInstance = {
  registerTool: (
    definition: unknown,
    callback: (argumentsValue: unknown) => unknown,
  ) => void;
  prompt: (content: string) => Promise<ChatResponse>;
  promptWithContent?: (content: ChatContent) => Promise<ChatResponse>;
  /** `AgentSession` (WASM) only. */
  registerContextCompressor?: (
    callback: (messages: unknown, estimatedInputTokens: unknown) => unknown,
  ) => void;
  streamPrompt?: (content: string) => Promise<unknown>;
  streamPromptWithContent?: (content: ChatContent) => Promise<unknown>;
};

export type WasmCreateSessionFunction = (
  options: ResolvedSessionOptions,
) => WasmToolSessionInstance;

export type WasmCreateSessionWithMemoryFileSystemFunction = (
  options: ResolvedSessionOptions,
  fileSystem: WasmMemoryFileSystemInstance,
) => WasmToolSessionInstance;

export type WasmCreateSessionWithFileSystemFunction = (
  options: ResolvedSessionOptions,
  fileSystem: WasmFileSystemBridge,
) => WasmToolSessionInstance;

export async function packageVersion(): Promise<string> {
  await initXlai();
  return getWasmModule().package_version();
}

export function normalizeInlineData<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map((entry) => normalizeInlineData(entry)) as T;
  }

  if (value === null || typeof value !== 'object') {
    return value;
  }

  const objectValue = value as Record<string, unknown>;
  const normalized = Object.fromEntries(
    Object.entries(objectValue).map(([key, entry]) => [
      key,
      normalizeInlineData(entry),
    ]),
  );

  if (normalized.kind === 'inline_data') {
    const data =
      typeof normalized.data === 'string'
        ? normalized.data
        : typeof normalized.data_base64 === 'string'
          ? normalized.data_base64
          : undefined;

    if (data !== undefined) {
      normalized.data = data;
    }

    delete normalized.data_base64;
  }

  if (normalized.type === 'audio_delta') {
    const data =
      typeof normalized.data === 'string'
        ? normalized.data
        : typeof normalized.data_base64 === 'string'
          ? normalized.data_base64
          : undefined;

    if (data !== undefined) {
      normalized.data = data;
    }

    delete normalized.data_base64;
  }

  return normalized as T;
}

function requireApiKey(apiKey?: string): string {
  const resolvedApiKey = apiKey ?? envValue('OPENAI_API_KEY');
  if (resolvedApiKey === undefined || resolvedApiKey.trim() === '') {
    throw new Error('OPENAI_API_KEY must be set or passed explicitly');
  }

  return resolvedApiKey;
}

export function resolveRequestOptions(
  options: ChatOptions,
): ResolvedRequestOptions {
  return {
    prompt: options.prompt ?? '',
    content: normalizeInlineData(options.content),
    apiKey: requireApiKey(options.apiKey),
    systemPrompt: options.systemPrompt,
    baseUrl: options.baseUrl ?? envValue('OPENAI_BASE_URL'),
    model: options.model ?? envValue('OPENAI_MODEL'),
    temperature: options.temperature,
    maxOutputTokens: options.maxOutputTokens,
  };
}

/** Resolves options for the one-shot `agent()` helper (includes `agentLoop` when set). */
export function resolveAgentRequestOptions(
  options: AgentOptions,
): ResolvedRequestOptions {
  const { agentLoop, ...chatLike } = options;
  const resolved = resolveRequestOptions(chatLike);
  return agentLoop === undefined ? resolved : { ...resolved, agentLoop };
}

export function resolveSessionOptions(
  options: Omit<ChatSessionOptions | AgentSessionOptions, 'fileSystem'>,
): ResolvedSessionOptions {
  const base: ResolvedSessionOptions = {
    apiKey: requireApiKey(options.apiKey),
    systemPrompt: options.systemPrompt,
    baseUrl: options.baseUrl ?? envValue('OPENAI_BASE_URL'),
    model: options.model ?? envValue('OPENAI_MODEL'),
    temperature: options.temperature,
    maxOutputTokens: options.maxOutputTokens,
    ...(options.qts !== undefined ? { qts: options.qts } : {}),
  };

  if ('agentLoop' in options && typeof options.agentLoop === 'boolean') {
    return { ...base, agentLoop: options.agentLoop };
  }

  return base;
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
  execution_mode: 'Concurrent' | 'Sequential';
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
    execution_mode:
      definition.executionMode === 'sequential' ? 'Sequential' : 'Concurrent',
  };
}

function toWasmToolResult(
  result: ToolResult,
  toolName: string,
): {
  tool_name: string;
  content: string;
  is_error: boolean;
  metadata: Record<string, unknown>;
} {
  return {
    tool_name: result.toolName ?? toolName,
    content: result.content,
    is_error: result.isError ?? false,
    metadata: result.metadata ?? {},
  };
}

export abstract class ToolSession {
  protected constructor(protected readonly inner: WasmToolSessionInstance) {}

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
    return normalizeInlineData(await this.inner.prompt(content));
  }

  async promptContent(content: ChatContent): Promise<ChatResponse> {
    if (this.inner.promptWithContent !== undefined) {
      return normalizeInlineData(
        await this.inner.promptWithContent(normalizeInlineData(content)),
      );
    }

    if (typeof content === 'string') {
      return normalizeInlineData(await this.inner.prompt(content));
    }

    throw new Error(
      'This xlai WASM build does not expose promptWithContent for multimodal input',
    );
  }

  async promptParts(parts: ContentPart[]): Promise<ChatResponse> {
    return this.promptContent({ parts });
  }
}

export async function createToolSession<TSession>(
  options: ChatSessionOptions | AgentSessionOptions,
  createSession: WasmCreateSessionFunction,
  createSessionWithMemoryFileSystem: WasmCreateSessionWithMemoryFileSystemFunction,
  createSessionWithFileSystem: WasmCreateSessionWithFileSystemFunction,
  fromWasm: (inner: WasmToolSessionInstance) => TSession,
): Promise<TSession> {
  await initXlai();

  const resolvedOptions = resolveSessionOptions(options);
  const inner =
    options.fileSystem === undefined
      ? createSession(resolvedOptions)
      : options.fileSystem instanceof MemoryFileSystem
        ? createSessionWithMemoryFileSystem(
            resolvedOptions,
            options.fileSystem.toWasmFileSystem(),
          )
        : createSessionWithFileSystem(
            resolvedOptions,
            toWasmFileSystemBridge(options.fileSystem),
          );

  return fromWasm(inner);
}
