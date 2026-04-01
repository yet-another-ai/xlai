import initWasm, * as wasmModule from '../pkg/xlai_wasm.js';

let initPromise: Promise<void> | null = null;
const wasmUrl = new URL('../pkg/xlai_wasm_bg.wasm', import.meta.url);

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

export type ChatFinishReason = 'completed' | 'tool_calls' | 'length' | 'stopped';

export interface ChatResponse {
  message: {
    role: 'assistant';
    content: string;
  };
  finishReason: ChatFinishReason;
  usage?: ChatUsage;
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

function isSsrRuntime(): boolean {
  return (
    (
      import.meta as ImportMeta & {
        env?: { SSR?: boolean };
      }
    ).env?.SSR ?? typeof window === 'undefined'
  );
}

function envValue(name: string): string | undefined {
  return (
    (
      globalThis as typeof globalThis & {
        process?: { env?: Record<string, string | undefined> };
      }
    ).process?.env?.[name]
  );
}

function wasmChatFunction(): WasmChatFunction {
  return (
    wasmModule as typeof wasmModule & {
      chat: WasmChatFunction;
    }
  ).chat;
}

async function loadWasmSource(): Promise<URL | Uint8Array> {
  if (isSsrRuntime()) {
    const { readFile } = await import('node:fs/promises');
    return readFile(wasmUrl);
  }

  return wasmUrl;
}

export async function initXlai(): Promise<void> {
  initPromise ??= loadWasmSource()
    .then((wasmSource) => initWasm({ module_or_path: wasmSource }))
    .then(() => undefined);
  return initPromise;
}

export async function packageVersion(): Promise<string> {
  await initXlai();
  return wasmModule.package_version();
}

export async function chat(options: ChatOptions): Promise<ChatResponse> {
  await initXlai();

  const apiKey = options.apiKey ?? envValue('OPENAI_API_KEY');
  if (apiKey === undefined || apiKey.trim() === '') {
    throw new Error('OPENAI_API_KEY must be set or passed to chat()');
  }

  return wasmChatFunction()({
    prompt: options.prompt,
    apiKey,
    systemPrompt: options.systemPrompt,
    baseUrl: options.baseUrl ?? envValue('OPENAI_BASE_URL'),
    model: options.model ?? envValue('OPENAI_MODEL'),
    temperature: options.temperature,
    maxOutputTokens: options.maxOutputTokens,
  });
}
