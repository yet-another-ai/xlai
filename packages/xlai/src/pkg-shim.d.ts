declare module '../pkg/xlai_wasm.js' {
  import type { ChatContent } from './types';

  type WasmModuleInput =
    | string
    | URL
    | Request
    | Response
    | BufferSource
    | WebAssembly.Module;
  const init: (options?: {
    module_or_path?: WasmModuleInput;
  }) => Promise<unknown>;

  export type WasmToolSession = {
    registerTool: (
      definition: unknown,
      callback: (argumentsValue: unknown) => unknown,
    ) => void;
    prompt: (content: string) => Promise<unknown>;
    promptWithContent?: (content: ChatContent) => Promise<unknown>;
    registerContextCompressor?: (
      callback: (messages: unknown, estimatedInputTokens: unknown) => unknown,
    ) => void;
    registerSystemReminder?: (callback: (messages: unknown) => unknown) => void;
    streamPrompt?: (content: string) => Promise<unknown>;
    streamPromptWithContent?: (content: ChatContent) => Promise<unknown>;
  };

  export default init;
  export function chat(options: unknown): Promise<unknown>;
  export function agent(options: unknown): Promise<unknown>;
  export function tts(options: unknown): Promise<unknown>;
  export function ttsStream(options: unknown): Promise<unknown>;
  export function createChatSession(options: unknown): WasmToolSession;
  export function createAgentSession(options: unknown): WasmToolSession;
  export function createChatSessionWithMemoryFileSystem(
    options: unknown,
    fileSystem: unknown,
  ): WasmToolSession;
  export function createAgentSessionWithMemoryFileSystem(
    options: unknown,
    fileSystem: unknown,
  ): WasmToolSession;
  export function createChatSessionWithFileSystem(
    options: unknown,
    fileSystem: unknown,
  ): WasmToolSession;
  export function createAgentSessionWithFileSystem(
    options: unknown,
    fileSystem: unknown,
  ): WasmToolSession;
  export function createTransformersChatSession(
    options: unknown,
  ): WasmToolSession;
  export function createTransformersChatSessionWithMemoryFileSystem(
    options: unknown,
    fileSystem: unknown,
  ): WasmToolSession;
  export function createTransformersChatSessionWithFileSystem(
    options: unknown,
    fileSystem: unknown,
  ): WasmToolSession;
  export function createTransformersAgentSession(
    options: unknown,
  ): WasmToolSession;
  export function createTransformersAgentSessionWithMemoryFileSystem(
    options: unknown,
    fileSystem: unknown,
  ): WasmToolSession;
  export function createTransformersAgentSessionWithFileSystem(
    options: unknown,
    fileSystem: unknown,
  ): WasmToolSession;
  export function package_version(): string;
}
