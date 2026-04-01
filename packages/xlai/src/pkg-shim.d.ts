declare module '../pkg/xlai_wasm.js' {
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
  };

  export default init;
  export function chat(options: unknown): Promise<unknown>;
  export function agent(options: unknown): Promise<unknown>;
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
  export function package_version(): string;
}
