declare module '../pkg/xlai_wasm.js' {
  type WasmModuleInput = string | URL | Request | Response | BufferSource | WebAssembly.Module;
  const init: (options?: { module_or_path?: WasmModuleInput }) => Promise<unknown>;

  export default init;
  export function package_version(): string;
}
