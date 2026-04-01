declare module '../pkg/xlai_wasm.js' {
  const init: (moduleOrPath?: string | URL | Request | Response | BufferSource | WebAssembly.Module) => Promise<unknown>;

  export default init;
  export function package_version(): string;
}
