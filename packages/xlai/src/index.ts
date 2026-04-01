import initWasm, * as wasmModule from '../pkg/xlai_wasm.js';

let initPromise: Promise<void> | null = null;
const wasmUrl = new URL('../pkg/xlai_wasm_bg.wasm', import.meta.url);

async function loadWasmSource(): Promise<URL | Uint8Array> {
  if (import.meta.env.SSR) {
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
