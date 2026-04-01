import initWasm, * as wasmModule from '../pkg/xlai_wasm.js';

let initPromise: Promise<void> | null = null;
const wasmUrl = new URL('../pkg/xlai_wasm_bg.wasm', import.meta.url);

function isSsrRuntime(): boolean {
  return (
    (
      import.meta as ImportMeta & {
        env?: { SSR?: boolean };
      }
    ).env?.SSR ?? typeof window === 'undefined'
  );
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

export function getWasmModule() {
  return wasmModule;
}

export function envValue(name: string): string | undefined {
  return (
    globalThis as typeof globalThis & {
      process?: { env?: Record<string, string | undefined> };
    }
  ).process?.env?.[name];
}
