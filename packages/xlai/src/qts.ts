import type {
  QtsModelFileEntry,
  QtsModelManifest,
  QtsSessionConfig,
  TtsChunk,
  TtsOptions,
  VoiceSpec,
} from './types';
import { initXlai, getWasmModule } from './wasm';

/** Wire shape from Rust `QtsBrowserCapabilities` (snake_case). */
export interface QtsBrowserCapabilities {
  engine_available: boolean;
  engine_status: string;
  talker_gpu_tier: string;
  vocoder_gpu_tier: string;
  schema_version: number;
}

export type { QtsModelFileEntry, QtsModelManifest };

/** Options for local QTS WASM calls (camelCase for JS; mirrored to Rust). */
export interface QtsTtsOptions {
  input: string;
  voice: VoiceSpec;
  responseFormat?: TtsOptions['responseFormat'];
  delivery?: TtsOptions['delivery'];
  manifest?: QtsModelManifest;
}

function toWasmQtsPayload(options: QtsTtsOptions): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    input: options.input,
    voice: options.voice,
  };
  if (options.responseFormat !== undefined) {
    payload.responseFormat = options.responseFormat;
  }
  if (options.delivery !== undefined) {
    payload.delivery = options.delivery;
  }
  if (options.manifest !== undefined) {
    payload.manifest = options.manifest;
  }
  return payload;
}

/** Capability snapshot for browser QTS (stub until engines are integrated). */
export async function qtsBrowserTtsCapabilities(): Promise<QtsBrowserCapabilities> {
  await initXlai();
  const wasm = getWasmModule() as typeof import('../pkg/xlai_wasm.js');
  return wasm.qtsBrowserTtsCapabilities() as QtsBrowserCapabilities;
}

/** Validates manifest required fields (`main_gguf`, `vocoder_onnx`). Throws on failure. */
export async function validateQtsModelManifest(
  manifest: QtsModelManifest,
): Promise<void> {
  await initXlai();
  const wasm = getWasmModule() as typeof import('../pkg/xlai_wasm.js');
  wasm.validateQtsModelManifest(manifest);
}

/** Local QTS unary synthesis. Stub build rejects with `details.code === 'qts_wasm_engine_pending'`. */
export async function qtsBrowserTts(options: QtsTtsOptions): Promise<unknown> {
  await initXlai();
  const wasm = getWasmModule() as typeof import('../pkg/xlai_wasm.js');
  return wasm.qtsBrowserTts(toWasmQtsPayload(options));
}

/** Local QTS stream (collects chunks). Stub returns a single error chunk path via thrown error. */
export async function qtsBrowserTtsStream(
  options: QtsTtsOptions,
): Promise<TtsChunk[]> {
  await initXlai();
  const wasm = getWasmModule() as typeof import('../pkg/xlai_wasm.js');
  return (await wasm.qtsBrowserTtsStream(
    toWasmQtsPayload(options),
  )) as TtsChunk[];
}

/** Handle returned by `createLocalTtsRuntime` in the generated WASM bindings. */
type WasmLocalTtsRuntimeHandle = {
  localTtsSynthesize(options: Record<string, unknown>): Promise<unknown>;
  localTtsStream(options: Record<string, unknown>): Promise<unknown>;
};

/** Narrow runtime built with only local QTS TTS (same engine path as session-backed runtimes). */
export class LocalTtsRuntime {
  private constructor(private readonly inner: WasmLocalTtsRuntimeHandle) {}

  static async create(
    options?: QtsSessionConfig | null,
  ): Promise<LocalTtsRuntime> {
    await initXlai();
    const wasm = getWasmModule() as typeof import('../pkg/xlai_wasm.js');
    const inner = wasm.createLocalTtsRuntime(
      options ?? null,
    ) as WasmLocalTtsRuntimeHandle;
    return new LocalTtsRuntime(inner);
  }

  synthesize(options: QtsTtsOptions): Promise<unknown> {
    return this.inner.localTtsSynthesize(toWasmQtsPayload(options));
  }

  stream(options: QtsTtsOptions): Promise<TtsChunk[]> {
    return this.inner.localTtsStream(toWasmQtsPayload(options)) as Promise<
      TtsChunk[]
    >;
  }
}
