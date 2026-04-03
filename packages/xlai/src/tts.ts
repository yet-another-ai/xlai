import type { TtsChunk, TtsOptions, TtsResponse } from './types';
import { envValue, getWasmModule, initXlai } from './wasm';

function requireApiKey(apiKey?: string): string {
  const resolved = apiKey ?? envValue('OPENAI_API_KEY');
  if (resolved === undefined || resolved.trim() === '') {
    throw new Error('OPENAI_API_KEY must be set or passed explicitly');
  }

  return resolved;
}

function toWasmTtsPayload(options: TtsOptions): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    input: options.input,
    apiKey: requireApiKey(options.apiKey),
    voice: options.voice,
  };

  const baseUrl = options.baseUrl ?? envValue('OPENAI_BASE_URL');
  if (baseUrl !== undefined && baseUrl !== '') {
    payload.baseUrl = baseUrl;
  }

  const model = options.model ?? envValue('OPENAI_MODEL');
  if (model !== undefined && model !== '') {
    payload.model = model;
  }

  const ttsModel = options.ttsModel ?? envValue('OPENAI_TTS_MODEL');
  if (ttsModel !== undefined && ttsModel !== '') {
    payload.ttsModel = ttsModel;
  }

  if (options.responseFormat !== undefined) {
    payload.responseFormat = options.responseFormat;
  }

  if (options.speed !== undefined) {
    payload.speed = options.speed;
  }

  if (options.instructions !== undefined) {
    payload.instructions = options.instructions;
  }

  if (options.delivery !== undefined) {
    payload.delivery = options.delivery;
  }

  return payload;
}

export async function tts(options: TtsOptions): Promise<TtsResponse> {
  await initXlai();
  const wasm = getWasmModule() as ReturnType<typeof getWasmModule> & {
    tts: (o: unknown) => Promise<TtsResponse>;
  };
  return wasm.tts(toWasmTtsPayload(options));
}

export async function ttsStream(options: TtsOptions): Promise<TtsChunk[]> {
  await initXlai();
  const wasm = getWasmModule() as ReturnType<typeof getWasmModule> & {
    ttsStream: (o: unknown) => Promise<TtsChunk[]>;
  };
  return wasm.ttsStream(toWasmTtsPayload(options));
}
