import type { EmbeddingOptions, EmbeddingResponse } from './types';
import { getWasmModule, initXlai } from './wasm';

function requireApiKey(apiKey?: string): string {
  const resolved =
    apiKey ??
    (
      globalThis as typeof globalThis & {
        process?: { env?: Record<string, string | undefined> };
      }
    ).process?.env?.OPENAI_API_KEY;
  if (resolved === undefined || resolved.trim() === '') {
    throw new Error('OPENAI_API_KEY must be set or passed explicitly');
  }

  return resolved;
}

function envValue(name: string): string | undefined {
  return (
    globalThis as typeof globalThis & {
      process?: { env?: Record<string, string | undefined> };
    }
  ).process?.env?.[name];
}

function toWasmEmbeddingPayload(
  options: EmbeddingOptions,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    inputs: options.inputs,
    apiKey: requireApiKey(options.apiKey),
  };

  const baseUrl = options.baseUrl ?? envValue('OPENAI_BASE_URL');
  if (baseUrl !== undefined && baseUrl !== '') {
    payload.baseUrl = baseUrl;
  }

  const embeddingModel =
    options.embeddingModel ??
    envValue('OPENAI_EMBEDDING_MODEL') ??
    options.model ??
    envValue('OPENAI_MODEL');
  if (embeddingModel !== undefined && embeddingModel !== '') {
    payload.embeddingModel = embeddingModel;
  }

  const model = options.model ?? envValue('OPENAI_MODEL');
  if (model !== undefined && model !== '') {
    payload.model = model;
  }

  if (options.dimensions !== undefined) {
    payload.dimensions = options.dimensions;
  }

  return payload;
}

export async function embed(
  options: EmbeddingOptions,
): Promise<EmbeddingResponse> {
  await initXlai();
  const wasm = getWasmModule() as ReturnType<typeof getWasmModule> & {
    embed: (o: unknown) => Promise<EmbeddingResponse>;
  };
  return wasm.embed(toWasmEmbeddingPayload(options));
}
