import type { ImageGenerationOptions, ImageGenerationResponse } from './types';
import { normalizeInlineData } from './shared';
import { envValue, getWasmModule, initXlai } from './wasm';

function requireApiKey(apiKey?: string): string {
  const resolved = apiKey ?? envValue('OPENAI_API_KEY');
  if (resolved === undefined || resolved.trim() === '') {
    throw new Error('OPENAI_API_KEY must be set or passed explicitly');
  }

  return resolved;
}

function toWasmImageGenerationPayload(
  options: ImageGenerationOptions,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    prompt: options.prompt,
    apiKey: requireApiKey(options.apiKey),
  };

  const baseUrl = options.baseUrl ?? envValue('OPENAI_BASE_URL');
  if (baseUrl !== undefined && baseUrl !== '') {
    payload.baseUrl = baseUrl;
  }

  const model = options.model ?? envValue('OPENAI_MODEL');
  if (model !== undefined && model !== '') {
    payload.model = model;
  }

  const imageModel = options.imageModel ?? envValue('OPENAI_IMAGE_MODEL');
  if (imageModel !== undefined && imageModel !== '') {
    payload.imageModel = imageModel;
  }

  if (options.size !== undefined) {
    payload.size = options.size;
  }

  if (options.quality !== undefined) {
    payload.quality = options.quality;
  }

  if (options.background !== undefined) {
    payload.background = options.background;
  }

  if (options.outputFormat !== undefined) {
    payload.outputFormat = options.outputFormat;
  }

  if (options.count !== undefined) {
    payload.count = options.count;
  }

  return payload;
}

export async function generateImage(
  options: ImageGenerationOptions,
): Promise<ImageGenerationResponse> {
  await initXlai();
  const wasm = getWasmModule() as ReturnType<typeof getWasmModule> & {
    generateImage: (o: unknown) => Promise<ImageGenerationResponse>;
  };
  return normalizeInlineData(
    await wasm.generateImage(
      normalizeInlineData(toWasmImageGenerationPayload(options)),
    ),
  );
}
