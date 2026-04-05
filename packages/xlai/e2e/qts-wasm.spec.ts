import { expect, test } from '@playwright/test';

import {
  LocalTtsRuntime,
  createChatSession,
  qtsBrowserTtsCapabilities,
  qtsBrowserTtsStream,
  validateQtsModelManifest,
} from '../src/index';

function qtsErrorCode(err: unknown): string | undefined {
  if (!err || typeof err !== 'object' || !('details' in err)) {
    return undefined;
  }
  const details = (err as { details: unknown }).details;
  if (details instanceof Map) {
    const v = details.get('code');
    return typeof v === 'string' ? v : undefined;
  }
  if (details && typeof details === 'object' && 'code' in details) {
    const c = (details as { code: unknown }).code;
    return typeof c === 'string' ? c : undefined;
  }
  return undefined;
}

/**
 * Browser matrix (Chromium / Firefox / WebKit) is covered by Playwright projects in
 * playwright.config.ts. This spec does not require WebGPU or real models: it asserts the
 * stub wiring and stable error codes until GGML/ORT WASM engines land
 * (see docs/qts-wasm-browser-runtime.md).
 */
test('qtsBrowserTtsCapabilities reports stub engine_pending', async () => {
  const cap = await qtsBrowserTtsCapabilities();
  expect(cap.engine_available).toBe(false);
  expect(cap.engine_status).toBe('engine_pending');
  expect(cap.schema_version).toBe(1);
});

test('validateQtsModelManifest accepts a minimal valid manifest', async () => {
  await validateQtsModelManifest({
    schema_version: 1,
    model_id: 'test',
    revision: '1',
    files: [
      { logical_name: 'main_gguf', filename: 'm.gguf' },
      { logical_name: 'vocoder_onnx', filename: 'qwen3-tts-vocoder.onnx' },
    ],
  });
});

test('validateQtsModelManifest throws when vocoder is missing', async () => {
  await expect(
    validateQtsModelManifest({
      schema_version: 1,
      model_id: 'test',
      revision: '1',
      files: [{ logical_name: 'main_gguf', filename: 'm.gguf' }],
    }),
  ).rejects.toMatch(/vocoder_onnx/u);
});

test('qtsBrowserTtsStream rejects with qts_wasm_engine_pending', async () => {
  try {
    await qtsBrowserTtsStream({
      input: 'hello',
      voice: { kind: 'preset', name: 'alloy' },
      responseFormat: 'wav',
      delivery: 'stream',
    });
    throw new Error('expected rejection');
  } catch (err) {
    expect(qtsErrorCode(err)).toBe('qts_wasm_engine_pending');
  }
});

test('createChatSession with qts constructs in browser', async () => {
  const session = await createChatSession({
    apiKey: 'test-key',
    qts: {},
  });
  expect(session).toBeDefined();
});

test('LocalTtsRuntime stream rejects with qts_wasm_engine_pending', async () => {
  const rt = await LocalTtsRuntime.create(null);
  try {
    await rt.stream({
      input: 'hello',
      voice: { kind: 'preset', name: 'alloy' },
      delivery: 'stream',
    });
    throw new Error('expected rejection');
  } catch (err) {
    expect(qtsErrorCode(err)).toBe('qts_wasm_engine_pending');
  }
});
