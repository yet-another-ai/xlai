import { afterEach, describe, expect, it, vi } from 'vitest';

import { createAgentSession, createChatSession } from '../src/index';
import {
  LocalTtsRuntime,
  qtsBrowserTtsCapabilities,
  qtsBrowserTtsStream,
  validateQtsModelManifest,
} from '../src/qts';

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

describe('qts wasm bindings', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('exposes stub capabilities', async () => {
    const cap = await qtsBrowserTtsCapabilities();
    expect(cap.engine_available).toBe(false);
    expect(cap.engine_status).toBe('engine_pending');
  });

  it('validates manifest', async () => {
    await validateQtsModelManifest({
      schema_version: 1,
      model_id: 'm',
      revision: 'r',
      files: [
        { logical_name: 'main_gguf', filename: 'a.gguf' },
        { logical_name: 'vocoder_onnx', filename: 'v.onnx' },
      ],
    });
  });

  it('rejects invalid manifest deterministically', async () => {
    await expect(
      validateQtsModelManifest({
        schema_version: 1,
        model_id: 'm',
        revision: 'r',
        files: [{ logical_name: 'main_gguf', filename: 'a.gguf' }],
      }),
    ).rejects.toMatch(/vocoder_onnx/u);
  });

  it('qtsBrowserTtsStream returns qts_wasm_engine_pending', async () => {
    try {
      await qtsBrowserTtsStream({
        input: 'hello',
        voice: { kind: 'preset', name: 'alloy' },
        responseFormat: 'wav',
        delivery: 'stream',
      });
      expect.fail('expected rejection');
    } catch (err) {
      expect(qtsErrorCode(err)).toBe('qts_wasm_engine_pending');
    }
  });

  it('createChatSession with qts option constructs', async () => {
    vi.stubEnv('OPENAI_API_KEY', 'test-key');
    const session = await createChatSession({
      qts: {
        manifest: {
          schema_version: 1,
          model_id: 'm',
          revision: 'r',
          files: [
            { logical_name: 'main_gguf', filename: 'a.gguf' },
            { logical_name: 'vocoder_onnx', filename: 'v.onnx' },
          ],
        },
      },
    });
    expect(session).toBeDefined();
  });

  it('createAgentSession with qts option constructs', async () => {
    vi.stubEnv('OPENAI_API_KEY', 'test-key');
    const session = await createAgentSession({
      qts: {},
    });
    expect(session).toBeDefined();
  });

  it('LocalTtsRuntime stream uses runtime path and returns stub error', async () => {
    const rt = await LocalTtsRuntime.create(null);
    try {
      await rt.stream({
        input: 'hi',
        voice: { kind: 'preset', name: 'alloy' },
        delivery: 'stream',
      });
      expect.fail('expected rejection');
    } catch (err) {
      expect(qtsErrorCode(err)).toBe('qts_wasm_engine_pending');
    }
  });
});
