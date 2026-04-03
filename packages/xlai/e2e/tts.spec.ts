import { Buffer } from 'node:buffer';

import { expect, test } from '@playwright/test';

import { tts, ttsStream } from '../src/index';

test('calls unary TTS and returns inline audio', async () => {
  test.skip(
    process.env.OPENAI_API_KEY === undefined ||
      process.env.OPENAI_API_KEY.trim() === '',
    'requires OPENAI_API_KEY from the environment or local .env file',
  );

  const response = await tts({
    input: 'Say only the token xlai-js-tts-e2e for testing.',
    voice: { kind: 'preset', name: 'alloy' },
    responseFormat: 'mp3',
  });

  expect(response.mime_type).toMatch(/audio\//u);
  const { audio } = response;
  expect(audio.kind).toBe('inline_data');
  if (audio.kind !== 'inline_data') {
    return;
  }

  const decoded = Buffer.from(audio.data_base64, 'base64');
  expect(decoded.length).toBeGreaterThan(64);
});

test('ttsStream collects started, audio delta, and finished (unary synthesis)', async () => {
  test.skip(
    process.env.OPENAI_API_KEY === undefined ||
      process.env.OPENAI_API_KEY.trim() === '',
    'requires OPENAI_API_KEY from the environment or local .env file',
  );

  const chunks = await ttsStream({
    input: 'Short stream test for xlai js e2e.',
    voice: { kind: 'preset', name: 'alloy' },
    delivery: 'unary',
    responseFormat: 'mp3',
  });

  expect(chunks.length).toBe(3);
  expect(chunks[0]?.type).toBe('started');
  expect(chunks[1]?.type).toBe('audio_delta');
  expect(chunks[2]?.type).toBe('finished');

  const delta = chunks[1];
  expect(delta?.type).toBe('audio_delta');
  if (delta?.type === 'audio_delta') {
    expect(Buffer.from(delta.data_base64, 'base64').length).toBeGreaterThan(64);
  }

  const last = chunks[2];
  expect(last?.type).toBe('finished');
  if (last?.type === 'finished') {
    expect(last.response.mime_type).toMatch(/audio\//u);
  }
});
