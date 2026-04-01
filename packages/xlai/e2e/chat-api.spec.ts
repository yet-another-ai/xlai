import { expect, test } from '@playwright/test';

import { chat } from '../src/index';

test('calls the chat API with env-backed credentials', async ({ browserName }) => {
  test.skip(
    browserName !== 'chromium',
    'run the provider-backed smoke test once to avoid duplicate API calls',
  );
  test.skip(
    process.env.OPENAI_API_KEY === undefined || process.env.OPENAI_API_KEY.trim() === '',
    'requires OPENAI_API_KEY from the environment or local .env file',
  );

  const response = await chat({
    prompt: 'Reply with the exact token xlai-js-e2e-ok and nothing else.',
  });

  expect(response.message.role).toBe('assistant');
  expect(response.message.content).toMatch(/xlai-js-e2e-ok/i);
});
