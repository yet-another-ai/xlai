import { expect, test } from '@playwright/test';

const E2E_MODEL_ID =
  process.env.XLAI_TRANSFORMERS_E2E_MODEL ?? 'Xenova/distilgpt2';

test('runs transformers.js constrained generation with a real browser model', async ({
  page,
  browserName,
}) => {
  test.skip(
    browserName !== 'chromium',
    'real browser transformers.js inference is validated on Chromium only',
  );
  test.slow();
  test.setTimeout(180_000);

  await page.goto('/');

  const result = await page.evaluate(
    async ({ modelId }) => {
      const { createTransformersChatSession, createXlaiTransformersJsAdapter } =
        await import('/src/transformers/index.ts');

      const adapter = await createXlaiTransformersJsAdapter({
        speculationDepth: 5,
      });

      const structured = await adapter.generate({
        prompt: 'Return the required JSON object and nothing else.',
        model: modelId,
        temperature: 0,
        maxNewTokens: 48,
        grammar: {
          type: 'json_schema',
          schema: {
            type: 'object',
            properties: {
              status: {
                type: 'string',
                enum: ['xlai-js-transformers-e2e-ok'],
              },
            },
            required: ['status'],
            additionalProperties: false,
          },
        },
      });

      const parsed = JSON.parse(structured.text) as { status: string };

      const session = await createTransformersChatSession({
        modelId,
        temperature: 0,
        maxOutputTokens: 12,
      });
      const chat = await session.prompt('Reply with a short lowercase word.');

      return {
        structured,
        parsed,
        chat,
      };
    },
    { modelId: E2E_MODEL_ID },
  );

  expect(result.structured.finishReason).toBe('stopped');
  expect(result.parsed).toEqual({
    status: 'xlai-js-transformers-e2e-ok',
  });
  expect(result.chat.message.role).toBe('assistant');
  expect(typeof result.chat.message.content).toBe('string');
  expect((result.chat.message.content as string).trim().length).toBeGreaterThan(
    0,
  );
});
