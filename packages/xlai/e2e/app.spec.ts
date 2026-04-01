import { expect, test } from '@playwright/test';

test('renders the wasm demo page', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'XLAI Wasm' })).toBeVisible();
  await expect(page.getByText('Rust package version:')).toBeVisible();
  await expect(page.locator('main code')).toHaveText(/\d+\.\d+\.\d+/);
});
