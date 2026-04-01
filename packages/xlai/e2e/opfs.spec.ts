import { expect, test } from '@playwright/test';

test('supports the filesystem API through OPFS', async ({
  page,
  browserName,
}) => {
  test.skip(
    browserName !== 'chromium',
    'run OPFS coverage where it is reliable',
  );

  await page.goto('/');

  const packageModulePath = '/src/index.ts';
  const result = await page.evaluate(async (modulePath) => {
    const { createOpfsFileSystem } = await import(modulePath);

    const fileSystem = await createOpfsFileSystem();
    await fileSystem.createDirAll('/e2e/opfs');
    await fileSystem.write(
      '/e2e/opfs/hello.txt',
      new TextEncoder().encode('xlai-opfs-ok'),
    );

    const existsBeforeDelete = await fileSystem.exists('/e2e/opfs/hello.txt');
    const text = new TextDecoder().decode(
      await fileSystem.read('/e2e/opfs/hello.txt'),
    );
    const entries = await fileSystem.list('/e2e/opfs');

    await fileSystem.deletePath('/e2e');

    return {
      existsBeforeDelete,
      text,
      entries,
      existsAfterDelete: await fileSystem.exists('/e2e/opfs/hello.txt'),
    };
  }, packageModulePath);

  expect(result.existsBeforeDelete).toBe(true);
  expect(result.text).toBe('xlai-opfs-ok');
  expect(result.entries).toEqual([
    {
      path: '/e2e/opfs/hello.txt',
      kind: 'file',
    },
  ]);
  expect(result.existsAfterDelete).toBe(false);
});
