import { spawn } from 'node:child_process';
import { rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const packageDir = resolve(scriptDir, '..');
const outputDir = resolve(packageDir, 'pkg');

await rm(outputDir, { recursive: true, force: true });

const command = process.platform === 'win32' ? 'wasm-pack.exe' : 'wasm-pack';
const args = [
  'build',
  '../../crates/xlai-wasm',
  '--target',
  'web',
  '--out-dir',
  '../../packages/xlai/pkg',
  '--out-name',
  'xlai_wasm',
];

await new Promise((resolvePromise, rejectPromise) => {
  const child = spawn(command, args, {
    cwd: packageDir,
    stdio: 'inherit',
  });

  child.once('error', rejectPromise);
  child.once('exit', (code, signal) => {
    if (code === 0) {
      resolvePromise(undefined);
      return;
    }

    if (signal !== null) {
      rejectPromise(new Error(`wasm-pack exited via signal ${signal}`));
      return;
    }

    rejectPromise(new Error(`wasm-pack exited with code ${code ?? 'unknown'}`));
  });
});
