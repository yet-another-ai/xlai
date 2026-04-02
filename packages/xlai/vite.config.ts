import { resolve } from 'node:path';
import { defineConfig } from 'vite';
import dts from 'vite-plugin-dts';
import wasm from 'vite-plugin-wasm';

export default defineConfig({
  plugins: [
    wasm(),
    dts({
      entryRoot: 'src',
      rollupTypes: true,
      exclude: ['tests/**', 'e2e/**'],
    }),
  ],
  build: {
    lib: {
      entry: {
        index: resolve(__dirname, 'src/index.ts'),
        transformers: resolve(__dirname, 'src/transformers/index.ts'),
      },
      formats: ['es'],
      fileName: (_format, entryName) => `${entryName}.js`,
    },
    rollupOptions: {
      external: (id) =>
        id === '@huggingface/transformers' ||
        id === 'transformers-llguidance' ||
        id.startsWith('@huggingface/transformers/') ||
        id.startsWith('transformers-llguidance/'),
    },
  },
});
