import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import dts from 'vite-plugin-dts';

export default defineConfig({
  plugins: [wasm(), dts()],
  build: {
    lib: {
      entry: 'src/index.ts',
      formats: ['es'],
      fileName: () => 'index.js',
    }
  },
});
