# `@yai-xlai/xlai`

Vite-based TypeScript package for the `xlai-wasm` Rust crate inside the `pnpm` workspace.

## Scripts

- `pnpm --filter @yai-xlai/xlai dev` builds the Rust wasm package and starts the Vite dev server.
- `pnpm --filter @yai-xlai/xlai build` rebuilds the Rust wasm package and emits the Vite bundle.
- `pnpm --filter @yai-xlai/xlai lint` runs ESLint for the package.
- `pnpm --filter @yai-xlai/xlai test` rebuilds the Rust wasm package and runs Vitest.
- `pnpm --filter @yai-xlai/xlai test:e2e` runs the Playwright browser smoke test against the demo page.

## Requirements

- `mise` for the pinned Node.js, pnpm, Rust, and wasm-pack toolchains
- The `wasm32-unknown-unknown` Rust target, installed by `mise install`

## Notes

The generated wasm bindings are written into `pkg/` and kept out of version control.

Chat and embedding responses include `usage` when a backend can provide token counts. The optional `usage.source` value is `provider_reported`, `tokenizer_exact`, or `estimated`; only the first two should be treated as exact final counts.
