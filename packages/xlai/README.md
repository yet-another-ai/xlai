# `@yai-xlai/xlai`

Vite-based TypeScript package for the `xlai-wasm` Rust crate inside the `pnpm` workspace.

## Scripts

- `pnpm --filter @yai-xlai/xlai dev` builds the Rust wasm package and starts the Vite dev server.
- `pnpm --filter @yai-xlai/xlai build` rebuilds the Rust wasm package and emits the Vite bundle.
- `pnpm --filter @yai-xlai/xlai lint` runs ESLint for the package.
- `pnpm --filter @yai-xlai/xlai test` rebuilds the Rust wasm package and runs Vitest.
- `pnpm --filter @yai-xlai/xlai test:e2e` runs the Playwright browser smoke test against the demo page.

## Requirements

- Node.js
- `pnpm`
- `wasm-pack`
- Rust with the `wasm32-unknown-unknown` target installed

## Notes

The generated wasm bindings are written into `pkg/` and kept out of version control.
