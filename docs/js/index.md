# JavaScript / `@yai-xlai/xlai`

The npm package **`@yai-xlai/xlai`** is a Vite-friendly TypeScript layer on top of the **`xlai-wasm`** crate. It ships a bundled library plus generated wasm bindings under `pkg/` during build (not committed).

## Scripts (from the repository root)

| Command                                 | Purpose                                     |
| --------------------------------------- | ------------------------------------------- |
| `pnpm --filter @yai-xlai/xlai dev`      | Rebuild wasm and start the Vite dev server  |
| `pnpm --filter @yai-xlai/xlai build`    | Rebuild wasm and emit the library bundle    |
| `pnpm --filter @yai-xlai/xlai test`     | Rebuild wasm and run Vitest                 |
| `pnpm --filter @yai-xlai/xlai test:e2e` | Playwright smoke test against the demo page |

## Requirements

- **mise** for the pinned Node.js and Rust toolchains
- **pnpm**
- **wasm-pack**
- The `wasm32-unknown-unknown` Rust target from `mise install`

## WASM surface

The WebAssembly package mirrors the Rust split between **chat** and **agent** APIs (`chat`, `createChatSession`, `agent`, `createAgentSession`) and supports the same streaming and context-compressor semantics as documented under [Rust / Chat, agents, and tools](/rust/chat-and-agents).

## Token usage

Chat and embedding responses expose `usage.inputTokens`, `usage.outputTokens`, and `usage.totalTokens` when available. `usage.source` is optional and distinguishes `provider_reported`, `tokenizer_exact`, and `estimated` counts. Use provider-reported counts for billing reconciliation; agent `estimatedInputTokens` values are context-management estimates only.

Peer dependencies and export maps are defined in [`packages/xlai/package.json`](https://github.com/yetanother.ai/xlai/blob/main/packages/xlai/package.json).

For package-local developer notes, see [`packages/xlai/README.md`](https://github.com/yetanother.ai/xlai/blob/main/packages/xlai/README.md).
