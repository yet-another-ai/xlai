# Getting started

## Requirements

- **mise** for the pinned Node.js and Rust toolchains
- **pnpm** (workspace uses `pnpm@10.x`)
- For the browser package: **wasm-pack**; `mise install` also installs the `wasm32-unknown-unknown` Rust target

The workspace uses Rust **edition 2024** and **Apache-2.0**.

## Clone and build (Rust)

```bash
git clone https://github.com/yetanother.ai/xlai.git
cd xlai
mise install
cargo build --workspace
```

## Tests

```bash
cargo test --workspace
```

### Check `wasm32`

```bash
cargo check -p xlai-wasm --target wasm32-unknown-unknown --features qts
```

## JavaScript workspace

From the repository root:

```bash
pnpm install
pnpm --filter @yai-xlai/xlai build
pnpm --filter @yai-xlai/xlai test
```

## Linting (Rust)

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Documentation site

To work on this site locally:

```bash
pnpm docs:dev
```

Build static output:

```bash
pnpm docs:build
```

Output is written to `docs/.vitepress/dist`.
