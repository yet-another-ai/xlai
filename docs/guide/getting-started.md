# Getting started

## Requirements

- **Rust** stable, minimum **1.94** (see `rust-toolchain.toml` in the repo)
- **pnpm** (workspace uses `pnpm@10.x`)
- For the browser package: **Node.js**, **wasm-pack**, and the `wasm32-unknown-unknown` target

The workspace uses Rust **edition 2024** and **Apache-2.0**.

## Clone and build (Rust)

```bash
git clone https://github.com/yetanother.ai/xlai.git
cd xlai
cargo build --workspace
```

Native crates that wrap `llama.cpp` / QTS default to requesting `openblas`, `cuda`, `hip`, and `openvino`. The vendored `ggml` / `llama.cpp` core is built and linked **statically**, while accelerator SDK runtimes (CUDA, OpenVINO, ROCm/HIP) are treated as **external system libraries**. The build scripts auto-discover them via `CUDA_PATH` / `OpenVINO_DIR` / `ROCM_PATH` (and standard install paths). When an SDK is missing, the corresponding backend is downgraded with a `cargo:warning` instead of failing the build. `hip` is also currently downgraded on every static-core build because upstream `ggml` does not support `GGML_HIP=ON` together with static linking. See [Native vendor layout](../development/native-vendor) for the full linking contract.

## Tests

```bash
cargo test --workspace
```

### Check `wasm32`

```bash
rustup target add wasm32-unknown-unknown
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
