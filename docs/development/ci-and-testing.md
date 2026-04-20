# CI and testing

## Test lanes

### Unit and mock tests

Default `cargo test` lane; no API tokens required.

```bash
cargo test --workspace
```

### Ignored end-to-end tests

Real provider credentials and local model paths. Run with:

```bash
cargo test --workspace -- --ignored --test-threads=1
```

OpenAI smoke tests load `.env` automatically for local runs. For embeddings/transcription/TTS/image e2e, set the corresponding model env vars such as `OPENAI_EMBEDDING_MODEL`, `OPENAI_TRANSCRIPTION_MODEL`, `OPENAI_TTS_MODEL`, and `OPENAI_IMAGE_MODEL`.

## GitHub Actions workflows

### Build (`.github/workflows/build.yml`)

Builds on Linux, Windows, macOS arm64, macOS x86_64 (cross-target), `wasm32-unknown-unknown`, and the `@yai-xlai/xlai` package through the pnpm workspace.

Native Rust jobs share `.github/actions/setup-xlai-rust-native`, which installs the Rust toolchain, caches cargo output, provisions OpenBLAS, installs the CUDA toolkit on Linux/Windows via `Jimver/cuda-toolkit@v0.2`, installs ROCm/HIP tooling on Linux and Windows, and finally prints the resolved accelerator SDK env vars so the static-core plus external-SDK linking behavior is visible in build logs. Vulkan remains an extra workflow step because only the Vulkan lanes need `glslc` / the SDK.

### Test (`.github/workflows/test.yml`)

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `pnpm --filter @yai-xlai/xlai test`

The Rust matrix reuses the same shared native setup action, so Linux/Windows unit-test lanes get the CUDA toolkit before building the default native feature set, and the ROCm/HIP setup is applied there as well. The sys-crate `build.rs` files honor `CUDA_PATH` / `OpenVINO_DIR` / `ROCM_PATH` (and standard install layouts) to discover the **external** accelerator SDKs they link against; missing SDKs degrade gracefully with `cargo:warning`. `hip` is currently still downgraded on every static-core build because upstream `ggml` does not allow `GGML_HIP=ON` with static linking, even when ROCm tooling is otherwise available.

### Publish (`.github/workflows/publish.yml`)

On pull requests and pushes to `main`: `cargo publish --dry-run` for the publishable Rust crate subset and `npm publish --dry-run` for `@yai-xlai/xlai`.

On tags `v*`: real `cargo publish` and `npm publish` using the **`publish`** environment (`CRATES_IO_TOKEN`, `NPM_TOKEN`).

See [Publishing](./publishing) for ordering and crates that are not on crates.io yet.

### E2E (`.github/workflows/e2e.yml`)

Runs ignored tests with provider secrets. Use a protected environment (for example `e2e`) with maintainer approval.

Expected secrets and variables include:

- OpenAI-compatible: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_IMAGE_MODEL`, `OPENAI_TRANSCRIPTION_MODEL`, `OPENAI_TTS_MODEL`
- Gemini: `GEMINI_API_KEY`, `GEMINI_BASE_URL`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`, `GEMINI_IMAGE_MODEL`
- OpenRouter: `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL`
- Local fixture-backed lanes: `LLAMA_CPP_MODEL` (or the default fixture path under `fixtures/llama.cpp/`)

If you want OpenRouter requests to carry ranking / app-identification headers in CI, also set `OPENROUTER_HTTP_REFERER`, `OPENROUTER_APP_TITLE`, and `OPENROUTER_APP_CATEGORIES` in the protected `e2e` environment.

QTS integration tests expect `XLAI_QTS_MODEL_DIR` with a full model bundle; CI may skip them until a fixture download step exists.

## Documentation site CI

The docs site is built with `pnpm docs:build` and output to `docs/.vitepress/dist`. See the repository workflow **Deploy documentation** for publishing to GitHub Pages (optional).
