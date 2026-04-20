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

OpenAI smoke tests load `.env` automatically for local runs. For transcription e2e, set `OPENAI_TRANSCRIPTION_MODEL` to a transcription-capable model.

## GitHub Actions workflows

### Build (`.github/workflows/build.yml`)

Builds on Linux, Windows, macOS arm64, macOS x86_64 (cross-target), `wasm32-unknown-unknown`, and the `@yai-xlai/xlai` package through the pnpm workspace.

### Test (`.github/workflows/test.yml`)

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `pnpm --filter @yai-xlai/xlai test`

### Publish (`.github/workflows/publish.yml`)

On pull requests and pushes to `main`: `cargo publish --dry-run` for the publishable Rust crate subset and `npm publish --dry-run` for `@yai-xlai/xlai`.

On tags `v*`: real `cargo publish` and `npm publish` using the **`publish`** environment (`CRATES_IO_TOKEN`, `NPM_TOKEN`).

See [Publishing](./publishing) for ordering and crates that are not on crates.io yet.

### E2E (`.github/workflows/e2e.yml`)

Runs ignored tests with provider secrets. Use a protected environment (for example `e2e`) with maintainer approval.

Expected secrets and variables include `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_TRANSCRIPTION_MODEL`, `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL`, and for llama.cpp smoke tests `LLAMA_CPP_MODEL` (or the default fixture path under `fixtures/llama.cpp/`).

If you want OpenRouter requests to carry ranking / app-identification headers in CI, also set `OPENROUTER_HTTP_REFERER`, `OPENROUTER_APP_TITLE`, and `OPENROUTER_APP_CATEGORIES` in the protected `e2e` environment.

QTS integration tests expect `XLAI_QTS_MODEL_DIR` with a full model bundle; CI may skip them until a fixture download step exists.

## Documentation site CI

The docs site is built with `pnpm docs:build` and output to `docs/.vitepress/dist`. See the repository workflow **Deploy documentation** for publishing to GitHub Pages (optional).
