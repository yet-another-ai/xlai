# Publishing crates.io and npm

This repository publishes a **subset** of Rust workspace members to [crates.io](https://crates.io) and the **`@yai-xlai/xlai`** package to [npm](https://www.npmjs.com/). Automation lives in [`.github/workflows/publish.yml`](https://github.com/yetanother.ai/xlai/blob/main/.github/workflows/publish.yml).

## What gets published

### Rust (crates.io)

These crates use shared `path` + `version` entries in the root [`Cargo.toml`](https://github.com/yetanother.ai/xlai/blob/main/Cargo.toml) under `[workspace.dependencies]` so `cargo publish` can resolve them once each version exists on the registry.

Publish **in this order** (the workflow does the same):

1. `xlai-core`
2. `xlai-runtime`
3. `xlai-backend-openai`
4. `xlai-backend-transformersjs`

### npm

- Package: **`@yai-xlai/xlai`** ([`packages/xlai/package.json`](https://github.com/yetanother.ai/xlai/blob/main/packages/xlai/package.json))
- Build runs `build:wasm` then the Vite library build; published files are `dist/` and `pkg/` per `files` in `package.json`.

## CI behavior

| Event                           | Rust                                                                             | npm                                                                |
| ------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Pull request, or push to `main` | `cargo publish -p … --dry-run --locked` for each crate in order (see note below) | `npm publish --dry-run` after `pnpm --filter @yai-xlai/xlai build` |
| Push tag `v*`                   | `cargo publish -p … --locked` in order (with pauses for index propagation)       | `npm publish --access public`                                      |

### `publish = false` crates

Crates marked `publish = false` are **not** released to crates.io. `cargo publish -p <crate> --dry-run` fails immediately with a clear error—this is expected for `xlai-backend-llama-cpp`, `xlai-native`, `xlai-ffi`, `xlai-wasm`, the QTS stack, etc. Use the [allowlist above](#rust-cratesio) for registry releases.

### Dry-run and unpublished dependencies

`cargo publish --dry-run` prepares the same manifest that would be uploaded: path dependencies on other workspace crates become **version-only** and must **already exist on crates.io**. So:

- **`xlai-core`** can always be dry-run before anything is on crates.io.
- Dependent crates may report `no matching package named 'xlai-core'` (or another xlai crate) until the previous crate in the chain has been published. The workflow treats that specific case as a **skipped dry-run with a warning** so PRs stay green before the first release. After the chain exists on crates.io, every crate's dry-run should pass.

## GitHub configuration

Create a GitHub Environment named **`publish`** (same name as the Hugging Face QTS workflow) and add:

| Secret            | Used by                      |
| ----------------- | ---------------------------- |
| `CRATES_IO_TOKEN` | Rust `cargo publish` on tags |
| `NPM_TOKEN`       | npm publish on tags          |

The Hugging Face workflow uses **`HF_TOKEN`** on the same environment; keep all release tokens scoped to `publish`.

## Version bumps

- Rust: bump `[workspace.package] version` in the root `Cargo.toml` and every `version = "…"` in `[workspace.dependencies]` for workspace members (keep them in sync with the tag you are releasing).
- npm: bump `"version"` in `packages/xlai/package.json`.
- Tag: push a tag matching `v*` (for example `v0.2.0`) after those bumps are on the branch you release from.

## Crates not published yet (refactor track)

These workspace members are **`publish = false`** (Cargo will refuse `cargo publish` / `--dry-run` for them) until the dependency graph is crates.io-ready:

| Crate / area                                         | Reason                                                                                                                                                       |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `xlai-build-native`                                  | Internal build helpers for native sys crates; not on crates.io                                                                                               |
| `xlai-sys-llama`, `xlai-sys-ggml`                    | Vendored native builds (`vendor/native/*`); not on crates.io                                                                                                 |
| `xlai-backend-llama-cpp`                             | Depends on `xlai-sys-llama`                                                                                                                                  |
| `xlai-backend-gemini`                                | Workspace-only HTTP backend; not in the crates.io publish chain                                                                                              |
| `xlai-native`, `xlai-ffi`                            | Depend on `xlai-backend-llama-cpp` / aggregate of internal backends                                                                                          |
| `xlai-wasm`                                          | Built for the npm package; not published as its own crate                                                                                                    |
| `xlai-facade`                                        | Internal integration re-exports for platform crates; `publish = false`                                                                                       |
| `xlai-qts-core`, `xlai-qts-cli`, `xlai-qts-manifest` | QTS stack / internal (engine in `xlai-qts-core`; browser manifest serde in `xlai-qts-manifest`; WASM QTS stub uses manifest crate only, not the GGML engine) |

To publish more crates later: remove or relax `publish = false`, ensure every dependency is either on crates.io with a version pin or optional behind features, add `description` / `repository.workspace = true` as needed, extend `[workspace.dependencies]` and `.github/workflows/publish.yml` `PUBLISH_CRATES_ORDER`.

## Manual publish (local)

With tokens set:

```bash
export CARGO_REGISTRY_TOKEN=…   # crates.io
# In order:
cargo publish -p xlai-core --locked
cargo publish -p xlai-runtime --locked
# … same order as above …
```

npm:

```bash
pnpm install --frozen-lockfile
pnpm --filter @yai-xlai/xlai build
cd packages/xlai && npm publish --access public
```
