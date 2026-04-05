# Publishing crates.io and npm

This repository publishes a **subset** of Rust workspace members to [crates.io](https://crates.io) and the **`@yai-xlai/xlai`** package to [npm](https://www.npmjs.com/). Automation lives in [`.github/workflows/publish.yml`](../.github/workflows/publish.yml).

## What gets published

### Rust (crates.io)

These crates use shared `path` + `version` entries in the root [`Cargo.toml`](../Cargo.toml) under `[workspace.dependencies]` so `cargo publish` can resolve them once each version exists on the registry.

Publish **in this order** (the workflow does the same):

1. `xlai-core`
2. `xlai-runtime`
3. `xlai-local-common`
4. `xlai-backend-openai`
5. `xlai-backend-transformersjs`

### npm

- Package: **`@yai-xlai/xlai`** ([`packages/xlai/package.json`](../packages/xlai/package.json))
- Build runs `build:wasm` then the Vite library build; published files are `dist/` and `pkg/` per `files` in `package.json`.

## CI behavior

| Event | Rust | npm |
|--------|------|-----|
| Pull request, or push to `main` | `cargo publish -p … --dry-run --locked` for each crate in order (see note below) | `npm publish --dry-run` after `pnpm --filter @yai-xlai/xlai build` |
| Push tag `v*` | `cargo publish -p … --locked` in order (with pauses for index propagation) | `npm publish --access public --provenance` |

### Dry-run and unpublished dependencies

`cargo publish --dry-run` prepares the same manifest that would be uploaded: path dependencies on other workspace crates become **version-only** and must **already exist on crates.io**. So:

- **`xlai-core`** can always be dry-run before anything is on crates.io.
- Dependent crates may report `no matching package named 'xlai-core'` (or another xlai crate) until the previous crate in the chain has been published. The workflow treats that specific case as a **skipped dry-run with a warning** so PRs stay green before the first release. After the chain exists on crates.io, every crate’s dry-run should pass.

## GitHub configuration

Create a GitHub Environment named **`publish`** (same name as the Hugging Face QTS workflow) and add:

| Secret | Used by |
|--------|---------|
| `CRATES_IO_TOKEN` | Rust `cargo publish` on tags |
| `NPM_TOKEN` | npm publish on tags |

The Hugging Face workflow uses **`HF_TOKEN`** on the same environment; keep all release tokens scoped to `publish`.

## Version bumps

- Rust: bump `[workspace.package] version` in the root `Cargo.toml` and the `version = "…"` fields in `[workspace.dependencies]` for the five publishable crate entries (keep them in sync).
- npm: bump `"version"` in `packages/xlai/package.json`.
- Tag: push a tag matching `v*` (for example `v0.2.0`) after those bumps are on the branch you release from.

## Crates not published yet (refactor track)

These workspace members are **`publish = false`** or are blocked because they depend on unpublished / internal crates:

| Crate / area | Reason |
|--------------|--------|
| `xlai-sys` | `publish = false`; vendored native build; blocks `xlai-backend-llama-cpp` and anything pulling it |
| `xlai-qts-core`, `xlai-qts-browser`, `xlai-backend-qts-wasm` | `publish = false`; QTS stack not on crates.io |
| `xlai-observability` | `publish = false`; blocks `xlai-qts-cli` until policy changes |
| `xlai-backend-qts`, `xlai-native`, `xlai-wasm`, `xlai-ffi`, `xlai-qts-cli` | Depend on the above or on each other in ways that need a published graph or feature splits |

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
