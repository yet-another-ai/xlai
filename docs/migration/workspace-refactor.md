# Workspace refactor migration notes

This page summarizes breaking or noteworthy changes from the workspace layout and crate boundary refactor.

## `xlai-local-common` removed

The `xlai-local-common` crate was merged into **`xlai-runtime`** as the **`xlai_runtime::local_common`** module (same public types and functions as before).

### Rust imports

Replace:

```rust
use xlai_local_common::{PreparedLocalChatRequest, LocalChatPrepareOptions, /* … */};
```

with:

```rust
use xlai_runtime::local_common::{PreparedLocalChatRequest, LocalChatPrepareOptions, /* … */};
```

The `xlai_runtime::local_common` re-export path for consumers of the runtime crate is unchanged.

### Dependencies

- `xlai-backend-llama-cpp` and `xlai-backend-transformersjs` now depend on **`xlai-runtime`** instead of `xlai-local-common`.

## Crate directories grouped by role

Workspace members moved under:

- `crates/core/` — `xlai-core`
- `crates/runtime/` — `xlai-runtime`
- `crates/backends/` — HTTP and local chat backends
- `crates/qts/` — QTS engine, manifest, CLI
- `crates/sys/` — native build helpers and `xlai-sys-*`
- `crates/platform/` — `xlai-facade`, `xlai-native`, `xlai-wasm`, `xlai-ffi`

Update any scripts or docs that hard-coded paths such as `crates/xlai-wasm` (for example the npm `build-wasm` script now points at `crates/platform/xlai-wasm`).

## `xlai-backend-gemini`

`xlai-backend-gemini` is explicitly **`publish = false`** to match the internal-only taxonomy; it is not part of the crates.io publish order.

## `xlai-facade` documentation

`xlai-facade` is documented as an **internal** integration crate (not published to crates.io). Application entrypoints remain **`xlai-native`** (native Rust), **`xlai-wasm`** (browser), and **`xlai-ffi`** (C ABI).
