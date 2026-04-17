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

## `xlai-facade` and platform entrypoints

- **`xlai-facade`** is **internal** (not on crates.io) and is used by **`xlai-native`** only for native feature wiring and shared re-exports.
- **`xlai-wasm`** has **no** dependency on `xlai-facade`; it re-exports `xlai-core`, `xlai-runtime`, and the browser backends directly.
- **`xlai-native`** lists **explicit** `pub use` items (not `pub use xlai_facade::*`). Workspace-only **Gemini** types are also available under **`xlai_native::gemini`**; root-level `Gemini*` names remain for compatibility.

Application entrypoints remain **`xlai-native`** (native Rust), **`xlai-wasm`** (browser), and **`xlai-ffi`** (C ABI).

## Workspace policy checks

CI runs `python3 scripts/check_workspace_policy.py` to keep `publish = false`, crate taxonomy, directory layout, and the crates.io publish order in sync.
