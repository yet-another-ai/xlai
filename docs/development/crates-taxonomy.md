# Workspace crate taxonomy

All members live under [`crates/`](https://github.com/yetanother.ai/xlai/tree/main/crates). This doc classifies them for dependency and release decisions. Authoritative publish order for crates.io is in [Publishing](./publishing) and [`.github/workflows/publish.yml`](https://github.com/yetanother.ai/xlai/blob/main/.github/workflows/publish.yml).

## Published to crates.io

| Crate                         | Role                                        |
| ----------------------------- | ------------------------------------------- |
| `xlai-core`                   | Domain types and provider traits            |
| `xlai-runtime`                | `RuntimeBuilder`, `Chat`, `Agent`, sessions |
| `xlai-backend-openai`         | OpenAI-compatible HTTP backend              |
| `xlai-backend-transformersjs` | WASM transformers.js chat backend           |

## Internal only (`publish = false`)

| Crate                    | Role                                                                         |
| ------------------------ | ---------------------------------------------------------------------------- |
| `xlai-sys-llama`         | Vendored `llama.cpp` build (CMake + bindgen)                                 |
| `xlai-sys-ggml`          | Vendored standalone `ggml` build for QTS (CMake + bindgen)                   |
| `xlai-local-common`      | Local chat prompt + tool JSON (used by llama.cpp / transformers.js backends) |
| `xlai-backend-llama-cpp` | Native llama.cpp chat backend                                                |
| `xlai-backend-gemini`    | Google Gemini HTTP backend                                                   |
| `xlai-facade`            | Shared re-exports for `xlai-native` / `xlai-wasm`                            |
| `xlai-native`            | Native app facade                                                            |
| `xlai-wasm`              | `wasm-bindgen` + npm package base                                            |
| `xlai-ffi`               | C ABI                                                                        |
| `xlai-qts-manifest`      | Browser QTS manifest / capability serde                                      |
| `xlai-qts-core`          | Native Qwen3 TTS engine + `QtsTtsModel`                                      |
| `xlai-qts-cli`           | `xlai-qts` binary                                                            |

## CI setup

Native Rust jobs share [`.github/actions/setup-xlai-rust-native`](https://github.com/yetanother.ai/xlai/blob/main/.github/actions/setup-xlai-rust-native/action.yml) (toolchain, sccache, `rust-cache`, OpenBLAS / vcpkg / shaderc). Jobs that need **Vulkan** (release `build.yml`) add Linux `glslc` / `libvulkan-dev` and the Vulkan SDK step after that action.

## Adding a new workspace member

1. Add the path to `[workspace].members` in the root `Cargo.toml`.
2. If the crate should ship to crates.io: set `publish = true` (default), add a `description`, wire `version.workspace = true`, add to `[workspace.dependencies]` with `path` + `version`, and append it to `PUBLISH_CRATES_ORDER` in `publish.yml` in dependency order.
3. If internal: set `publish = false` and list it here and in [Publishing](./publishing) under "not published".
