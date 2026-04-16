# Native vendor layout (`vendor/native`)

The `llama.cpp` and standalone `ggml` trees used by `xlai-sys-llama` and `xlai-sys-ggml` live as **git submodules** under:

- `vendor/native/llama.cpp`
- `vendor/native/ggml`

After cloning, initialize them (for example):

```bash
git submodule update --init --recursive vendor/native/llama.cpp vendor/native/ggml
```

## Override source paths (optional)

- **`GGML_SRC`**: absolute path to a `ggml` checkout for `xlai-sys-ggml` (defaults to `vendor/native/ggml`).
- **`LLAMA_CPP_SRC`**: absolute path to a `llama.cpp` checkout for `xlai-sys-llama` (defaults to `vendor/native/llama.cpp`).

## Dual native stacks

Enabling both local chat (`xlai-sys-llama`, which bundles `ggml` with `llama.cpp`) and native QTS (`xlai-sys-ggml`) links **two** native `ggml` implementations into one binary. Build scripts emit a `cargo:warning` when `xlai-facade` has `llama` + `qts`, or when `xlai-native` enables `qts`. Prefer separate processes or a single stack if you hit duplicate symbols or linker issues.
