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

## Native sys-crate linking contract

The two sys crates that wrap upstream native sources (`xlai-sys-ggml` and `xlai-sys-llama`) follow a **mixed static-core plus external-SDK** linking model:

- **Vendored core (always static)**: `ggml`, `ggml-base`, `ggml-cpu`, the `llama.cpp` core (`llama`, `common`, `cpp-httplib`, `llguidance`), and any backend that upstream `ggml` / `llama.cpp` can produce as a static archive in this configuration (for example `ggml-blas`, `ggml-metal`, `ggml-vulkan`, `ggml-cuda`, `ggml-openvino`).
- **External accelerator SDKs (system / dynamic)**: the runtime libraries that ship with vendor-provided SDKs are treated as **external prerequisites**. The sys crate emits `cargo:rustc-link-search=native=...` / `cargo:rustc-link-lib=...` directives for them but does **not** vendor or statically embed them. This applies to:
  - **CUDA**: `cudart`, `cublas`, `cublasLt` from `CUDA_PATH` / `CUDA_HOME` / standard install layouts.
  - **OpenVINO**: `openvino`, `openvino_c` from `OpenVINO_DIR` / `OPENVINO_ROOT` / standard install layouts.
  - **ROCm / HIP**: `amdhip64`, `hipblas`, `rocblas` from `ROCM_PATH` / `HIP_PATH`. HIP additionally requires building `ggml-hip` against shared SDK libraries because upstream `ggml` rejects fully static HIP/ROCm builds.
- **Always-system libraries**: platform loaders such as the Vulkan loader (`vulkan` / `vulkan-1`), OpenBLAS, system C++ runtime, OpenMP, and OS frameworks (`Accelerate`, `Metal`, `MetalKit`, `Foundation`) remain external.

Per-backend gating is now driven by three signals rather than a single static-build switch:

1. **Target OS / platform support** (for example `metal` only on Apple, `cuda` not on Apple).
2. **Upstream backend constraints** (for example HIP cannot be statically embedded in `ggml`, so the helper builds it against the external ROCm SDK instead of trying to statically link it).
3. **External SDK presence** discovered via shared helpers in `xlai-build-native` (CUDA / OpenVINO / ROCm). When the SDK is missing on a non-Apple target, the build emits a `cargo:warning` and downgrades the backend instead of forcing an unbuildable CMake configuration.

## Default accelerator set

The native `llama.cpp` / QTS crates default to requesting:

- `openblas`
- `cuda`
- `hip`
- `openvino`

Behavior is platform-dependent:

- On unsupported Apple targets, `cuda`, `hip`, and `openvino` are skipped with build warnings so default local builds still work.
- On supported Linux / Windows targets, the corresponding SDKs must be reachable through the helpers above. When an SDK is not detected the backend is downgraded with a warning rather than failing the build.
- Vulkan and Metal remain opt-in feature flags.

### Environment variables for accelerator discovery

The shared helpers honor these environment variables (set them when the SDK lives in a non-default location):

| Backend  | Variables                                                                  |
| -------- | -------------------------------------------------------------------------- |
| CUDA     | `CUDA_PATH`, `CUDA_HOME`, `CUDA_TOOLKIT_ROOT_DIR`                          |
| OpenVINO | `OpenVINO_DIR`, `OPENVINO_ROOT`, `INTEL_OPENVINO_DIR`                      |
| ROCm/HIP | `ROCM_PATH`, `HIP_PATH`, `HIPCXX` (must be Clang, not the `hipcc` wrapper) |

CMake's `CMAKE_PREFIX_PATH` is also forwarded to the upstream build, so adding the SDK roots there continues to work.

## Dual native stacks

Enabling both local chat (`xlai-sys-llama`, which bundles `ggml` with `llama.cpp`) and native QTS (`xlai-sys-ggml`) links **two** native `ggml` implementations into one binary. Build scripts emit a `cargo:warning` when `xlai-facade` has `llama` + `qts`, or when `xlai-native` enables `qts`. Prefer separate processes or a single stack if you hit duplicate symbols or linker issues.
