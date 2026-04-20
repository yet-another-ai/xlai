//! Discovery and linking helpers for external accelerator SDKs (CUDA, OpenVINO, ROCm/HIP).
//!
//! These helpers implement the **mixed static-core plus external-SDK** linking contract used by
//! [`xlai-sys-ggml`](../../xlai-sys-ggml/build.rs) and
//! [`xlai-sys-llama`](../../xlai-sys-llama/build.rs). The vendored `ggml` / `llama.cpp` core is
//! built and linked statically by the cmake step that produced its archives, but the accelerator
//! SDK runtime libraries (`cudart`, `openvino`, `amdhip64`, ...) are treated as **external system
//! dependencies** that must be discovered at build time and linked dynamically.
//!
//! Each helper:
//!
//! 1. Emits `cargo:rerun-if-env-changed=...` for the accelerator's discovery variables.
//! 2. Walks the configured environment variables, then the platform's standard install path, to
//!    locate the SDK root.
//! 3. Emits `cargo:rustc-link-search=native=...` for every SDK lib directory that exists.
//! 4. Emits `cargo:rustc-link-lib=...` for the runtime libraries this workspace consumes.
//! 5. Returns whether SDK linkage was emitted, so callers can downgrade the backend with a warning
//!    when the SDK is missing.
//!
//! See [`docs/development/native-vendor.md`](../../../../docs/development/native-vendor.md) for the
//! linking contract this module implements.

use std::env;
use std::path::{Path, PathBuf};

use crate::link_search::{emit_search_path, emit_search_path_variants};

/// Result of locating an external accelerator SDK on the build host.
#[derive(Debug, Clone)]
pub struct SdkLayout {
    /// SDK install root (the directory whose `lib*` / `runtime` / `bin` subtrees we will search).
    pub root: PathBuf,
    /// Name of the env var or fixed path that resolved the SDK, used for diagnostics.
    pub source: String,
    /// Concrete `lib` directories under [`Self::root`] that exist on disk.
    pub lib_dirs: Vec<PathBuf>,
}

/// Emit `cargo:rerun-if-env-changed=` for every CUDA discovery variable.
pub fn rerun_cuda_env() {
    for name in CUDA_ENV_VARS {
        println!("cargo:rerun-if-env-changed={name}");
    }
}

/// Emit `cargo:rerun-if-env-changed=` for every OpenVINO discovery variable.
pub fn rerun_openvino_env() {
    for name in OPENVINO_ENV_VARS {
        println!("cargo:rerun-if-env-changed={name}");
    }
}

/// Emit `cargo:rerun-if-env-changed=` for every ROCm/HIP discovery variable.
pub fn rerun_rocm_env() {
    for name in ROCM_ENV_VARS {
        println!("cargo:rerun-if-env-changed={name}");
    }
}

const CUDA_ENV_VARS: &[&str] = &[
    "CUDA_PATH",
    "CUDA_HOME",
    "CUDA_TOOLKIT_ROOT_DIR",
    "CUDAToolkit_ROOT",
];

const OPENVINO_ENV_VARS: &[&str] = &[
    "OpenVINO_DIR",
    "OPENVINO_ROOT",
    "INTEL_OPENVINO_DIR",
    "OPENVINO_HOME",
];

const ROCM_ENV_VARS: &[&str] = &["ROCM_PATH", "HIP_PATH", "ROCM_HOME", "HIPCXX"];

/// Try to locate the CUDA toolkit on the build host.
///
/// Search order:
///
/// 1. The first non-empty value among `CUDA_PATH`, `CUDA_HOME`, `CUDA_TOOLKIT_ROOT_DIR`,
///    `CUDAToolkit_ROOT`.
/// 2. Standard OS install paths (`/usr/local/cuda*` on Linux, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*` on Windows).
#[must_use]
pub fn detect_cuda_sdk(target: &str) -> Option<SdkLayout> {
    if let Some(layout) = layout_from_env(CUDA_ENV_VARS, |root| cuda_lib_dirs(root, target)) {
        return Some(layout);
    }

    for (label, root) in default_cuda_roots(target) {
        if root.is_dir() {
            let lib_dirs = cuda_lib_dirs(&root, target);
            if !lib_dirs.is_empty() {
                return Some(SdkLayout {
                    root,
                    source: label.to_string(),
                    lib_dirs,
                });
            }
        }
    }

    None
}

/// Try to locate an OpenVINO runtime install on the build host.
#[must_use]
pub fn detect_openvino_sdk(target: &str) -> Option<SdkLayout> {
    if let Some(layout) = layout_from_env(OPENVINO_ENV_VARS, |root| openvino_lib_dirs(root, target))
    {
        return Some(layout);
    }

    for (label, root) in default_openvino_roots(target) {
        if root.is_dir() {
            let lib_dirs = openvino_lib_dirs(&root, target);
            if !lib_dirs.is_empty() {
                return Some(SdkLayout {
                    root,
                    source: label.to_string(),
                    lib_dirs,
                });
            }
        }
    }

    None
}

/// Try to locate a ROCm/HIP install on the build host.
#[must_use]
pub fn detect_rocm_sdk(target: &str) -> Option<SdkLayout> {
    if let Some(layout) = layout_from_env(ROCM_ENV_VARS, |root| rocm_lib_dirs(root, target)) {
        return Some(layout);
    }

    for (label, root) in default_rocm_roots(target) {
        if root.is_dir() {
            let lib_dirs = rocm_lib_dirs(&root, target);
            if !lib_dirs.is_empty() {
                return Some(SdkLayout {
                    root,
                    source: label.to_string(),
                    lib_dirs,
                });
            }
        }
    }

    None
}

/// Emit search paths and dynamic link directives for CUDA runtime libraries.
///
/// Returns `true` when SDK linkage was emitted. When `false`, callers should downgrade the
/// requested backend with a `cargo:warning`.
pub fn emit_cuda_link(target: &str) -> bool {
    rerun_cuda_env();
    let Some(layout) = detect_cuda_sdk(target) else {
        return false;
    };

    for dir in &layout.lib_dirs {
        emit_search_path_variants(dir);
    }
    for lib in cuda_runtime_libs(target) {
        println!("cargo:rustc-link-lib={lib}");
    }
    true
}

/// Emit search paths and dynamic link directives for OpenVINO runtime libraries.
pub fn emit_openvino_link(target: &str) -> bool {
    rerun_openvino_env();
    let Some(layout) = detect_openvino_sdk(target) else {
        return false;
    };

    for dir in &layout.lib_dirs {
        emit_search_path_variants(dir);
    }
    for lib in openvino_runtime_libs(target) {
        println!("cargo:rustc-link-lib={lib}");
    }
    true
}

/// Emit search paths and dynamic link directives for ROCm/HIP runtime libraries.
pub fn emit_rocm_link(target: &str) -> bool {
    rerun_rocm_env();
    let Some(layout) = detect_rocm_sdk(target) else {
        return false;
    };

    for dir in &layout.lib_dirs {
        emit_search_path_variants(dir);
    }
    for lib in rocm_runtime_libs(target) {
        println!("cargo:rustc-link-lib={lib}");
    }
    true
}

fn layout_from_env<F>(env_vars: &'static [&'static str], lib_dirs: F) -> Option<SdkLayout>
where
    F: Fn(&Path) -> Vec<PathBuf>,
{
    for name in env_vars {
        let Some(value) = env::var_os(name) else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        let initial = PathBuf::from(value);
        let initial = if name == &"HIPCXX" {
            // HIPCXX usually points at `<rocm>/llvm/bin/clang`; walk up to the rocm install root.
            initial
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .map(PathBuf::from)
                .unwrap_or(initial)
        } else {
            initial
        };
        // Try the env value first, then walk up a few parents in case the user pointed at a
        // subdir like `<root>/runtime/cmake` (typical for `OpenVINO_DIR`) or `<root>/lib/cmake`.
        let mut candidate = Some(initial.clone());
        for _ in 0..4 {
            let Some(current) = candidate else { break };
            if current.is_dir() {
                let dirs = lib_dirs(&current);
                if !dirs.is_empty() {
                    return Some(SdkLayout {
                        root: current,
                        source: (*name).to_string(),
                        lib_dirs: dirs,
                    });
                }
            }
            candidate = current.parent().map(PathBuf::from);
        }
    }
    None
}

fn cuda_lib_dirs(root: &Path, target: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if target.contains("windows") {
        push_existing(&mut dirs, root.join("lib").join("x64"));
        push_existing(&mut dirs, root.join("lib"));
    } else {
        push_existing(&mut dirs, root.join("lib64"));
        push_existing(&mut dirs, root.join("lib"));
        push_existing(&mut dirs, root.join("targets/x86_64-linux/lib"));
        push_existing(&mut dirs, root.join("targets/x86_64-linux/lib/stubs"));
        push_existing(&mut dirs, root.join("lib/stubs"));
        push_existing(&mut dirs, root.join("lib64/stubs"));
    }
    dirs
}

fn openvino_lib_dirs(root: &Path, target: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let runtime = root.join("runtime");
    if target.contains("windows") {
        push_existing(&mut dirs, runtime.join("lib/intel64/Release"));
        push_existing(&mut dirs, runtime.join("lib/intel64"));
        push_existing(&mut dirs, runtime.join("bin/intel64/Release"));
        push_existing(&mut dirs, root.join("lib/intel64/Release"));
        push_existing(&mut dirs, root.join("lib/intel64"));
    } else {
        push_existing(&mut dirs, runtime.join("lib/intel64"));
        push_existing(&mut dirs, runtime.join("lib"));
        push_existing(&mut dirs, root.join("lib/intel64"));
        push_existing(&mut dirs, root.join("lib"));
    }
    dirs
}

fn rocm_lib_dirs(root: &Path, target: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if target.contains("windows") {
        push_existing(&mut dirs, root.join("lib"));
        push_existing(&mut dirs, root.join("bin"));
    } else {
        push_existing(&mut dirs, root.join("lib"));
        push_existing(&mut dirs, root.join("lib64"));
        push_existing(&mut dirs, root.join("hip/lib"));
    }
    dirs
}

fn push_existing(dirs: &mut Vec<PathBuf>, candidate: PathBuf) {
    if candidate.is_dir() && !dirs.contains(&candidate) {
        dirs.push(candidate);
    }
}

fn default_cuda_roots(target: &str) -> Vec<(&'static str, PathBuf)> {
    let mut out = Vec::new();
    if target.contains("windows") {
        let base = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
        if base.is_dir()
            && let Ok(entries) = std::fs::read_dir(&base)
        {
            let mut versions: Vec<_> = entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|p| p.is_dir())
                .collect();
            versions.sort();
            versions.reverse();
            for v in versions {
                out.push(("CUDA default install", v));
            }
        }
    } else if !target.contains("apple") {
        for path in ["/usr/local/cuda", "/opt/cuda"] {
            out.push(("CUDA default install", PathBuf::from(path)));
        }
    }
    out
}

fn default_openvino_roots(target: &str) -> Vec<(&'static str, PathBuf)> {
    let mut out = Vec::new();
    if target.contains("windows") {
        out.push((
            "OpenVINO default install",
            PathBuf::from(r"C:\Program Files (x86)\Intel\openvino"),
        ));
        out.push((
            "OpenVINO default install",
            PathBuf::from(r"C:\Program Files\Intel\openvino"),
        ));
    } else if !target.contains("apple") {
        for path in [
            "/opt/intel/openvino",
            "/opt/intel/openvino_2024",
            "/opt/intel/openvino_2025",
        ] {
            out.push(("OpenVINO default install", PathBuf::from(path)));
        }
    }
    out
}

fn default_rocm_roots(target: &str) -> Vec<(&'static str, PathBuf)> {
    let mut out = Vec::new();
    if target.contains("windows") {
        let base = PathBuf::from(r"C:\Program Files\AMD\ROCm");
        if base.is_dir()
            && let Ok(entries) = std::fs::read_dir(&base)
        {
            let mut versions: Vec<_> = entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|p| p.is_dir())
                .collect();
            versions.sort();
            versions.reverse();
            for v in versions {
                out.push(("ROCm default install", v));
            }
        }
    } else if !target.contains("apple") {
        out.push(("ROCm default install", PathBuf::from("/opt/rocm")));
    }
    out
}

fn cuda_runtime_libs(_target: &str) -> &'static [&'static str] {
    // `cuda` is the NVIDIA driver library (libcuda.so / cuda.lib). ggml-cuda's VMM pool uses
    // the driver API (cuMemCreate / cuMemMap / ...), not just the runtime API in `cudart`.
    // The driver lib ships at runtime with the NVIDIA driver; the CUDA toolkit provides a
    // build-time stub under `lib64/stubs/` (Linux) or alongside `lib/x64/cuda.lib` (Windows).
    &["cudart", "cublas", "cublasLt", "cuda"]
}

fn openvino_runtime_libs(_target: &str) -> &'static [&'static str] {
    &["openvino", "openvino_c"]
}

fn rocm_runtime_libs(_target: &str) -> &'static [&'static str] {
    &["amdhip64", "hipblas", "rocblas"]
}

/// Emit a single `cargo:rustc-link-search=native=` directive for the SDK root's `bin`
/// directory, useful on Windows where DLLs (and import libs) live alongside binaries.
pub fn emit_sdk_bin_search(layout: &SdkLayout) {
    emit_search_path(&layout.root.join("bin"));
}
