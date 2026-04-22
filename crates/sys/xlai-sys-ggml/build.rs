#![allow(clippy::expect_used)]
#![allow(clippy::panic)]
#![allow(clippy::unwrap_used)]

use std::env;
use std::path::{Path, PathBuf};

use xlai_build_native::{
    apply_cmake_env_overrides, emit_cuda_link, emit_openblas_search_paths, emit_openvino_link,
    emit_rocm_link, emit_vulkan_loader_links, executable_in_path, feature_enabled,
    find_ggml_lib_dir, map_feature_cmake, native_vendor_ggml, normalize_source_path,
    rerun_cuda_env, rerun_openvino_env, rerun_rocm_env, validate_ggml_features,
    workspace_root_from_sys_crate_manifest,
};

const CRATE_LABEL: &str = "xlai-sys-ggml";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    println!("cargo:rerun-if-changed=build.rs");
    build_qts_standalone_ggml(&manifest_dir, &out_dir);
}

/// Linking model for the vendored GGML core in this crate.
///
/// We build the core libraries statically via `BUILD_SHARED_LIBS=OFF` / `GGML_STATIC=ON`. This
/// flag exists so per-backend gating can ask "does the configured static-core build mode allow
/// embedding this backend?" without re-deriving CMake state.
const STATIC_CORE: bool = true;

/// Per-backend gating decision, keyed on three independent signals:
///
/// 1. Target OS / platform support (e.g. `metal` is Apple-only).
/// 2. Upstream backend constraints (e.g. `ggml-hip` cannot be embedded in a static core build).
/// 3. External SDK presence detected via [`xlai_build_native::detect_*_sdk`] helpers.
#[derive(Default, Clone, Copy)]
struct BackendDecisions {
    cuda: bool,
    hip: bool,
    openvino: bool,
}

fn build_qts_standalone_ggml(manifest_dir: &Path, out_dir: &Path) {
    let target = env::var("TARGET").expect("TARGET");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let openblas_fe = feature_enabled("openblas");
    let enable_linux_windows_blas =
        openblas_fe && matches!(target_os.as_str(), "linux" | "windows");
    let link_ggml_blas = openblas_fe && (enable_linux_windows_blas || target.contains("apple"));

    let workspace_root =
        workspace_root_from_sys_crate_manifest(manifest_dir).expect("workspace root");
    let ggml_root = match env::var("GGML_SRC") {
        Ok(path) => normalize_source_path(PathBuf::from(path)),
        Err(_) => native_vendor_ggml(&workspace_root),
    };
    let include = ggml_root.join("include");

    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("wrapper.h").display()
    );
    println!("cargo:rerun-if-env-changed=GGML_SRC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=BLA_VENDOR");
    println!("cargo:rerun-if-env-changed=GGML_BLAS_VENDOR");
    println!("cargo:rerun-if-env-changed=CMAKE_TOOLCHAIN_FILE");
    println!("cargo:rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo:rerun-if-env-changed=OpenBLAS_ROOT");
    println!("cargo:rerun-if-env-changed=VCPKG_INSTALLATION_ROOT");
    println!("cargo:rerun-if-env-changed=VCPKG_TARGET_TRIPLET");
    rerun_cuda_env();
    rerun_openvino_env();
    rerun_rocm_env();
    validate_ggml_features(&target, CRATE_LABEL);

    if feature_enabled("vulkan") {
        println!("cargo:rerun-if-env-changed=PATH");
        if !executable_in_path("glslc") {
            println!(
                "cargo:warning=xlai-sys-ggml: Vulkan builds require a Vulkan SDK/loader and `glslc` (for example `libvulkan-dev` and `glslc` on Linux)"
            );
        }
    }

    let backends = decide_backends(&target);

    let mut cfg = cmake::Config::new(&ggml_root);
    cfg.profile("Release");
    cfg.define("BUILD_SHARED_LIBS", "OFF");
    cfg.define("GGML_STATIC", "ON");
    cfg.define("GGML_BUILD_EXAMPLES", "OFF");
    cfg.define("GGML_BUILD_TESTS", "OFF");

    if feature_enabled("native") {
        cfg.define("GGML_NATIVE", "ON");
    } else {
        cfg.define("GGML_NATIVE", "OFF");
    }

    if feature_enabled("metal") && target.contains("apple") {
        cfg.define("GGML_METAL", "ON");
        cfg.define("GGML_METAL_EMBED_LIBRARY", "ON");
    } else {
        cfg.define("GGML_METAL", "OFF");
    }

    if openblas_fe {
        if enable_linux_windows_blas {
            cfg.define("GGML_BLAS", "ON");
            cfg.define("GGML_BLAS_VENDOR", "OpenBLAS");
            cfg.define("GGML_ACCELERATE", "OFF");
            if let Some(blas_vendor) = env::var("GGML_BLAS_VENDOR")
                .ok()
                .or_else(|| env::var("BLA_VENDOR").ok())
                .filter(|value| !value.trim().is_empty())
            {
                cfg.define("BLA_VENDOR", &blas_vendor);
            }
        } else if target.contains("apple") {
            cfg.define("GGML_BLAS", "ON");
            cfg.define("GGML_ACCELERATE", "ON");
        } else {
            cfg.define("GGML_BLAS", "OFF");
            cfg.define("GGML_ACCELERATE", "OFF");
        }
    } else {
        cfg.define("GGML_BLAS", "OFF");
        cfg.define(
            "GGML_ACCELERATE",
            if target.contains("apple") {
                "ON"
            } else {
                "OFF"
            },
        );
    }

    apply_cmake_env_overrides(&mut cfg, enable_linux_windows_blas);

    cfg.define("GGML_CUDA", if backends.cuda { "ON" } else { "OFF" });
    map_feature_cmake(&mut cfg, "vulkan", "GGML_VULKAN");
    cfg.define("GGML_HIP", if backends.hip { "ON" } else { "OFF" });
    map_feature_cmake(&mut cfg, "musa", "GGML_MUSA");
    map_feature_cmake(&mut cfg, "opencl", "GGML_OPENCL");
    map_feature_cmake(&mut cfg, "rpc", "GGML_RPC");
    map_feature_cmake(&mut cfg, "sycl", "GGML_SYCL");
    map_feature_cmake(&mut cfg, "webgpu", "GGML_WEBGPU");
    cfg.define(
        "GGML_OPENVINO",
        if backends.openvino { "ON" } else { "OFF" },
    );
    map_feature_cmake(&mut cfg, "hexagon", "GGML_HEXAGON");
    map_feature_cmake(&mut cfg, "cann", "GGML_CANN");
    map_feature_cmake(&mut cfg, "zendnn", "GGML_ZENDNN");
    map_feature_cmake(&mut cfg, "zdnn", "GGML_ZDNN");
    map_feature_cmake(&mut cfg, "virtgpu", "GGML_VIRTGPU");

    let dst = cfg.build();
    let lib_dir = find_ggml_lib_dir(&dst, out_dir).unwrap_or_else(|| {
        panic!(
            "xlai-sys-ggml: could not locate static libs under cmake output {:?} or {:?}/build",
            dst, out_dir
        )
    });
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    emit_vendored_static_links(&target, backends, link_ggml_blas);
    emit_external_sdk_links(&target, backends);
    emit_system_links(&target, &target_os, enable_linux_windows_blas);

    generate_qts_ggml_bindings(&include, out_dir, manifest_dir);
}

/// Decide which accelerator backends are actually built into the vendored static core,
/// downgrading any backend whose preconditions fail with a `cargo:warning`.
fn decide_backends(target: &str) -> BackendDecisions {
    let on_apple = target.contains("apple");

    let mut decisions = BackendDecisions::default();

    if feature_enabled("cuda") {
        if on_apple {
            // Already warned by `validate_ggml_features`; nothing to add.
        } else if xlai_build_native::detect_cuda_sdk(target).is_some() {
            decisions.cuda = true;
        } else {
            println!(
                "cargo:warning={CRATE_LABEL}: `cuda` feature requested but no CUDA toolkit was found via CUDA_PATH / CUDA_HOME / standard install paths; backend will be skipped"
            );
        }
    }

    if feature_enabled("openvino") {
        if on_apple {
            // Already warned by `validate_ggml_features`.
        } else if xlai_build_native::detect_openvino_sdk(target).is_some() {
            decisions.openvino = true;
        } else {
            println!(
                "cargo:warning={CRATE_LABEL}: `openvino` feature requested but no OpenVINO runtime was found via OpenVINO_DIR / OPENVINO_ROOT / standard install paths; backend will be skipped"
            );
        }
    }

    if feature_enabled("hip") {
        let sdk_present = xlai_build_native::detect_rocm_sdk(target).is_some();
        if on_apple {
            // Already warned by `validate_ggml_features`.
        } else if STATIC_CORE {
            // Upstream `ggml` rejects `GGML_HIP=ON` together with a static-core build, so even
            // when ROCm is provisioned we cannot embed `ggml-hip` into the all-static archive.
            let detail = if sdk_present {
                "(ROCm SDK was detected, but upstream ggml does not support HIP/ROCm with static linking)"
            } else {
                "(no ROCm SDK detected and upstream ggml does not support HIP/ROCm with static linking)"
            };
            println!("cargo:warning={CRATE_LABEL}: `hip` feature ignored {detail}");
        } else if !sdk_present {
            println!(
                "cargo:warning={CRATE_LABEL}: `hip` feature requested but no ROCm SDK was found via ROCM_PATH / HIP_PATH / standard install paths; backend will be skipped"
            );
        } else {
            decisions.hip = true;
        }
    }

    decisions
}

/// Emit `cargo:rustc-link-lib=static=...` directives for the libraries produced by the vendored
/// CMake build (the "static core" half of the contract).
fn emit_vendored_static_links(target: &str, backends: BackendDecisions, link_ggml_blas: bool) {
    println!("cargo:rustc-link-lib=static=ggml");
    if feature_enabled("metal") && target.contains("apple") {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
    if backends.cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    if feature_enabled("vulkan") {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
        emit_vulkan_loader_links(target);
    }
    if backends.hip {
        println!("cargo:rustc-link-lib=static=ggml-hip");
    }
    if feature_enabled("musa") {
        println!("cargo:rustc-link-lib=static=ggml-musa");
    }
    if feature_enabled("opencl") {
        println!("cargo:rustc-link-lib=static=ggml-opencl");
    }
    if link_ggml_blas {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }
    if feature_enabled("rpc") {
        println!("cargo:rustc-link-lib=static=ggml-rpc");
    }
    if feature_enabled("sycl") {
        println!("cargo:rustc-link-lib=static=ggml-sycl");
    }
    if feature_enabled("webgpu") {
        println!("cargo:rustc-link-lib=static=ggml-webgpu");
    }
    if backends.openvino {
        println!("cargo:rustc-link-lib=static=ggml-openvino");
    }
    if feature_enabled("hexagon") {
        println!("cargo:rustc-link-lib=static=ggml-hexagon");
    }
    if feature_enabled("cann") {
        println!("cargo:rustc-link-lib=static=ggml-cann");
    }
    if feature_enabled("zendnn") {
        println!("cargo:rustc-link-lib=static=ggml-zendnn");
    }
    if feature_enabled("zdnn") {
        println!("cargo:rustc-link-lib=static=ggml-zdnn");
    }
    if feature_enabled("virtgpu") {
        println!("cargo:rustc-link-lib=static=ggml-virtgpu");
    }

    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");
}

/// Emit search paths and dynamic link directives for the **external** accelerator SDK runtimes
/// that the vendored static cores depend on (CUDA runtime, OpenVINO runtime, ROCm/HIP runtime).
fn emit_external_sdk_links(target: &str, backends: BackendDecisions) {
    if backends.cuda && !emit_cuda_link(target) {
        println!(
            "cargo:warning={CRATE_LABEL}: enabled CUDA backend but could not locate CUDA runtime libraries at link time; expect linker errors"
        );
    }
    if backends.openvino && !emit_openvino_link(target) {
        println!(
            "cargo:warning={CRATE_LABEL}: enabled OpenVINO backend but could not locate OpenVINO runtime libraries at link time; expect linker errors"
        );
    }
    if backends.hip && !emit_rocm_link(target) {
        println!(
            "cargo:warning={CRATE_LABEL}: enabled HIP backend but could not locate ROCm runtime libraries at link time; expect linker errors"
        );
    }
}

fn emit_system_links(target: &str, target_os: &str, enable_linux_windows_blas: bool) {
    if feature_enabled("metal") && target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("windows") && target.contains("msvc") {
        println!("cargo:rustc-link-lib=advapi32");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    if target_os == "linux" {
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
    }

    if enable_linux_windows_blas {
        emit_openblas_search_paths();
        println!("cargo:rustc-link-lib=openblas");
    }
}

fn generate_qts_ggml_bindings(include: &Path, out_dir: &Path, manifest_dir: &Path) {
    let wrapper_h = manifest_dir.join("wrapper.h");
    let bindings = bindgen::Builder::default()
        .header(wrapper_h.to_str().expect("wrapper.h path must be UTF-8"))
        .clang_arg(format!("-I{}", include.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("gguf_.*")
        .allowlist_var("GGML_.*")
        .allowlist_var("GGUF_.*")
        .size_t_is_usize(true)
        .generate()
        .expect("bindgen failed on ggml headers");

    let path = out_dir.join("qts_ggml_bindings.rs");
    bindings
        .write_to_file(&path)
        .expect("write qts_ggml_bindings.rs");
}
