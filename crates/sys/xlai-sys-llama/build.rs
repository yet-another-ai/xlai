#![allow(clippy::expect_used)]
#![allow(clippy::panic)]
#![allow(clippy::unwrap_used)]

use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};

use xlai_build_native::{
    apply_cmake_env_overrides, detect_cuda_sdk, detect_openvino_sdk, detect_rocm_sdk,
    emit_cuda_link, emit_llama_vulkan_sdk_paths, emit_openblas_search_paths, emit_openvino_link,
    emit_rocm_link, emit_search_path_variants, feature_enabled, native_vendor_llama_cpp,
    prepare_patched_llama_source, rerun_cuda_env, rerun_openvino_env, rerun_rocm_env,
    workspace_root_from_sys_crate_manifest,
};

type BuildResult<T> = Result<T, Box<dyn Error>>;

const CRATE_LABEL: &str = "xlai-sys-llama";

/// Linking model for the vendored llama.cpp / GGML core in this crate.
///
/// `xlai-sys-llama` builds the vendored `llama.cpp` and bundled `ggml` statically (`BUILD_SHARED_LIBS=OFF`,
/// `GGML_STATIC=ON`, `GGML_BACKEND_DL=OFF`). External accelerator SDK runtime libraries are still
/// linked dynamically; see [`docs/development/native-vendor.md`](../../../../docs/development/native-vendor.md).
const STATIC_CORE: bool = true;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    println!("cargo:rerun-if-changed=build.rs");
    build_llama_cpp_stack(&manifest_dir).expect("xlai-sys-llama: llama.cpp build failed");
}

#[derive(Clone, Copy, Debug, Default)]
struct BackendFeatureSet {
    openblas: bool,
    metal: bool,
    vulkan: bool,
    cuda: bool,
    hip: bool,
    openvino: bool,
}

impl BackendFeatureSet {
    /// Decide which backends to actually request from CMake, downgrading any backend whose
    /// preconditions fail with a `cargo:warning`. Gating is driven by:
    ///
    /// 1. **Target OS / platform support** (`metal` is Apple-only; `cuda` / `hip` / `openvino`
    ///    are not built on macOS).
    /// 2. **Upstream backend constraints** (`ggml-hip` cannot be embedded in a static-core build).
    /// 3. **External SDK presence** for CUDA / OpenVINO / ROCm via [`detect_cuda_sdk`] etc.
    fn from_cargo_features(target: &str, target_os: &str) -> BuildResult<Self> {
        let on_apple = target_os == "macos" || target_os == "ios";

        let requested_metal = feature_enabled("metal");
        let requested_cuda = feature_enabled("cuda");
        let requested_hip = feature_enabled("hip");
        let requested_openvino = feature_enabled("openvino");

        if requested_metal && !on_apple {
            return Err(std::io::Error::other(
                "the `metal` Cargo feature is only supported on macOS targets",
            )
            .into());
        }
        if requested_cuda && on_apple {
            println!("cargo:warning={CRATE_LABEL}: `cuda` feature ignored on macOS targets");
        }
        if requested_hip && on_apple {
            println!("cargo:warning={CRATE_LABEL}: `hip` feature ignored on macOS targets");
        }
        if requested_openvino && on_apple {
            println!("cargo:warning={CRATE_LABEL}: `openvino` feature ignored on macOS targets");
        }

        let cuda = requested_cuda
            && !on_apple
            && match detect_cuda_sdk(target) {
                Some(_) => true,
                None => {
                    println!(
                        "cargo:warning={CRATE_LABEL}: `cuda` feature requested but no CUDA toolkit was found via CUDA_PATH / CUDA_HOME / standard install paths; backend will be skipped"
                    );
                    false
                }
            };

        let openvino = requested_openvino
            && !on_apple
            && match detect_openvino_sdk(target) {
                Some(_) => true,
                None => {
                    println!(
                        "cargo:warning={CRATE_LABEL}: `openvino` feature requested but no OpenVINO runtime was found via OpenVINO_DIR / OPENVINO_ROOT / standard install paths; backend will be skipped"
                    );
                    false
                }
            };

        let hip = if requested_hip && !on_apple {
            let sdk_present = detect_rocm_sdk(target).is_some();
            if STATIC_CORE {
                let detail = if sdk_present {
                    "(ROCm SDK was detected, but upstream ggml does not support HIP/ROCm with static linking)"
                } else {
                    "(no ROCm SDK detected and upstream ggml does not support HIP/ROCm with static linking)"
                };
                println!("cargo:warning={CRATE_LABEL}: `hip` feature ignored {detail}");
                false
            } else if !sdk_present {
                println!(
                    "cargo:warning={CRATE_LABEL}: `hip` feature requested but no ROCm SDK was found via ROCM_PATH / HIP_PATH / standard install paths; backend will be skipped"
                );
                false
            } else {
                true
            }
        } else {
            false
        };

        Ok(Self {
            openblas: feature_enabled("openblas"),
            metal: requested_metal && on_apple,
            vulkan: feature_enabled("vulkan"),
            cuda,
            hip,
            openvino,
        })
    }

    fn enabled_backend_names(self) -> impl Iterator<Item = &'static str> {
        [
            self.openblas.then_some("openblas"),
            self.metal.then_some("metal"),
            self.vulkan.then_some("vulkan"),
            self.cuda.then_some("cuda"),
            self.hip.then_some("hip"),
            self.openvino.then_some("openvino"),
        ]
        .into_iter()
        .flatten()
    }

    fn suffix(self) -> String {
        let mut suffix = String::new();

        for backend in self.enabled_backend_names() {
            if !suffix.is_empty() {
                suffix.push('-');
            }
            suffix.push_str(backend);
        }

        if suffix.is_empty() {
            suffix.push_str("default");
        }

        suffix
    }
}

fn build_llama_cpp_stack(manifest_dir: &Path) -> BuildResult<()> {
    let workspace_root = workspace_root_from_sys_crate_manifest(manifest_dir)?;
    let vendor_source_dir = env::var("LLAMA_CPP_SRC")
        .map(PathBuf::from)
        .map(xlai_build_native::normalize_source_path)
        .unwrap_or_else(|_| native_vendor_llama_cpp(&workspace_root));

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let source_dir =
        prepare_patched_llama_source(&vendor_source_dir, &out_dir, feature_enabled("vulkan"))?;
    let include_dir = source_dir.join("include");
    let common_dir = source_dir.join("common");
    let vendor_dir = source_dir.join("vendor");
    let ggml_include_dir = source_dir.join("ggml/include");
    let wrapper_cpp = manifest_dir.join("src/wrapper.cpp");
    let target = env::var("TARGET").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let feature_set = BackendFeatureSet::from_cargo_features(&target, &target_os)?;
    let enable_accelerate = target_os == "macos";
    let enable_openblas = feature_set.openblas && matches!(target_os.as_str(), "linux" | "windows");

    println!("cargo:rerun-if-changed={}", vendor_source_dir.display());
    println!("cargo:rerun-if-changed={}", wrapper_cpp.display());
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_SRC");
    println!("cargo:rerun-if-env-changed=CMAKE_GENERATOR");
    println!("cargo:rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo:rerun-if-env-changed=CMAKE_TOOLCHAIN_FILE");
    println!("cargo:rerun-if-env-changed=CARGO_TARGET_DIR");
    println!("cargo:rerun-if-env-changed=OpenBLAS_ROOT");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=VCPKG_INSTALLATION_ROOT");
    println!("cargo:rerun-if-env-changed=VCPKG_TARGET_TRIPLET");
    rerun_cuda_env();
    rerun_openvino_env();
    rerun_rocm_env();

    let mut config = cmake::Config::new(&source_dir);
    config
        .profile("Release")
        .build_target("common")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_STATIC", "ON")
        .define("GGML_BACKEND_DL", "OFF")
        .define("LLAMA_BUILD_COMMON", "ON")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_WEBUI", "OFF")
        .define("LLAMA_OPENSSL", "OFF")
        .define("LLAMA_LLGUIDANCE", "ON")
        .define("GGML_OPENMP", "OFF")
        .define("GGML_BLAS", if enable_openblas { "ON" } else { "OFF" })
        .define(
            "GGML_BLAS_VENDOR",
            if enable_openblas { "OpenBLAS" } else { "" },
        )
        .define(
            "GGML_ACCELERATE",
            if enable_accelerate { "ON" } else { "OFF" },
        )
        .define("GGML_METAL", if feature_set.metal { "ON" } else { "OFF" })
        .define("GGML_RPC", "OFF")
        .define("GGML_VULKAN", if feature_set.vulkan { "ON" } else { "OFF" })
        .define("GGML_CUDA", if feature_set.cuda { "ON" } else { "OFF" })
        .define("GGML_HIP", if feature_set.hip { "ON" } else { "OFF" })
        .define("GGML_SYCL", "OFF")
        .define("GGML_OPENCL", "OFF")
        .define(
            "GGML_OPENVINO",
            if feature_set.openvino { "ON" } else { "OFF" },
        )
        .define("GGML_ZDNN", "OFF")
        .define("GGML_VIRTGPU", "OFF")
        .define("GGML_WEBGPU", "OFF");

    if target_os == "windows" {
        config.out_dir(short_cmake_out_dir(feature_set)?);
    }

    apply_cmake_env_overrides(&mut config, enable_openblas);

    if let Ok(generator) = env::var("CMAKE_GENERATOR") {
        config.generator(generator);
    } else if target_os == "windows" {
        config.generator("Ninja");
    }

    if target_os == "windows" {
        config.configure_arg("-Wno-dev");
        config.build_arg("-v");
    }

    let dst = config.build();
    build_wrapper(
        &include_dir,
        &common_dir,
        &vendor_dir,
        &ggml_include_dir,
        &wrapper_cpp,
    );
    generate_llama_bindings(&include_dir, &ggml_include_dir)?;

    emit_llama_search_paths(&dst, feature_set, enable_openblas);
    emit_search_path_variants(&llguidance_output_dir(&dst));
    emit_llama_vulkan_sdk_paths(feature_set.vulkan, &target_os);
    emit_search_path_variants(&dst.join("build/bin"));

    emit_vendored_static_libs(feature_set, enable_openblas);
    emit_external_sdk_links(&target, feature_set);
    emit_system_links(
        feature_set,
        &target_os,
        &target_env,
        enable_accelerate,
        enable_openblas,
    );

    Ok(())
}

fn short_cmake_out_dir(feature_set: BackendFeatureSet) -> BuildResult<PathBuf> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let workspace_root = workspace_root_from_sys_crate_manifest(&manifest_dir)?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "unknown".to_string());
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_else(|_| "unknown".to_string());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    Ok(workspace_root.join(".xlcm").join(format!(
        "{target_arch}-{target_env}-{profile}-llama-{}",
        feature_set.suffix()
    )))
}

fn emit_llama_search_paths(dst: &Path, feature_set: BackendFeatureSet, enable_openblas: bool) {
    emit_search_path_variants(&dst.join("lib"));
    emit_search_path_variants(&dst.join("lib64"));
    emit_search_path_variants(&dst.join("build"));
    emit_search_path_variants(&dst.join("build/common"));
    emit_search_path_variants(&dst.join("build/src"));
    emit_search_path_variants(&dst.join("build/ggml/src"));
    emit_search_path_variants(&dst.join("build/vendor/cpp-httplib"));
    let ggml_src_dir = dst.join("build/ggml/src");

    if enable_openblas {
        emit_search_path_variants(&ggml_src_dir.join("ggml-blas"));
    }
    if feature_set.metal {
        emit_search_path_variants(&ggml_src_dir.join("ggml-metal"));
    }
    if feature_set.vulkan {
        emit_search_path_variants(&ggml_src_dir.join("ggml-vulkan"));
    }
    if feature_set.cuda {
        emit_search_path_variants(&ggml_src_dir.join("ggml-cuda"));
    }
    if feature_set.hip {
        emit_search_path_variants(&ggml_src_dir.join("ggml-hip"));
    }
    if feature_set.openvino {
        emit_search_path_variants(&ggml_src_dir.join("ggml-openvino"));
    }
}

fn llguidance_output_dir(dst: &Path) -> PathBuf {
    let target_root = if let Ok(cargo_target_dir) = env::var("CARGO_TARGET_DIR") {
        PathBuf::from(cargo_target_dir)
    } else {
        dst.join("build/llguidance/source/target")
    };

    if let Ok(target_triple) = env::var("TARGET") {
        target_root.join(target_triple).join("release")
    } else {
        target_root.join("release")
    }
}

fn build_wrapper(
    include_dir: &Path,
    common_dir: &Path,
    vendor_dir: &Path,
    ggml_include_dir: &Path,
    wrapper_cpp: &Path,
) {
    cc::Build::new()
        .cpp(true)
        .file(wrapper_cpp)
        .include(include_dir)
        .include(common_dir)
        .include(vendor_dir)
        .include(ggml_include_dir)
        .std("c++17")
        .compile("xlai_llama_cpp_wrapper");
}

fn generate_llama_bindings(include_dir: &Path, ggml_include_dir: &Path) -> BuildResult<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let bindings = bindgen::Builder::default()
        .header(include_dir.join("llama.h").display().to_string())
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", ggml_include_dir.display()))
        .allowlist_type("llama_.*")
        .allowlist_function("llama_.*")
        .allowlist_var("LLAMA_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .map_err(|error| {
            std::io::Error::other(format!("failed to generate llama.cpp bindings: {error}"))
        })?;

    let output = out_dir.join("llama_bindings.rs");
    bindings.write_to_file(output).map_err(|error| {
        std::io::Error::other(format!("failed to write llama.cpp bindings: {error}"))
    })?;

    Ok(())
}

/// Emit `cargo:rustc-link-lib=static=...` directives for the vendored static-core libraries
/// produced by the cmake build above.
fn emit_vendored_static_libs(feature_set: BackendFeatureSet, enable_openblas: bool) {
    let mut libraries = vec![
        "common",
        "cpp-httplib",
        "llguidance",
        "llama",
        "ggml",
        "ggml-base",
        "ggml-cpu",
    ];
    if feature_set.metal {
        libraries.push("ggml-metal");
    }
    if enable_openblas {
        libraries.push("ggml-blas");
    }
    if feature_set.vulkan {
        libraries.push("ggml-vulkan");
    }
    if feature_set.cuda {
        libraries.push("ggml-cuda");
    }
    if feature_set.hip {
        libraries.push("ggml-hip");
    }
    if feature_set.openvino {
        libraries.push("ggml-openvino");
    }
    for library in libraries {
        println!("cargo:rustc-link-lib=static={library}");
    }
}

/// Emit search paths and dynamic link directives for the **external** accelerator SDK runtime
/// libraries that the vendored static cores depend on at link time.
fn emit_external_sdk_links(target: &str, feature_set: BackendFeatureSet) {
    if feature_set.cuda && !emit_cuda_link(target) {
        println!(
            "cargo:warning={CRATE_LABEL}: enabled CUDA backend but could not locate CUDA runtime libraries at link time; expect linker errors"
        );
    }
    if feature_set.openvino && !emit_openvino_link(target) {
        println!(
            "cargo:warning={CRATE_LABEL}: enabled OpenVINO backend but could not locate OpenVINO runtime libraries at link time; expect linker errors"
        );
    }
    if feature_set.hip && !emit_rocm_link(target) {
        println!(
            "cargo:warning={CRATE_LABEL}: enabled HIP backend but could not locate ROCm runtime libraries at link time; expect linker errors"
        );
    }
}

fn emit_system_links(
    feature_set: BackendFeatureSet,
    target_os: &str,
    target_env: &str,
    enable_accelerate: bool,
    enable_openblas: bool,
) {
    match target_os {
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
            if enable_accelerate {
                println!("cargo:rustc-link-lib=framework=Accelerate");
            }
            if feature_set.metal {
                println!("cargo:rustc-link-lib=framework=Foundation");
                println!("cargo:rustc-link-lib=framework=Metal");
                println!("cargo:rustc-link-lib=framework=MetalKit");
            }
            if feature_set.vulkan {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }
        "linux" | "android" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
            if enable_openblas {
                println!("cargo:rustc-link-lib=openblas");
            }
            if feature_set.vulkan {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }
        "windows" => {
            if enable_openblas {
                emit_openblas_search_paths();
                println!("cargo:rustc-link-lib=openblas");
            }
            if feature_set.vulkan {
                let library_name = if target_env == "msvc" {
                    "vulkan-1"
                } else {
                    "vulkan"
                };
                println!("cargo:rustc-link-lib={library_name}");
            }
        }
        _ => {}
    }
}
