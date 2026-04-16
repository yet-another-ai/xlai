#![allow(clippy::expect_used)]
#![allow(clippy::panic)]
#![allow(clippy::unwrap_used)]

use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};

use xlai_build_native::{
    apply_cmake_env_overrides, emit_llama_vulkan_sdk_paths, emit_openblas_search_paths,
    emit_search_path_variants, feature_enabled, native_vendor_llama_cpp,
    prepare_patched_llama_source, workspace_root_from_sys_crate_manifest,
};

type BuildResult<T> = Result<T, Box<dyn Error>>;

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
}

impl BackendFeatureSet {
    fn from_cargo_features(target_os: &str) -> BuildResult<Self> {
        let feature_set = Self {
            openblas: feature_enabled("openblas"),
            metal: feature_enabled("metal"),
            vulkan: feature_enabled("vulkan"),
        };

        if feature_set.metal && target_os != "macos" {
            return Err(std::io::Error::other(
                "the `metal` Cargo feature is only supported on macOS targets",
            )
            .into());
        }

        Ok(feature_set)
    }

    fn enabled_backend_names(self) -> impl Iterator<Item = &'static str> {
        [
            self.openblas.then_some("openblas"),
            self.metal.then_some("metal"),
            self.vulkan.then_some("vulkan"),
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
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let feature_set = BackendFeatureSet::from_cargo_features(&target_os)?;
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
        .define("GGML_CUDA", "OFF")
        .define("GGML_HIP", "OFF")
        .define("GGML_SYCL", "OFF")
        .define("GGML_OPENCL", "OFF")
        .define("GGML_OPENVINO", "OFF")
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
    for library in libraries {
        println!("cargo:rustc-link-lib=static={library}");
    }

    match target_os.as_str() {
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
