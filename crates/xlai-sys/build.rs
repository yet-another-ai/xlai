#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]

use std::env;
use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

type BuildResult<T> = Result<T, Box<dyn Error>>;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let want_llama = feature_enabled("llama");
    let want_qts = feature_enabled("qts-ggml");
    println!("cargo:rerun-if-changed=build.rs");

    if !want_llama && !want_qts {
        println!(
            "cargo:warning=xlai-sys: enable `llama` and/or `qts-ggml` for native builds (skipped)"
        );
        return;
    }

    if want_qts {
        build_qts_standalone_ggml(&manifest_dir, &out_dir);
    }
    if want_llama {
        build_llama_cpp_stack(&manifest_dir).expect("xlai-sys: llama.cpp build failed");
    }
}

fn build_qts_standalone_ggml(manifest_dir: &Path, out_dir: &Path) {
    let target = env::var("TARGET").expect("TARGET");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let openblas_fe = feature_enabled("openblas");
    let enable_linux_windows_blas =
        openblas_fe && matches!(target_os.as_str(), "linux" | "windows");
    let link_ggml_blas = openblas_fe && (enable_linux_windows_blas || target.contains("apple"));

    let ggml_root = env::var("GGML_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("vendor/ggml"));
    let ggml_root = normalize_source_path(ggml_root);
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
    validate_features(&target);

    if feature_enabled("vulkan") {
        println!("cargo:rerun-if-env-changed=PATH");
        if !executable_in_path("glslc") {
            println!(
                "cargo:warning=ggml-sys: Vulkan builds require a Vulkan SDK/loader and `glslc` (for example `libvulkan-dev` and `glslc` on Linux)"
            );
        }
    }

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

    // Metal is Apple-only; silently skip on other targets even if the feature is enabled.
    if feature_enabled("metal") && target.contains("apple") {
        cfg.define("GGML_METAL", "ON");
        cfg.define("GGML_METAL_EMBED_LIBRARY", "ON");
    } else {
        cfg.define("GGML_METAL", "OFF");
    }

    // Align with the llama.cpp stack in this crate: OpenBLAS on Linux/Windows, Accelerate on Apple when `openblas` is enabled.
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

    map_feature_cmake(&mut cfg, "cuda", "GGML_CUDA");
    map_feature_cmake(&mut cfg, "vulkan", "GGML_VULKAN");
    map_feature_cmake(&mut cfg, "hip", "GGML_HIP");
    map_feature_cmake(&mut cfg, "musa", "GGML_MUSA");
    map_feature_cmake(&mut cfg, "opencl", "GGML_OPENCL");
    map_feature_cmake(&mut cfg, "rpc", "GGML_RPC");
    map_feature_cmake(&mut cfg, "sycl", "GGML_SYCL");
    map_feature_cmake(&mut cfg, "webgpu", "GGML_WEBGPU");
    map_feature_cmake(&mut cfg, "openvino", "GGML_OPENVINO");
    map_feature_cmake(&mut cfg, "hexagon", "GGML_HEXAGON");
    map_feature_cmake(&mut cfg, "cann", "GGML_CANN");
    map_feature_cmake(&mut cfg, "zendnn", "GGML_ZENDNN");
    map_feature_cmake(&mut cfg, "zdnn", "GGML_ZDNN");
    map_feature_cmake(&mut cfg, "virtgpu", "GGML_VIRTGPU");

    let dst = cfg.build();
    let lib_dir = find_lib_dir(&dst, out_dir).unwrap_or_else(|| {
        panic!(
            "xlai-sys (qts-ggml): could not locate static libs under cmake output {:?} or {:?}/build",
            dst, out_dir
        )
    });
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Link order: dependents before their dependencies (GNU ld).
    println!("cargo:rustc-link-lib=static=ggml");
    if feature_enabled("metal") && target.contains("apple") {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
    if feature_enabled("cuda") {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    if feature_enabled("vulkan") {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
        emit_vulkan_loader_links(&target);
    }
    if feature_enabled("hip") {
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
    if feature_enabled("openvino") {
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
        // C++ runtime linked via ggml's MSVC build flags.
        println!("cargo:rustc-link-lib=advapi32");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    if target.contains("linux") {
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
    }

    if enable_linux_windows_blas {
        emit_openblas_search_paths();
        println!("cargo:rustc-link-lib=openblas");
    }

    generate_qts_ggml_bindings(&include, out_dir, manifest_dir);
}

fn apply_cmake_env_overrides(config: &mut cmake::Config, enable_openblas: bool) {
    if let Ok(toolchain_file) = env::var("CMAKE_TOOLCHAIN_FILE") {
        config.define("CMAKE_TOOLCHAIN_FILE", toolchain_file);
    }

    if let Ok(prefix_path) = env::var("CMAKE_PREFIX_PATH") {
        config.define("CMAKE_PREFIX_PATH", prefix_path);
    }

    if let Ok(openblas_root) = env::var("OpenBLAS_ROOT") {
        config.define("OpenBLAS_ROOT", openblas_root);
    }

    if let Ok(vcpkg_triplet) = env::var("VCPKG_TARGET_TRIPLET") {
        config.define("VCPKG_TARGET_TRIPLET", vcpkg_triplet);
    }

    if enable_openblas && let Some(include_dir) = resolve_openblas_include_dir() {
        let include_dir = include_dir.display().to_string();
        config.define("BLAS_INCLUDE_DIRS", &include_dir);
        config.define("OpenBLAS_INCLUDE_DIR", include_dir);
    }
}

fn resolve_openblas_include_dir() -> Option<PathBuf> {
    if let Ok(openblas_root) = env::var("OpenBLAS_ROOT")
        && let Some(include_dir) = find_openblas_include_dir(&PathBuf::from(openblas_root))
    {
        return Some(include_dir);
    }

    let Ok(vcpkg_root) = env::var("VCPKG_INSTALLATION_ROOT") else {
        return None;
    };
    let Ok(vcpkg_triplet) = env::var("VCPKG_TARGET_TRIPLET") else {
        return None;
    };

    find_openblas_include_dir(&Path::new(&vcpkg_root).join("installed").join(vcpkg_triplet))
}

fn find_openblas_include_dir(root: &Path) -> Option<PathBuf> {
    let candidates = [
        root.join("include"),
        root.join("include/openblas"),
        root.join("include/OpenBLAS"),
        root.join("include/openblas/include"),
    ];

    candidates
        .into_iter()
        .find(|dir| dir.join("cblas.h").exists())
}

fn emit_openblas_search_paths() {
    if let Ok(openblas_root) = env::var("OpenBLAS_ROOT") {
        let openblas_root = PathBuf::from(openblas_root);
        emit_search_path_variants(&openblas_root.join("lib"));
        emit_search_path_variants(&openblas_root.join("lib64"));
        emit_search_path_variants(&openblas_root.join("bin"));
    }

    let Ok(vcpkg_root) = env::var("VCPKG_INSTALLATION_ROOT") else {
        return;
    };
    let Ok(vcpkg_triplet) = env::var("VCPKG_TARGET_TRIPLET") else {
        return;
    };

    let installed_dir = Path::new(&vcpkg_root).join("installed").join(vcpkg_triplet);
    emit_search_path_variants(&installed_dir.join("lib"));
    emit_search_path_variants(&installed_dir.join("bin"));
}

fn emit_search_path(path: &Path) {
    if path.exists() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
}

fn emit_search_path_variants(path: &Path) {
    emit_search_path(path);

    for config in ["Release", "RelWithDebInfo", "Debug", "MinSizeRel"] {
        emit_search_path(&path.join(config));
    }
}

fn executable_in_path(name: &str) -> bool {
    let Some(path_var) = env::var_os("PATH") else {
        return false;
    };
    for dir in env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if fs::metadata(&candidate)
            .map(|m| m.is_file())
            .unwrap_or(false)
        {
            return true;
        }
        #[cfg(windows)]
        {
            let candidate_exe = dir.join(format!("{name}.exe"));
            if fs::metadata(&candidate_exe)
                .map(|m| m.is_file())
                .unwrap_or(false)
            {
                return true;
            }
        }
    }
    false
}

fn feature_enabled(name: &str) -> bool {
    env::var(format!(
        "CARGO_FEATURE_{}",
        name.to_ascii_uppercase().replace('-', "_")
    ))
    .is_ok()
}

fn normalize_source_path(path: PathBuf) -> PathBuf {
    let path = path
        .canonicalize()
        .unwrap_or_else(|e| panic!("ggml source path missing or invalid ({path:?}): {e}"));

    #[cfg(windows)]
    {
        if let Some(stripped) = path
            .to_str()
            .and_then(|raw| raw.strip_prefix(r"\\?\"))
            .map(PathBuf::from)
        {
            return stripped;
        }
    }

    path
}

fn map_feature_cmake(cfg: &mut cmake::Config, feature: &str, cmake_opt: &str) {
    if feature_enabled(feature) {
        cfg.define(cmake_opt, "ON");
    }
}

fn validate_features(target: &str) {
    if feature_enabled("metal") && !target.contains("apple") {
        println!("cargo:warning=xlai-sys: `metal` feature ignored on non-Apple target ({target})");
    }
    if feature_enabled("cuda") && target.contains("apple") {
        panic!("xlai-sys: `cuda` feature is not supported on Apple targets");
    }
}

fn emit_vulkan_loader_links(target: &str) {
    for dir in vulkan_search_dirs(target) {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }

    let lib = if target.contains("apple") {
        "dylib=vulkan"
    } else if target.contains("windows") {
        "vulkan-1"
    } else {
        "vulkan"
    };

    println!("cargo:rustc-link-lib={lib}");
}

fn vulkan_search_dirs(target: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let sdk = PathBuf::from(sdk);
        if target.contains("windows") {
            dirs.push(sdk.join("Lib"));
        } else {
            dirs.push(sdk.join("lib"));
            if target.contains("apple") {
                dirs.push(sdk.join("macOS").join("lib"));
            }
        }
    }
    if target.contains("apple") {
        dirs.push(PathBuf::from("/opt/homebrew/lib"));
        dirs.push(PathBuf::from("/usr/local/lib"));
    }
    dirs
}

fn find_lib_dir(dst: &Path, out_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        dst.join("src"),
        dst.join("lib"),
        dst.to_path_buf(),
        out_dir.join("build").join("src"),
        out_dir.join("build").join("lib"),
        out_dir.join("build").join("Release").join("src"),
        out_dir.join("build").join("Debug").join("src"),
    ];
    candidates.into_iter().find(|p| has_ggml(p))
}

fn has_ggml(dir: &Path) -> bool {
    dir.join("libggml.a").exists()
        || dir.join("libggml.lib").exists()
        || dir.join("ggml.lib").exists()
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

// --- llama.cpp (from former xlai-llama-cpp-sys) ---

#[derive(Clone, Copy, Debug, Default)]
struct BackendFeatureSet {
    openblas: bool,
    metal: bool,
    vulkan: bool,
}

fn build_llama_cpp_stack(manifest_dir: &Path) -> BuildResult<()> {
    let vendor_source_dir = manifest_dir.join("vendor/llama.cpp");
    let source_dir = prepare_llama_source(&vendor_source_dir)?;
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
        &source_dir.join("ggml/src"),
        &wrapper_cpp,
    );
    generate_llama_bindings(&include_dir, &ggml_include_dir)?;

    emit_llama_search_paths(&dst, feature_set, enable_openblas);
    emit_search_path_variants(&llguidance_output_dir(&dst));
    emit_llama_vulkan_sdk_paths(feature_set.vulkan, &target_os);
    emit_search_path_variants(&dst.join("build/bin"));

    let mut libraries = vec![
        "xlai_llama_cpp_wrapper",
        "common",
        "cpp-httplib",
        "llguidance",
        "llama",
        "ggml",
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

impl BackendFeatureSet {
    fn from_cargo_features(target_os: &str) -> BuildResult<Self> {
        let feature_set = Self {
            openblas: feature_enabled("openblas"),
            metal: feature_enabled("metal"),
            vulkan: feature_enabled("vulkan"),
        };

        if feature_set.metal && target_os != "macos" {
            return Err(io::Error::other(
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

fn short_cmake_out_dir(feature_set: BackendFeatureSet) -> BuildResult<PathBuf> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| {
            io::Error::other(format!(
                "unexpected CARGO_MANIFEST_DIR layout: {}",
                manifest_dir.display()
            ))
        })?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "unknown".to_string());
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_else(|_| "unknown".to_string());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    Ok(workspace_root.join(".xlcm").join(format!(
        "{target_arch}-{target_env}-{profile}-{}",
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

fn emit_llama_vulkan_sdk_paths(enable_vulkan: bool, target_os: &str) {
    if !enable_vulkan {
        return;
    }

    let Some(vulkan_sdk) = env::var_os("VULKAN_SDK") else {
        return;
    };
    let vulkan_sdk = PathBuf::from(vulkan_sdk);

    match target_os {
        "windows" => {
            emit_search_path(&vulkan_sdk.join("Lib"));
        }
        "macos" | "ios" | "linux" | "android" => {
            emit_search_path(&vulkan_sdk.join("lib"));
            emit_search_path(&vulkan_sdk.join("macOS/lib"));
            emit_search_path(&vulkan_sdk.join("Lib"));
        }
        _ => {}
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

fn prepare_llama_source(vendor_source_dir: &Path) -> BuildResult<PathBuf> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let patched_source_dir = out_dir.join("llama.cpp-patched");

    if patched_source_dir.exists() {
        fs::remove_dir_all(&patched_source_dir).map_err(|error| {
            io::Error::other(format!(
                "failed to clear patched llama.cpp source directory `{}`: {error}",
                patched_source_dir.display()
            ))
        })?;
    }

    copy_dir_all(vendor_source_dir, &patched_source_dir).map_err(|error| {
        io::Error::other(format!(
            "failed to copy llama.cpp sources from `{}` to `{}`: {error}",
            vendor_source_dir.display(),
            patched_source_dir.display()
        ))
    })?;

    patch_llama_common_cmake(&patched_source_dir.join("common/CMakeLists.txt"))?;
    patch_llama_ggml_vulkan_cmake(&patched_source_dir.join("ggml/src/ggml-vulkan/CMakeLists.txt"))?;
    Ok(patched_source_dir)
}

fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let file_name = entry.file_name();

        if file_name == OsStr::new(".git") {
            continue;
        }

        let destination = dst.join(entry.file_name());
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            copy_dir_all(&path, &destination)?;
        } else if file_type.is_symlink() {
            copy_symlink(&path, &destination)?;
        } else {
            fs::copy(&path, &destination)?;
        }
    }

    Ok(())
}

#[cfg(unix)]
fn copy_symlink(src: &Path, dst: &Path) -> io::Result<()> {
    use std::os::unix::fs::symlink;

    let target = fs::read_link(src)?;
    symlink(target, dst)
}

#[cfg(windows)]
fn copy_symlink(src: &Path, dst: &Path) -> io::Result<()> {
    use std::os::windows::fs::{symlink_dir, symlink_file};

    let target = fs::read_link(src)?;
    if src.is_dir() {
        symlink_dir(target, dst)
    } else {
        symlink_file(target, dst)
    }
}

#[cfg(not(any(unix, windows)))]
fn copy_symlink(src: &Path, dst: &Path) -> io::Result<()> {
    let metadata = fs::metadata(src)?;
    if metadata.is_dir() {
        copy_dir_all(src, dst)
    } else {
        fs::copy(src, dst).map(|_| ())
    }
}

fn patch_llama_common_cmake(path: &Path) -> BuildResult<()> {
    let mut contents = fs::read_to_string(path).map_err(|error| {
        io::Error::other(format!(
            "failed to read patched llama.cpp CMake file `{}`: {error}",
            path.display()
        ))
    })?;
    contents = normalize_newlines(&contents);

    contents = replace_once(
        &contents,
        "    set(LLGUIDANCE_PATH ${LLGUIDANCE_SRC}/target/release)\n    set(LLGUIDANCE_LIB_NAME \"${CMAKE_STATIC_LIBRARY_PREFIX}llguidance${CMAKE_STATIC_LIBRARY_SUFFIX}\")\n",
        "    set(LLGUIDANCE_LIB_NAME \"${CMAKE_STATIC_LIBRARY_PREFIX}llguidance${CMAKE_STATIC_LIBRARY_SUFFIX}\")\n\n    if (DEFINED ENV{CARGO_TARGET_DIR} AND NOT \"$ENV{CARGO_TARGET_DIR}\" STREQUAL \"\")\n        set(LLGUIDANCE_TARGET_DIR $ENV{CARGO_TARGET_DIR})\n    else()\n        set(LLGUIDANCE_TARGET_DIR ${LLGUIDANCE_SRC}/target)\n    endif()\n\n    if (DEFINED ENV{TARGET} AND NOT \"$ENV{TARGET}\" STREQUAL \"\")\n        set(LLGUIDANCE_PATH ${LLGUIDANCE_TARGET_DIR}/$ENV{TARGET}/release)\n        set(LLGUIDANCE_CARGO_TARGET_ARGS --target $ENV{TARGET})\n    else()\n        set(LLGUIDANCE_PATH ${LLGUIDANCE_TARGET_DIR}/release)\n        set(LLGUIDANCE_CARGO_TARGET_ARGS)\n    endif()\n",
        path,
    )?;

    contents = replace_once(
        &contents,
        "    add_dependencies(llguidance llguidance_ext)\n\n    target_include_directories(${TARGET} PRIVATE ${LLGUIDANCE_PATH})\n",
        "    add_dependencies(llguidance llguidance_ext)\n    add_dependencies(${TARGET} llguidance_ext)\n\n    target_include_directories(${TARGET} PRIVATE ${LLGUIDANCE_PATH})\n",
        path,
    )?;

    contents = replace_once(
        &contents,
        "        BUILD_COMMAND cargo build --release --package llguidance\n",
        "        BUILD_COMMAND ${CMAKE_COMMAND} -E env --unset=RUSTC --unset=RUSTC_WRAPPER --unset=RUSTC_WORKSPACE_WRAPPER --unset=CLIPPY_ARGS --unset=RUSTFLAGS --unset=CARGO_ENCODED_RUSTFLAGS --unset=CARGO_BUILD_RUSTFLAGS cargo build --release --package llguidance ${LLGUIDANCE_CARGO_TARGET_ARGS}\n",
        path,
    )?;

    fs::write(path, contents).map_err(|error| {
        io::Error::other(format!(
            "failed to write patched llama.cpp CMake file `{}`: {error}",
            path.display()
        ))
    })?;

    Ok(())
}

fn patch_llama_ggml_vulkan_cmake(path: &Path) -> BuildResult<()> {
    let mut contents = fs::read_to_string(path).map_err(|error| {
        io::Error::other(format!(
            "failed to read patched llama.cpp Vulkan CMake file `{}`: {error}",
            path.display()
        ))
    })?;
    contents = normalize_newlines(&contents);

    contents = replace_once(
        &contents,
        "    ExternalProject_Add(\n        vulkan-shaders-gen\n        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders\n",
        "    ExternalProject_Add(\n        vulkan-shaders-gen\n        PREFIX ${CMAKE_BINARY_DIR}/vkgen\n        TMP_DIR ${CMAKE_BINARY_DIR}/vkgen-tmp\n        STAMP_DIR ${CMAKE_BINARY_DIR}/vkgen-stamp\n        BINARY_DIR ${CMAKE_BINARY_DIR}/vkgen-build\n        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders\n",
        path,
    )?;

    fs::write(path, contents).map_err(|error| {
        io::Error::other(format!(
            "failed to write patched llama.cpp Vulkan CMake file `{}`: {error}",
            path.display()
        ))
    })?;

    Ok(())
}

fn normalize_newlines(contents: &str) -> String {
    contents.replace("\r\n", "\n").replace('\r', "\n")
}

fn replace_once(contents: &str, from: &str, to: &str, path: &Path) -> BuildResult<String> {
    if !contents.contains(from) {
        return Err(io::Error::other(format!(
            "expected to patch `{}` but the target snippet was not found; upstream llama.cpp layout may have changed",
            path.display()
        ))
        .into());
    }

    Ok(contents.replacen(from, to, 1))
}

fn build_wrapper(
    include_dir: &Path,
    common_dir: &Path,
    vendor_dir: &Path,
    ggml_include_dir: &Path,
    ggml_src_dir: &Path,
    wrapper_cpp: &Path,
) {
    cc::Build::new()
        .cpp(true)
        .file(wrapper_cpp)
        // Keep the meta backend implementation bundled with our wrapper objects so
        // final Rust links do not depend on extracting it from `libggml-base.a`.
        .file(ggml_src_dir.join("ggml-backend-meta.cpp"))
        .include(include_dir)
        .include(common_dir)
        .include(vendor_dir)
        .include(ggml_include_dir)
        .include(ggml_src_dir)
        .std("c++17")
        .compile("xlai_llama_cpp_wrapper");
}

fn generate_llama_bindings(include_dir: &Path, ggml_include_dir: &Path) -> BuildResult<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let bindings = bindgen::Builder::default()
        .header(include_dir.join("llama.h").display().to_string())
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", ggml_include_dir.display()))
        // Do not emit `ggml_*` here: when `qts-ggml` is also enabled, `qts_ggml` provides those
        // symbols and duplicate `extern "C"` items would clash at crate scope.
        .allowlist_type("llama_.*")
        .allowlist_function("llama_.*")
        .allowlist_var("LLAMA_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .map_err(|error| {
            io::Error::other(format!("failed to generate llama.cpp bindings: {error}"))
        })?;

    let output = out_dir.join("llama_bindings.rs");
    bindings.write_to_file(output).map_err(|error| {
        io::Error::other(format!("failed to write llama.cpp bindings: {error}"))
    })?;

    Ok(())
}
