#![allow(clippy::expect_used)]
#![allow(clippy::panic)]
#![allow(clippy::unwrap_used)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    println!("cargo:rerun-if-changed=build.rs");
    build_qts_standalone_ggml(&manifest_dir, &out_dir);
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
        .unwrap_or_else(|_| ggml_vendor_dir(manifest_dir));
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
                "cargo:warning=xlai-sys-ggml: Vulkan builds require a Vulkan SDK/loader and `glslc` (for example `libvulkan-dev` and `glslc` on Linux)"
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
            "xlai-sys-ggml: could not locate static libs under cmake output {:?} or {:?}/build",
            dst, out_dir
        )
    });
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

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
    println!("cargo:rustc-link-lib=static=ggml-base");

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

fn ggml_vendor_dir(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join("../xlai-sys/vendor/ggml")
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
        println!(
            "cargo:warning=xlai-sys-ggml: `metal` feature ignored on non-Apple target ({target})"
        );
    }
    if feature_enabled("cuda") && target.contains("apple") {
        panic!("xlai-sys-ggml: `cuda` feature is not supported on Apple targets");
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
