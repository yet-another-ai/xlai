#![allow(clippy::expect_used)]
#![allow(clippy::panic)]
#![allow(clippy::unwrap_used)]

use std::env;
use std::path::{Path, PathBuf};

use xlai_build_native::{
    apply_cmake_env_overrides, emit_openblas_search_paths, emit_vulkan_loader_links,
    executable_in_path, feature_enabled, find_ggml_lib_dir, map_feature_cmake, native_vendor_ggml,
    normalize_source_path, validate_ggml_features, workspace_root_from_sys_crate_manifest,
};

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
    let enable_cuda = feature_enabled("cuda") && !target.contains("apple");
    let enable_hip = feature_enabled("hip") && !target.contains("apple");
    let enable_openvino = feature_enabled("openvino") && !target.contains("apple");
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
    validate_ggml_features(&target, "xlai-sys-ggml");

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

    cfg.define("GGML_CUDA", if enable_cuda { "ON" } else { "OFF" });
    map_feature_cmake(&mut cfg, "vulkan", "GGML_VULKAN");
    cfg.define("GGML_HIP", if enable_hip { "ON" } else { "OFF" });
    map_feature_cmake(&mut cfg, "musa", "GGML_MUSA");
    map_feature_cmake(&mut cfg, "opencl", "GGML_OPENCL");
    map_feature_cmake(&mut cfg, "rpc", "GGML_RPC");
    map_feature_cmake(&mut cfg, "sycl", "GGML_SYCL");
    map_feature_cmake(&mut cfg, "webgpu", "GGML_WEBGPU");
    cfg.define("GGML_OPENVINO", if enable_openvino { "ON" } else { "OFF" });
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

    println!("cargo:rustc-link-lib=static=ggml");
    if feature_enabled("metal") && target.contains("apple") {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
    if enable_cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    if feature_enabled("vulkan") {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
        emit_vulkan_loader_links(&target);
    }
    if enable_hip {
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
    if enable_openvino {
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
