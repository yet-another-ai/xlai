use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let source_dir = Path::new("vendor/llama.cpp");
    let include_dir = source_dir.join("include");
    let ggml_include_dir = source_dir.join("ggml/include");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", source_dir.display());

    let mut config = cmake::Config::new(source_dir);
    config
        .profile("Release")
        .build_target("llama")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_STATIC", "ON")
        .define("GGML_BACKEND_DL", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_WEBUI", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("GGML_BLAS", "OFF")
        .define("GGML_ACCELERATE", "OFF")
        .define("GGML_METAL", "OFF")
        .define("GGML_RPC", "OFF")
        .define("GGML_VULKAN", "OFF")
        .define("GGML_CUDA", "OFF")
        .define("GGML_HIP", "OFF")
        .define("GGML_SYCL", "OFF")
        .define("GGML_OPENCL", "OFF")
        .define("GGML_OPENVINO", "OFF")
        .define("GGML_ZDNN", "OFF")
        .define("GGML_VIRTGPU", "OFF")
        .define("GGML_WEBGPU", "OFF");

    if let Ok(generator) = env::var("CMAKE_GENERATOR") {
        config.generator(generator);
    }

    let dst = config.build();
    generate_bindings(&include_dir, &ggml_include_dir);

    emit_search_path(&dst.join("lib"));
    emit_search_path(&dst.join("lib64"));
    emit_search_path(&dst.join("build"));
    emit_search_path(&dst.join("build/src"));
    emit_search_path(&dst.join("build/ggml/src"));
    emit_search_path(&dst.join("build/bin"));

    for library in ["llama", "ggml", "ggml-base", "ggml-cpu"] {
        println!("cargo:rustc-link-lib=static={library}");
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
        }
        "linux" | "android" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
        }
        _ => {}
    }
}

fn emit_search_path(path: &Path) {
    if path.exists() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
}

fn generate_bindings(include_dir: &Path, ggml_include_dir: &Path) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let bindings = bindgen::Builder::default()
        .header(include_dir.join("llama.h").display().to_string())
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", ggml_include_dir.display()))
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("GGML_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate();

    let bindings = match bindings {
        Ok(bindings) => bindings,
        Err(error) => {
            eprintln!("failed to generate llama.cpp bindings: {error}");
            std::process::exit(1);
        }
    };

    let output = out_dir.join("bindings.rs");
    if let Err(error) = bindings.write_to_file(output) {
        eprintln!("failed to write llama.cpp bindings: {error}");
        std::process::exit(1);
    }
}
