use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

type BuildResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, Default)]
struct BackendFeatureSet {
    metal: bool,
    vulkan: bool,
}

fn main() -> BuildResult<()> {
    let vendor_source_dir = Path::new("vendor/llama.cpp");
    let source_dir = prepare_llama_source(vendor_source_dir)?;
    let include_dir = source_dir.join("include");
    let common_dir = source_dir.join("common");
    let vendor_dir = source_dir.join("vendor");
    let ggml_include_dir = source_dir.join("ggml/include");
    let wrapper_cpp = Path::new("src/wrapper.cpp");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let feature_set = BackendFeatureSet::from_cargo_features(&target_os)?;
    let enable_accelerate = target_os == "macos";

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", vendor_source_dir.display());
    println!("cargo:rerun-if-changed={}", wrapper_cpp.display());
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

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
        .define("GGML_BLAS", "OFF")
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

    if let Ok(generator) = env::var("CMAKE_GENERATOR") {
        config.generator(generator);
    }

    let dst = config.build();
    build_wrapper(
        &include_dir,
        &common_dir,
        &vendor_dir,
        &ggml_include_dir,
        wrapper_cpp,
    );
    generate_bindings(&include_dir, &ggml_include_dir)?;

    emit_search_path_variants(&dst.join("lib"));
    emit_search_path_variants(&dst.join("lib64"));
    emit_search_path_variants(&dst.join("build"));
    emit_search_path_variants(&dst.join("build/common"));
    emit_search_path_variants(&dst.join("build/src"));
    emit_search_path_variants(&dst.join("build/ggml/src"));
    emit_search_path_variants(&dst.join("build/ggml/src/ggml-metal"));
    emit_search_path_variants(&dst.join("build/ggml/src/ggml-vulkan"));
    emit_search_path_variants(&dst.join("build/vendor/cpp-httplib"));
    if let Ok(cargo_target_dir) = env::var("CARGO_TARGET_DIR") {
        emit_search_path(Path::new(&cargo_target_dir).join("release").as_path());
    } else {
        emit_search_path_variants(&dst.join("build/llguidance/source/target/release"));
    }
    emit_vulkan_search_paths(feature_set.vulkan, &target_os);
    emit_search_path_variants(&dst.join("build/bin"));

    let mut libraries = vec![
        "xlai_llama_cpp_wrapper",
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
            if feature_set.vulkan {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }
        "windows" => {
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
            metal: env::var_os("CARGO_FEATURE_METAL").is_some(),
            vulkan: env::var_os("CARGO_FEATURE_VULKAN").is_some(),
        };

        if feature_set.metal && target_os != "macos" {
            return Err(io::Error::other(
                "the `metal` Cargo feature is only supported on macOS targets",
            )
            .into());
        }

        Ok(feature_set)
    }
}

fn emit_vulkan_search_paths(enable_vulkan: bool, target_os: &str) {
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
    Ok(patched_source_dir)
}

fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
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
        "    set(LLGUIDANCE_LIB_NAME \"${CMAKE_STATIC_LIBRARY_PREFIX}llguidance${CMAKE_STATIC_LIBRARY_SUFFIX}\")\n\n    if (DEFINED ENV{CARGO_TARGET_DIR})\n        set(LLGUIDANCE_PATH $ENV{CARGO_TARGET_DIR}/release)\n    else()\n        set(LLGUIDANCE_PATH ${LLGUIDANCE_SRC}/target/release)\n    endif()\n",
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
        "        BUILD_COMMAND ${CMAKE_COMMAND} -E env --unset=RUSTC --unset=RUSTC_WRAPPER --unset=RUSTC_WORKSPACE_WRAPPER --unset=CLIPPY_ARGS --unset=RUSTFLAGS --unset=CARGO_ENCODED_RUSTFLAGS --unset=CARGO_BUILD_RUSTFLAGS cargo build --release --package llguidance\n",
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

fn generate_bindings(include_dir: &Path, ggml_include_dir: &Path) -> BuildResult<()> {
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
        .generate()
        .map_err(|error| {
            io::Error::other(format!("failed to generate llama.cpp bindings: {error}"))
        })?;

    let output = out_dir.join("bindings.rs");
    bindings.write_to_file(output).map_err(|error| {
        io::Error::other(format!("failed to write llama.cpp bindings: {error}"))
    })?;

    Ok(())
}
