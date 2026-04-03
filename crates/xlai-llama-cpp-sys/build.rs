use std::env;
use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

type BuildResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, Default)]
struct BackendFeatureSet {
    openblas: bool,
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
    let enable_openblas = feature_set.openblas && matches!(target_os.as_str(), "linux" | "windows");

    println!("cargo:rerun-if-changed=build.rs");
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
        // Visual Studio's multi-config generator has been flaky for llama.cpp's
        // nested vulkan-shaders-gen ExternalProject in CI. Prefer Ninja unless
        // the caller explicitly overrides the generator.
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
        wrapper_cpp,
    );
    generate_bindings(&include_dir, &ggml_include_dir)?;

    emit_llama_search_paths(&dst, feature_set, enable_openblas);
    emit_search_path_variants(&llguidance_output_dir(&dst));
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
            openblas: env::var_os("CARGO_FEATURE_OPENBLAS").is_some(),
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

    fn suffix(self) -> &'static str {
        match (self.openblas, self.metal, self.vulkan) {
            (false, false, false) => "default",
            (true, false, false) => "openblas",
            (false, true, false) => "metal",
            (false, false, true) => "vulkan",
            (true, true, false) => "openblas-metal",
            (true, false, true) => "openblas-vulkan",
            (false, true, true) => "metal-vulkan",
            (true, true, true) => "openblas-metal-vulkan",
        }
    }
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

fn short_cmake_out_dir(feature_set: BackendFeatureSet) -> BuildResult<PathBuf> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown-target".to_string());
    let host = env::var("HOST").unwrap_or_default();
    let target_dir_ancestor = if target == host { 4 } else { 5 };
    let target_dir = out_dir
        .ancestors()
        .nth(target_dir_ancestor)
        .ok_or_else(|| {
            io::Error::other(format!("unexpected OUT_DIR layout: {}", out_dir.display()))
        })?;
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    Ok(target_dir
        .join(".xlai-cmake")
        .join(target)
        .join(profile)
        .join(feature_set.suffix()))
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

fn resolve_openblas_include_dir() -> Option<PathBuf> {
    if let Ok(openblas_root) = env::var("OpenBLAS_ROOT") {
        let include_dir = PathBuf::from(openblas_root).join("include");
        if include_dir.exists() {
            return Some(include_dir);
        }
    }

    let Ok(vcpkg_root) = env::var("VCPKG_INSTALLATION_ROOT") else {
        return None;
    };
    let Ok(vcpkg_triplet) = env::var("VCPKG_TARGET_TRIPLET") else {
        return None;
    };

    let include_dir = Path::new(&vcpkg_root)
        .join("installed")
        .join(vcpkg_triplet)
        .join("include");
    if include_dir.exists() {
        Some(include_dir)
    } else {
        None
    }
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
