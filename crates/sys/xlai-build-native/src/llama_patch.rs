//! Copy vendored llama.cpp into `OUT_DIR` and apply CMake patches required for Cargo-driven builds.

use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

type BuildResult<T> = Result<T, Box<dyn Error>>;

/// Copy `vendor_source_dir` to `out_dir/llama.cpp-patched` and patch CMake files.
///
/// `patch_vulkan_external_project` should match the consuming crate's Vulkan feature:
/// build-dependency crates do not inherit `CARGO_FEATURE_*` from the parent, so the caller must pass this explicitly.
pub fn prepare_patched_llama_source(
    vendor_source_dir: &Path,
    out_dir: &Path,
    patch_vulkan_external_project: bool,
) -> BuildResult<PathBuf> {
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
    if patch_vulkan_external_project {
        patch_llama_ggml_vulkan_cmake(
            &patched_source_dir.join("ggml/src/ggml-vulkan/CMakeLists.txt"),
        )?;
    }
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

/// Vulkan shader ExternalProject: pin PREFIX/BINARY_DIR to avoid path-length and stamp issues.
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
