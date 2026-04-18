//! Shared logic for `build.rs` in `xlai-sys-llama`, `xlai-sys-ggml`, and similar native crates.
//!
//! This crate is a normal dependency of build scripts only; it is not linked into application binaries.

#![allow(clippy::expect_used)]
#![allow(clippy::panic)]
#![allow(clippy::unwrap_used)]

mod cmake_env;
mod features;
mod ggml_build;
mod link_search;
mod llama_patch;
mod paths;

pub use cmake_env::{
    apply_cmake_env_overrides, emit_openblas_search_paths, find_openblas_include_dir,
    resolve_openblas_include_dir,
};
pub use features::feature_enabled;
pub use ggml_build::{
    executable_in_path, find_ggml_lib_dir, has_ggml_static_lib, map_feature_cmake,
    normalize_source_path, validate_ggml_features,
};
pub use link_search::{
    emit_llama_vulkan_sdk_paths, emit_search_path, emit_search_path_variants,
    emit_vulkan_loader_links, vulkan_search_dirs,
};
pub use llama_patch::prepare_patched_llama_source;
pub use paths::{
    native_vendor_ggml, native_vendor_llama_cpp, workspace_root_from_sys_crate_manifest,
};
