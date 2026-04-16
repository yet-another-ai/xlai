//! Resolve workspace root and vendored native source locations.

use std::io;
use std::path::{Path, PathBuf};

/// `manifest_dir` must be `.../crates/<sys-crate>` (two parents = workspace root).
pub fn workspace_root_from_sys_crate_manifest(manifest_dir: &Path) -> io::Result<PathBuf> {
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .ok_or_else(|| {
            io::Error::other(format!(
                "unexpected CARGO_MANIFEST_DIR layout (expected crates/<pkg>): {}",
                manifest_dir.display()
            ))
        })
}

pub fn native_vendor_llama_cpp(workspace_root: &Path) -> PathBuf {
    workspace_root.join("vendor/native/llama.cpp")
}

pub fn native_vendor_ggml(workspace_root: &Path) -> PathBuf {
    workspace_root.join("vendor/native/ggml")
}
