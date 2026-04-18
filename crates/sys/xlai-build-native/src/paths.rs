//! Resolve workspace root and vendored native source locations.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Resolves the workspace root by walking ancestors of `manifest_dir` until a `Cargo.toml`
/// containing a `[workspace]` section is found (supports `crates/<pkg>` and `crates/<family>/<pkg>`).
pub fn workspace_root_from_sys_crate_manifest(manifest_dir: &Path) -> io::Result<PathBuf> {
    for ancestor in manifest_dir.ancestors() {
        let cargo_toml = ancestor.join("Cargo.toml");
        if !cargo_toml.is_file() {
            continue;
        }
        let contents = fs::read_to_string(&cargo_toml).map_err(|e| {
            io::Error::other(format!("failed to read {}: {e}", cargo_toml.display()))
        })?;
        if contents.lines().any(|line| line.trim() == "[workspace]") {
            return Ok(ancestor.to_path_buf());
        }
    }

    Err(io::Error::other(format!(
        "could not find workspace root above {}",
        manifest_dir.display()
    )))
}

pub fn native_vendor_llama_cpp(workspace_root: &Path) -> PathBuf {
    workspace_root.join("vendor/native/llama.cpp")
}

pub fn native_vendor_ggml(workspace_root: &Path) -> PathBuf {
    workspace_root.join("vendor/native/ggml")
}
