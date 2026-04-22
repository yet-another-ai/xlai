use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::features::feature_enabled;

#[must_use]
pub fn executable_in_path(name: &str) -> bool {
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

pub fn normalize_source_path(path: PathBuf) -> PathBuf {
    let path = path
        .canonicalize()
        .unwrap_or_else(|e| panic!("source path missing or invalid ({path:?}): {e}"));

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

pub fn map_feature_cmake(cfg: &mut cmake::Config, feature: &str, cmake_opt: &str) {
    if feature_enabled(feature) {
        cfg.define(cmake_opt, "ON");
    }
}

pub fn validate_ggml_features(target: &str, crate_label: &str) {
    if feature_enabled("metal") && !target.contains("apple") {
        println!(
            "cargo:warning={crate_label}: `metal` feature ignored on non-Apple target ({target})"
        );
    }
    if feature_enabled("cuda") && target.contains("apple") {
        println!("cargo:warning={crate_label}: `cuda` feature ignored on Apple target ({target})");
    }
    if feature_enabled("hip") && target.contains("apple") {
        println!("cargo:warning={crate_label}: `hip` feature ignored on Apple target ({target})");
    }
    if feature_enabled("openvino") && target.contains("apple") {
        println!(
            "cargo:warning={crate_label}: `openvino` feature ignored on Apple target ({target})"
        );
    }
}

#[must_use]
pub fn find_ggml_lib_dir(dst: &Path, out_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        dst.join("src"),
        dst.join("lib"),
        dst.to_path_buf(),
        out_dir.join("build").join("src"),
        out_dir.join("build").join("lib"),
        out_dir.join("build").join("Release").join("src"),
        out_dir.join("build").join("Debug").join("src"),
    ];
    candidates.into_iter().find(|p| has_ggml_static_lib(p))
}

#[must_use]
pub fn has_ggml_static_lib(dir: &Path) -> bool {
    dir.join("libggml.a").exists()
        || dir.join("libggml.lib").exists()
        || dir.join("ggml.lib").exists()
}
