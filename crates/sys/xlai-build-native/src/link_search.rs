use std::env;
use std::path::{Path, PathBuf};

pub fn emit_search_path(path: &Path) {
    if path.exists() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
}

pub fn emit_search_path_variants(path: &Path) {
    emit_search_path(path);

    for config in ["Release", "RelWithDebInfo", "Debug", "MinSizeRel"] {
        emit_search_path(&path.join(config));
    }
}

pub fn emit_llama_vulkan_sdk_paths(enable_vulkan: bool, target_os: &str) {
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

pub fn emit_vulkan_loader_links(target: &str) {
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

#[must_use]
pub fn vulkan_search_dirs(target: &str) -> Vec<PathBuf> {
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
