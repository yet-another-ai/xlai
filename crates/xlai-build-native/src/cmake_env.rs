use std::env;
use std::path::{Path, PathBuf};

pub fn apply_cmake_env_overrides(config: &mut cmake::Config, enable_openblas: bool) {
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

#[must_use]
pub fn resolve_openblas_include_dir() -> Option<PathBuf> {
    if let Ok(openblas_root) = env::var("OpenBLAS_ROOT")
        && let Some(include_dir) = find_openblas_include_dir(&PathBuf::from(openblas_root))
    {
        return Some(include_dir);
    }

    let Ok(vcpkg_root) = env::var("VCPKG_INSTALLATION_ROOT") else {
        return None;
    };
    let Ok(vcpkg_triplet) = env::var("VCPKG_TARGET_TRIPLET") else {
        return None;
    };

    find_openblas_include_dir(&Path::new(&vcpkg_root).join("installed").join(vcpkg_triplet))
}

#[must_use]
pub fn find_openblas_include_dir(root: &Path) -> Option<PathBuf> {
    let candidates = [
        root.join("include"),
        root.join("include/openblas"),
        root.join("include/OpenBLAS"),
        root.join("include/openblas/include"),
    ];

    candidates
        .into_iter()
        .find(|dir| dir.join("cblas.h").exists())
}

pub fn emit_openblas_search_paths() {
    if let Ok(openblas_root) = env::var("OpenBLAS_ROOT") {
        let openblas_root = PathBuf::from(openblas_root);
        crate::link_search::emit_search_path_variants(&openblas_root.join("lib"));
        crate::link_search::emit_search_path_variants(&openblas_root.join("lib64"));
        crate::link_search::emit_search_path_variants(&openblas_root.join("bin"));
    }

    let Ok(vcpkg_root) = env::var("VCPKG_INSTALLATION_ROOT") else {
        return;
    };
    let Ok(vcpkg_triplet) = env::var("VCPKG_TARGET_TRIPLET") else {
        return;
    };

    let installed_dir = Path::new(&vcpkg_root).join("installed").join(vcpkg_triplet);
    crate::link_search::emit_search_path_variants(&installed_dir.join("lib"));
    crate::link_search::emit_search_path_variants(&installed_dir.join("bin"));
}
