//! Browser-side QTS manifest and capability types (shared by `xlai-qts-core::browser` and `xlai-wasm` via `include!`).
//!
//! This file intentionally avoids `xlai-sys`, `ort`, or filesystem GGUF loading.

use serde::{Deserialize, Serialize};

/// Logical names for artifacts referenced by a [`QtsModelManifest`].
pub mod logical_names {
    pub const MAIN_GGUF: &str = "main_gguf";
    pub const VOCODER_ONNX: &str = "vocoder_onnx";
    pub const REFERENCE_CODEC_ONNX: &str = "reference_codec_onnx";
    pub const REFERENCE_CODEC_PREPROCESS_JSON: &str = "reference_codec_preprocess_json";
}

/// One file in a QTS model bundle (see `docs/qts-wasm-model-manifest.md`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QtsModelFileEntry {
    pub logical_name: String,
    pub filename: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Versioned manifest for fetch/cache and validation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QtsModelManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub revision: String,
    pub files: Vec<QtsModelFileEntry>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ManifestError {
    #[error("unsupported manifest schema_version {0} (expected 1)")]
    UnsupportedSchema(u32),
    #[error("manifest must declare a file with logical_name '{0}'")]
    MissingLogicalName(&'static str),
}

impl QtsModelManifest {
    /// Returns `Ok(())` if required logical names for baseline TTS are present.
    ///
    /// ICL / reference codec files are optional unless the caller enables clone mode.
    pub fn validate_required_files(&self) -> Result<(), ManifestError> {
        if self.schema_version != 1 {
            return Err(ManifestError::UnsupportedSchema(self.schema_version));
        }
        Self::require_logical(self, logical_names::MAIN_GGUF)?;
        Self::require_logical(self, logical_names::VOCODER_ONNX)?;
        Ok(())
    }

    fn require_logical(&self, name: &'static str) -> Result<(), ManifestError> {
        let found = self.files.iter().any(|entry| entry.logical_name == name);
        if found {
            Ok(())
        } else {
            Err(ManifestError::MissingLogicalName(name))
        }
    }
}

/// Reported engine / GPU tier for browser UIs (see `docs/qts-wasm-browser-runtime.md`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QtsBrowserGpuTier {
    /// No WebGPU or not yet probed.
    Unknown,
    /// WebGPU adapter obtained (implementation-specific probe may be added later).
    WebGpuAvailable,
    /// Explicit CPU-only path.
    CpuOnly,
}

/// WASM-facing capability snapshot (stable JSON for JS).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QtsBrowserCapabilities {
    /// Engine can run local synthesis in this build.
    pub engine_available: bool,
    /// Stable reason when `engine_available` is false.
    pub engine_status: String,
    pub talker_gpu_tier: QtsBrowserGpuTier,
    pub vocoder_gpu_tier: QtsBrowserGpuTier,
    pub schema_version: u32,
}

impl QtsBrowserCapabilities {
    /// Current stub build: manifests validate; GGML + ORT browser engines are not wired.
    pub const STUB_SCHEMA_VERSION: u32 = 1;
    pub const STUB_STATUS: &'static str = "engine_pending";

    #[must_use]
    pub fn current_stub() -> Self {
        Self {
            engine_available: false,
            engine_status: Self::STUB_STATUS.to_owned(),
            talker_gpu_tier: QtsBrowserGpuTier::Unknown,
            vocoder_gpu_tier: QtsBrowserGpuTier::Unknown,
            schema_version: Self::STUB_SCHEMA_VERSION,
        }
    }
}

#[cfg(test)]
mod browser_include_tests {
    use super::*;

    #[test]
    fn validate_requires_main_and_vocoder() {
        let manifest = QtsModelManifest {
            schema_version: 1,
            model_id: "test".into(),
            revision: "1".into(),
            files: vec![QtsModelFileEntry {
                logical_name: logical_names::MAIN_GGUF.into(),
                filename: "m.gguf".into(),
                sha256: None,
                size_bytes: None,
                url: None,
            }],
        };
        assert_eq!(
            manifest.validate_required_files(),
            Err(ManifestError::MissingLogicalName(
                logical_names::VOCODER_ONNX
            ))
        );
    }
}
