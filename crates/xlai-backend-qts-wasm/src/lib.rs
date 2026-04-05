//! QTS [`TtsModel`] for browser / WASM builds.
//!
//! Today this backend is a **stub**: native GGML (`xlai-sys`) and ONNX Runtime (`ort`) do not
//! yet support `wasm32-unknown-unknown` in this workspace (see `docs/qts-wasm-browser-runtime.md`).
//! Callers should use [`QtsBrowserCapabilities::current_stub`] and handle
//! `details.code == "qts_wasm_engine_pending"` on errors.

use futures_util::stream;
use serde::{Deserialize, Serialize};
use xlai_core::{
    BoxFuture, BoxStream, ErrorKind, TtsAudioFormat, TtsChunk, TtsModel, TtsRequest, TtsResponse,
    XlaiError,
};
use xlai_qts_browser::{QtsBrowserCapabilities, QtsModelManifest};

/// JSON-friendly config from JS (optional manifest for future asset loading).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QtsBrowserTtsConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest: Option<QtsModelManifest>,
}

/// Browser QTS TTS model (stub implementation).
#[derive(Debug, Clone, Default)]
pub struct QtsBrowserTtsModel {
    pub config: QtsBrowserTtsConfig,
}

impl QtsBrowserTtsModel {
    #[must_use]
    pub fn new(config: QtsBrowserTtsConfig) -> Self {
        Self { config }
    }

    fn engine_pending_error() -> XlaiError {
        XlaiError::new(
            ErrorKind::Unsupported,
            "local QTS engine is not yet available in this WASM build; see docs/qts-wasm-browser-runtime.md",
        )
        .with_details(serde_json::json!({
            "code": "qts_wasm_engine_pending",
            "capabilities": QtsBrowserCapabilities::current_stub(),
        }))
    }

    fn validate_manifest_if_present(&self) -> Result<(), XlaiError> {
        let Some(ref manifest) = self.config.manifest else {
            return Ok(());
        };
        manifest.validate_required_files().map_err(|e| {
            XlaiError::new(ErrorKind::Validation, e.to_string()).with_details(serde_json::json!({
                "code": "qts_wasm_manifest_invalid",
            }))
        })
    }
}

impl TtsModel for QtsBrowserTtsModel {
    fn provider_name(&self) -> &'static str {
        "qts-browser"
    }

    fn synthesize(&self, request: TtsRequest) -> BoxFuture<'_, Result<TtsResponse, XlaiError>> {
        let check = self.validate_manifest_if_present();
        Box::pin(async move {
            check?;
            if let Some(fmt) = &request.response_format
                && *fmt != TtsAudioFormat::Wav
            {
                return Err(XlaiError::new(
                    ErrorKind::Unsupported,
                    format!(
                        "xlai-backend-qts-wasm stub only documents WAV for future local QTS (got {fmt:?})"
                    ),
                ));
            }
            Err(Self::engine_pending_error())
        })
    }

    fn synthesize_stream(
        &self,
        _request: TtsRequest,
    ) -> BoxStream<'_, Result<TtsChunk, XlaiError>> {
        let err = match self.validate_manifest_if_present() {
            Ok(()) => Self::engine_pending_error(),
            Err(e) => e,
        };
        Box::pin(stream::once(async move { Err(err) }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlai_core::{TtsDeliveryMode, VoiceSpec};

    #[tokio::test]
    async fn synthesize_returns_engine_pending() {
        let model = QtsBrowserTtsModel::default();
        let err = model
            .synthesize(TtsRequest {
                model: None,
                input: "hi".into(),
                voice: VoiceSpec::Preset {
                    name: "alloy".into(),
                },
                response_format: Some(TtsAudioFormat::Wav),
                speed: None,
                instructions: None,
                delivery: TtsDeliveryMode::Unary,
                metadata: Default::default(),
            })
            .await
            .expect_err("stub must error");
        assert_eq!(err.kind, ErrorKind::Unsupported);
        let details = err.details.expect("details");
        assert_eq!(details["code"], "qts_wasm_engine_pending");
    }
}
