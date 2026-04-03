//! Qwen3 TTS (`xlai-qts-core`) exposed as an [`xlai_core::TtsModel`].
//!
//! ## Voice cloning
//!
//! [`VoiceSpec::Clone`](xlai_core::VoiceSpec::Clone) is **not** supported in this release.
//! A Rust-native voice-conditioning implementation is planned as phase 2.
//!
//! ## QTS tuning via metadata
//!
//! Optional keys on [`TtsRequest::metadata`](xlai_core::TtsRequest::metadata) (`serde_json::Value`):
//! - `xlai.qts.thread_count` (number)
//! - `xlai.qts.max_audio_frames` (number)
//! - `xlai.qts.temperature` (number)
//! - `xlai.qts.top_p` (number)
//! - `xlai.qts.top_k` (number)
//! - `xlai.qts.repetition_penalty` (number)
//! - `xlai.qts.language_id` (number)
//! - `xlai.qts.vocoder_thread_count` (number)
//! - `xlai.qts.vocoder_chunk_size` (number)
//! - `xlai.qts.talker_kv_mode` (string: `f16` or `turboquant`)

mod request_map;

use std::path::PathBuf;

use base64::{Engine, engine::general_purpose::STANDARD};
use hound::{SampleFormat, WavSpec, WavWriter};
use xlai_core::{
    BoxFuture, ErrorKind, MediaSource, Metadata, TtsAudioFormat, TtsModel, TtsRequest, TtsResponse,
    XlaiError,
};
use xlai_qts_core::{ModelPaths, Qwen3TtsEngine, Qwen3TtsError};

pub use request_map::synthesize_request_from_tts;

/// Configuration for [`QtsTtsModel`].
#[derive(Clone, Debug)]
pub struct QtsTtsConfig {
    /// Directory containing the main GGUF and `qwen3-tts-vocoder.onnx`.
    pub model_dir: PathBuf,
}

impl QtsTtsConfig {
    #[must_use]
    pub fn new(model_dir: PathBuf) -> Self {
        Self { model_dir }
    }

    fn model_paths(&self) -> ModelPaths {
        ModelPaths::from_model_dir(&self.model_dir)
    }
}

/// Native Qwen3 TTS backend implementing [`TtsModel`].
///
/// `Qwen3TtsEngine` is not `Send`; to satisfy [`TtsModel`]'s `Send` futures this type keeps only
/// the model directory and loads the engine inside [`tokio::task::spawn_blocking`] on each call.
/// Cache/reuse can be added later with a dedicated worker thread if needed.
#[derive(Clone, Debug)]
pub struct QtsTtsModel {
    model_dir: PathBuf,
}

impl QtsTtsModel {
    /// Validates that models can be loaded from disk (blocking).
    ///
    /// # Errors
    ///
    /// Returns [`XlaiError`] when the engine fails to load.
    pub fn new(config: QtsTtsConfig) -> Result<Self, XlaiError> {
        let paths = config.model_paths();
        let _probe = Qwen3TtsEngine::load(paths).map_err(map_qts_err)?;
        drop(_probe);
        Ok(Self {
            model_dir: config.model_dir,
        })
    }
}

impl TtsModel for QtsTtsModel {
    fn provider_name(&self) -> &'static str {
        "qts"
    }

    fn synthesize(&self, request: TtsRequest) -> BoxFuture<'_, Result<TtsResponse, XlaiError>> {
        let model_dir = self.model_dir.clone();
        Box::pin(async move {
            if let Some(fmt) = &request.response_format
                && *fmt != TtsAudioFormat::Wav
            {
                return Err(XlaiError::new(
                    ErrorKind::Unsupported,
                    format!(
                        "xlai-backend-qts only supports WAV output in this release (got {fmt:?})"
                    ),
                ));
            }

            let inner = request;
            let sync_result = tokio::task::spawn_blocking(move || {
                let paths = ModelPaths::from_model_dir(&model_dir);
                let engine = Qwen3TtsEngine::load(paths).map_err(map_qts_err)?;
                let req = synthesize_request_from_tts(&inner)?;
                engine.synthesize(&req).map_err(map_qts_err)
            })
            .await
            .map_err(|join_err| {
                XlaiError::new(
                    ErrorKind::Provider,
                    format!("QTS synthesis task failed: {join_err}"),
                )
            })?;

            let result = sync_result?;
            let wav = pcm_f32_to_wav_bytes(&result.pcm_f32, result.sample_rate_hz)
                .map_err(|message| XlaiError::new(ErrorKind::Provider, message))?;

            Ok(TtsResponse {
                audio: MediaSource::InlineData {
                    mime_type: "audio/wav".to_owned(),
                    data_base64: STANDARD.encode(wav),
                },
                mime_type: "audio/wav".to_owned(),
                metadata: Metadata::default(),
            })
        })
    }
}

fn map_qts_err(err: Qwen3TtsError) -> XlaiError {
    XlaiError::new(ErrorKind::Provider, err.to_string())
}

fn pcm_f32_to_wav_bytes(pcm_f32: &[f32], sample_rate_hz: u32) -> Result<Vec<u8>, String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut cursor, spec).map_err(|e| e.to_string())?;
    for sample in pcm_f32.iter().copied() {
        let clamped = sample.clamp(-1.0, 1.0);
        writer
            .write_sample((clamped * f32::from(i16::MAX)) as i16)
            .map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    Ok(cursor.into_inner())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
