//! Qwen3 TTS (`xlai-qts-core`) exposed as an [`xlai_core::TtsModel`].
//!
//! ## Voice cloning
//!
//! [`VoiceSpec::Clone`](xlai_core::VoiceSpec::Clone) uses the **first**
//! [`VoiceReferenceSample`](xlai_core::VoiceReferenceSample) only.
//! Reference audio must be **inline WAV** (`MediaSource::InlineData` with `audio/wav`).
//!
//! - Default mode: **ICL** when `VoiceReferenceSample::transcript` is non-empty, otherwise **x-vector only**.
//! - Override with metadata `xlai.qts.voice_clone_mode`: `icl` or `xvector`.
//!
//! ICL requires `qwen3-tts-reference-codec.onnx` + `qwen3-tts-reference-codec-preprocess.json` in the model dir
//! (export with `uv run export-model-artifacts`; see `docs/qts-export-and-hf-publish.md` in the repo root).
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
//! - `xlai.qts.voice_clone_mode` (string: `icl` or `xvector`)

mod request_map;

use std::path::PathBuf;

use hound::{SampleFormat, WavSpec, WavWriter};
use xlai_core::{
    BoxFuture, ErrorKind, MediaSource, Metadata, TtsAudioFormat, TtsModel, TtsRequest, TtsResponse,
    XlaiError,
};
use xlai_qts_core::{ModelPaths, Qwen3TtsEngine, Qwen3TtsError, VoiceCloneMode};

pub use request_map::{
    QtsVoiceCloneParams, synthesize_request_from_tts, voice_clone_params_from_tts,
};

/// Configuration for [`QtsTtsModel`].
#[derive(Clone, Debug)]
pub struct QtsTtsConfig {
    /// Directory containing the main GGUF, `qwen3-tts-vocoder.onnx`, and optionally
    /// `qwen3-tts-reference-codec.onnx` + `qwen3-tts-reference-codec-preprocess.json` for ICL clone.
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
                if let Some(clone) = voice_clone_params_from_tts(&inner)? {
                    if matches!(clone.mode, VoiceCloneMode::Icl)
                        && clone
                            .ref_text
                            .as_ref()
                            .map(|t| t.trim().is_empty())
                            .unwrap_or(true)
                    {
                        return Err(XlaiError::new(
                            ErrorKind::Validation,
                            "ICL voice clone requires a non-empty transcript on the reference sample (or use xlai.qts.voice_clone_mode=xvector)",
                        ));
                    }
                    let prompt = engine
                        .create_voice_clone_prompt(
                            &clone.ref_wav,
                            clone.ref_text.as_deref(),
                            clone.mode,
                        )
                        .map_err(map_qts_err)?;
                    return engine
                        .synthesize_with_voice_clone_prompt(&req, &prompt)
                        .map_err(map_qts_err);
                }
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
                    data: wav,
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
