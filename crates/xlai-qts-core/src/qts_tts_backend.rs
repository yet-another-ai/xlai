//! Qwen3 TTS engine exposed as an [`xlai_core::TtsModel`].
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
//! (export with `uv run export-model-artifacts`; see `docs/qts/export-and-hf-publish.md` in the repo root).
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
//! - `xlai.qts.vocoder_chunk_size` (number; `> 0` enables pipelining — use **4** for 12Hz, see `QTS12HZ_RECOMMENDED_VOCODER_CHUNK_FRAMES` / `docs/qts/vocoder-streaming.md`)
//! - `xlai.qts.talker_kv_mode` (string: `f16` or `turboquant`)
//! - `xlai.qts.voice_clone_mode` (string: `icl` or `xvector`)

use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, mpsc as std_mpsc};
use std::thread;

use async_stream::try_stream;
use hound::{SampleFormat, WavSpec, WavWriter};
use tokio::sync::{mpsc as tokio_mpsc, oneshot};
use xlai_core::{
    BoxFuture, BoxStream, ErrorKind, MediaSource, Metadata, TtsAudioFormat, TtsChunk,
    TtsDeliveryMode, TtsModel, TtsRequest, TtsResponse, XlaiError,
};

use crate::qts_request_map::{synthesize_request_from_tts, voice_clone_params_from_tts};
use crate::{ModelPaths, Qwen3TtsEngine, Qwen3TtsError, SAMPLE_RATE_HZ, VoiceCloneMode};

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
}

/// Native Qwen3 TTS backend implementing [`TtsModel`].
///
/// The loaded engine is cached in a dedicated worker thread for the lifetime of the backend
/// instance so repeated synthesis requests do not pay model load costs on every call.
#[derive(Clone)]
pub struct QtsTtsModel {
    model_dir: PathBuf,
    runtime: Arc<RuntimeState>,
}

#[derive(Default)]
struct RuntimeState {
    worker: OnceLock<Result<QtsWorkerHandle, String>>,
}

#[derive(Clone)]
struct QtsWorkerHandle {
    sender: Arc<Mutex<std_mpsc::Sender<WorkerCommand>>>,
}

enum WorkerCommand {
    Synthesize {
        request: TtsRequest,
        response: oneshot::Sender<Result<TtsResponse, XlaiError>>,
    },
    StreamSynthesize {
        request: TtsRequest,
        chunks: tokio_mpsc::UnboundedSender<Result<TtsChunk, XlaiError>>,
        done: oneshot::Sender<Result<(), XlaiError>>,
    },
}

impl fmt::Debug for QtsTtsModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QtsTtsModel")
            .field("model_dir", &self.model_dir)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for RuntimeState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeState")
            .field("worker_initialized", &self.worker.get().is_some())
            .finish()
    }
}

impl QtsTtsModel {
    /// Validates that models can be loaded from disk (blocking).
    ///
    /// # Errors
    ///
    /// Returns [`XlaiError`] when the engine fails to load.
    pub fn new(config: QtsTtsConfig) -> Result<Self, XlaiError> {
        let model = Self {
            model_dir: config.model_dir,
            runtime: Arc::new(RuntimeState::default()),
        };
        let _ = model.load_worker()?;
        Ok(model)
    }

    fn load_worker(&self) -> Result<QtsWorkerHandle, XlaiError> {
        let loaded = self
            .runtime
            .worker
            .get_or_init(|| spawn_worker(self.model_dir.clone()).map_err(|error| error.message));

        loaded
            .clone()
            .map_err(|message| XlaiError::new(ErrorKind::Provider, message))
    }
}

impl TtsModel for QtsTtsModel {
    fn provider_name(&self) -> &'static str {
        "qts"
    }

    fn synthesize(&self, request: TtsRequest) -> BoxFuture<'_, Result<TtsResponse, XlaiError>> {
        let worker = self.load_worker();
        Box::pin(async move {
            if let Some(fmt) = &request.response_format
                && *fmt != TtsAudioFormat::Wav
            {
                return Err(XlaiError::new(
                    ErrorKind::Unsupported,
                    format!("xlai-qts-core only supports WAV output in this release (got {fmt:?})"),
                ));
            }

            let worker = worker?;
            let (response_tx, response_rx) = oneshot::channel();
            worker.send(WorkerCommand::Synthesize {
                request,
                response: response_tx,
            })?;
            response_rx.await.map_err(|_| {
                XlaiError::new(
                    ErrorKind::Provider,
                    "QTS worker thread terminated before sending a response",
                )
            })?
        })
    }

    fn synthesize_stream(&self, request: TtsRequest) -> BoxStream<'_, Result<TtsChunk, XlaiError>> {
        match request.delivery {
            TtsDeliveryMode::Unary => Box::pin(try_stream! {
                let response = self.synthesize(request).await?;
                yield TtsChunk::Started {
                    mime_type: response.mime_type.clone(),
                    metadata: Metadata::default(),
                };
                let data = match &response.audio {
                    MediaSource::InlineData { data, .. } => data.clone(),
                    MediaSource::Url { .. } => Err(XlaiError::new(
                        ErrorKind::Unsupported,
                        "qts stream fallback requires inline audio bytes",
                    ))?,
                };
                yield TtsChunk::AudioDelta { data };
                yield TtsChunk::Finished { response };
            }),
            TtsDeliveryMode::Stream => {
                let worker = self.load_worker();
                Box::pin(try_stream! {
                    let worker = worker?;
                    let (chunk_tx, mut chunk_rx) = tokio_mpsc::unbounded_channel();
                    let (done_tx, done_rx) = oneshot::channel();
                    worker.send(WorkerCommand::StreamSynthesize {
                        request,
                        chunks: chunk_tx,
                        done: done_tx,
                    })?;

                    while let Some(chunk) = chunk_rx.recv().await {
                        yield chunk?;
                    }

                    match done_rx.await {
                        Ok(Ok(())) => {}
                        Ok(Err(error)) => Err(error)?,
                        Err(_) => Err(XlaiError::new(
                            ErrorKind::Provider,
                            "QTS worker thread terminated before finishing the stream",
                        ))?,
                    }
                })
            }
        }
    }
}

impl QtsWorkerHandle {
    fn send(&self, command: WorkerCommand) -> Result<(), XlaiError> {
        let sender = self.sender.lock().map_err(|_| {
            XlaiError::new(
                ErrorKind::Provider,
                "QTS worker sender lock was poisoned by a previous panic",
            )
        })?;
        sender.send(command).map_err(|_| {
            XlaiError::new(
                ErrorKind::Provider,
                "QTS worker thread is no longer accepting requests",
            )
        })
    }
}

fn spawn_worker(model_dir: PathBuf) -> Result<QtsWorkerHandle, XlaiError> {
    let (command_tx, command_rx) = std_mpsc::channel();
    let (ready_tx, ready_rx) = std_mpsc::channel();
    thread::Builder::new()
        .name("xlai-qts-worker".to_owned())
        .spawn(move || {
            let engine =
                Qwen3TtsEngine::load(ModelPaths::from_model_dir(&model_dir)).map_err(map_qts_err);

            match engine {
                Ok(engine) => {
                    let _ = ready_tx.send(Ok(()));
                    run_worker_loop(engine, command_rx);
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error.message));
                }
            }
        })
        .map_err(|error| {
            XlaiError::new(
                ErrorKind::Provider,
                format!("failed to spawn QTS worker thread: {error}"),
            )
        })?;

    match ready_rx.recv() {
        Ok(Ok(())) => Ok(QtsWorkerHandle {
            sender: Arc::new(Mutex::new(command_tx)),
        }),
        Ok(Err(message)) => Err(XlaiError::new(ErrorKind::Provider, message)),
        Err(_) => Err(XlaiError::new(
            ErrorKind::Provider,
            "QTS worker thread exited before completing initialization",
        )),
    }
}

fn run_worker_loop(engine: Qwen3TtsEngine, command_rx: std_mpsc::Receiver<WorkerCommand>) {
    for command in command_rx {
        match command {
            WorkerCommand::Synthesize { request, response } => {
                let _ = response.send(run_synthesis(&engine, request));
            }
            WorkerCommand::StreamSynthesize {
                request,
                chunks,
                done,
            } => {
                let result = run_stream_synthesis(&engine, request, chunks);
                let _ = done.send(result);
            }
        }
    }
}

fn run_synthesis(engine: &Qwen3TtsEngine, request: TtsRequest) -> Result<TtsResponse, XlaiError> {
    let req = synthesize_request_from_tts(&request)?;
    let result = if let Some(clone) = voice_clone_params_from_tts(&request)? {
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
            .create_voice_clone_prompt(&clone.ref_wav, clone.ref_text.as_deref(), clone.mode)
            .map_err(map_qts_err)?;
        engine
            .synthesize_with_voice_clone_prompt(&req, &prompt)
            .map_err(map_qts_err)?
    } else {
        engine.synthesize(&req).map_err(map_qts_err)?
    };

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
}

fn run_stream_synthesis(
    engine: &Qwen3TtsEngine,
    request: TtsRequest,
    chunks: tokio_mpsc::UnboundedSender<Result<TtsChunk, XlaiError>>,
) -> Result<(), XlaiError> {
    send_stream_chunk(
        &chunks,
        Ok(TtsChunk::Started {
            mime_type: "audio/wav".to_owned(),
            metadata: Metadata::default(),
        }),
    )?;

    let req = synthesize_request_from_tts(&request)?;
    let mut full_pcm = Vec::new();
    let mut sink = |pcm_f32: &[f32]| -> Result<(), Qwen3TtsError> {
        full_pcm.extend_from_slice(pcm_f32);
        let wav_chunk =
            pcm_f32_to_wav_bytes(pcm_f32, SAMPLE_RATE_HZ).map_err(Qwen3TtsError::InvalidInput)?;
        send_stream_chunk(&chunks, Ok(TtsChunk::AudioDelta { data: wav_chunk }))
            .map_err(|error| Qwen3TtsError::InvalidInput(error.message))?;
        Ok(())
    };

    let result = if let Some(clone) = voice_clone_params_from_tts(&request)? {
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
            .create_voice_clone_prompt(&clone.ref_wav, clone.ref_text.as_deref(), clone.mode)
            .map_err(map_qts_err)?;
        engine
            .synthesize_with_voice_clone_prompt_streaming(&req, &prompt, &mut sink)
            .map_err(map_qts_err)?
    } else {
        engine
            .synthesize_streaming(&req, &mut sink)
            .map_err(map_qts_err)?
    };

    let wav = pcm_f32_to_wav_bytes(&full_pcm, result.sample_rate_hz)
        .map_err(|message| XlaiError::new(ErrorKind::Provider, message))?;
    send_stream_chunk(
        &chunks,
        Ok(TtsChunk::Finished {
            response: TtsResponse {
                audio: MediaSource::InlineData {
                    mime_type: "audio/wav".to_owned(),
                    data: wav,
                },
                mime_type: "audio/wav".to_owned(),
                metadata: Metadata::default(),
            },
        }),
    )
}

fn send_stream_chunk(
    chunks: &tokio_mpsc::UnboundedSender<Result<TtsChunk, XlaiError>>,
    chunk: Result<TtsChunk, XlaiError>,
) -> Result<(), XlaiError> {
    chunks.send(chunk).map_err(|_| {
        XlaiError::new(
            ErrorKind::Provider,
            "QTS streaming receiver dropped before synthesis completed",
        )
    })
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
    // Standard PCM WAV header is 44 bytes; 16-bit mono samples = 2 bytes each.
    let buf = Vec::with_capacity(44usize.saturating_add(pcm_f32.len().saturating_mul(2)));
    let mut cursor = std::io::Cursor::new(buf);
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
#[path = "qts_tts_backend_tests.rs"]
mod tests;
