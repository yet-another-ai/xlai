//! Qwen3 TTS (GGUF + GGML) — native inference library.
//!
//! The [`browser`] module re-exports serde manifest types from `xlai-qts-manifest` (shared with `xlai-wasm`).
#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::collapsible_if)]
//!
//! Output sample rate for the published Qwen3-TTS checkpoints.
pub const SAMPLE_RATE_HZ: u32 = 24_000;

mod error;
pub mod browser {
    pub use xlai_qts_manifest::*;
}
mod ggml;
mod model;
pub mod pipeline;
mod synthesis_profile;
mod voice_clone_builder;
mod voice_clone_prompt;

pub use error::Qwen3TtsError;
pub use model::{GgufFile, ModelPaths, load_and_validate};
pub use pipeline::backend::BackendKind;
pub use pipeline::reference_codec_encoder::{
    ReferenceCodecEncoder, ReferenceCodecPreprocessConfig,
};
pub use pipeline::speaker_encoder::{SpeakerEncoder, SpeakerEncoderConfig};
pub use pipeline::tokenizer::{TextTokenizer, TokenizerConfig};
pub use pipeline::tts_transformer::{
    CodecRollout, CodecRolloutSubTimings, IclPrefillConditioning, PrefillConditioning,
    PrefillForwardOutputs, PreparedPrefillInputs, SelectedCodecFrame, TtsTransformer,
    TtsTransformerConfig, VocoderChunk,
};
pub use pipeline::vocoder::{
    Vocoder, VocoderConfig, VocoderExecutionProvider, VocoderGraphTemplate,
};
pub use pipeline::vocoder_streaming::QTS12HZ_RECOMMENDED_VOCODER_CHUNK_FRAMES;
pub use synthesis_profile::SynthesisStageTimings;
pub use voice_clone_builder::{
    VoiceCloneMode, build_icl_voice_clone_prompt, build_xvector_voice_clone_prompt,
};
pub use voice_clone_prompt::{
    TensorF32, TensorI32, VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION, VoiceClonePromptV2,
};

use std::time::Instant;

/// User-facing synthesis parameters (stable for future `gdext` bindings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TalkerKvMode {
    #[default]
    F16,
    TurboQuant,
}

impl TalkerKvMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::TurboQuant => "turboquant",
        }
    }

    pub fn parse(value: &str) -> Result<Self, Qwen3TtsError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "turboquant" | "turbo" | "q8_0" | "q8" => Ok(Self::TurboQuant),
            other => Err(Qwen3TtsError::InvalidInput(format!(
                "unknown talker KV mode '{other}' (expected f16 or turboquant)"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynthesizeRequest {
    pub text: String,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub max_audio_frames: usize,
    pub thread_count: usize,
    pub repetition_penalty: f32,
    /// Codec language id (e.g. 2050=en, 2055=zh, 2058=ja).
    pub language_id: i32,
    /// Number of CPU threads for vocoder decode when pipelining is enabled.
    /// Defaults to 4. Set to 0 to derive a backend-agnostic fallback from `thread_count`.
    pub vocoder_thread_count: usize,
    /// When > 0, pipeline transformer (GPU) and vocoder (CPU) by processing
    /// vocoder chunks of this many frames in a background thread while the
    /// transformer continues generating. Set to 0 to disable (sequential).
    pub vocoder_chunk_size: usize,
    /// Experimental talker KV cache storage mode.
    pub talker_kv_mode: TalkerKvMode,
}

impl Default for SynthesizeRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            max_audio_frames: 4096,
            thread_count: 4,
            repetition_penalty: 1.05,
            language_id: 2050,
            vocoder_thread_count: 4,
            vocoder_chunk_size: 0,
            talker_kv_mode: TalkerKvMode::F16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynthesizeResult {
    pub pcm_f32: Vec<f32>,
    pub sample_rate_hz: u32,
    /// Best-effort: frame count not yet exposed by C API; 0 for now.
    pub generated_frames: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingSynthesizeResult {
    pub sample_rate_hz: u32,
    pub generated_frames: usize,
    pub generated_samples: usize,
}

struct PreparedSynthesis<'a> {
    prepared_inputs: PreparedPrefillInputs,
    prompt_frames: Vec<Vec<i32>>,
    prefix_frame_count: usize,
    speaker_encode: std::time::Duration,
    tokenize: std::time::Duration,
    prefill_build: std::time::Duration,
    _speaker_embedding: SpeakerEmbeddingStorage<'a>,
}

enum SpeakerEmbeddingStorage<'a> {
    Borrowed(&'a [f32]),
    Owned(Vec<f32>),
}

impl<'a> SpeakerEmbeddingStorage<'a> {
    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Borrowed(values) => values,
            Self::Owned(values) => values,
        }
    }
}

pub struct Qwen3TtsEngine {
    paths: ModelPaths,
    tokenizer: TextTokenizer,
    transformer: TtsTransformer,
    vocoder: Vocoder,
    speaker_encoder: SpeakerEncoder,
    reference_codec: Option<ReferenceCodecEncoder>,
}

impl Qwen3TtsEngine {
    pub fn load(paths: ModelPaths) -> Result<Self, Qwen3TtsError> {
        load_and_validate(&paths)?;
        let main = GgufFile::open(&paths.main_gguf)?;
        let tokenizer = TextTokenizer::load_from_gguf(&main)?;
        let transformer = TtsTransformer::load_from_gguf(&main)?;
        let vocoder = Vocoder::load_from_onnx(&paths.vocoder_onnx)?;
        let speaker_encoder = SpeakerEncoder::new(transformer.config().hidden_size as usize)?;
        let reference_codec = if paths.reference_codec_exists() {
            Some(ReferenceCodecEncoder::load_from_onnx(
                &paths.reference_codec_onnx,
                None,
            )?)
        } else {
            None
        };

        Ok(Self {
            paths,
            tokenizer,
            transformer,
            vocoder,
            speaker_encoder,
            reference_codec,
        })
    }

    pub fn from_model_dir(dir: impl AsRef<std::path::Path>) -> Result<Self, Qwen3TtsError> {
        Self::load(ModelPaths::from_model_dir(dir))
    }

    pub fn model_paths(&self) -> &ModelPaths {
        &self.paths
    }

    #[must_use]
    pub fn tokenizer(&self) -> &TextTokenizer {
        &self.tokenizer
    }

    #[must_use]
    pub fn transformer(&self) -> &TtsTransformer {
        &self.transformer
    }

    #[must_use]
    pub fn vocoder(&self) -> &Vocoder {
        &self.vocoder
    }

    #[must_use]
    pub fn vocoder_execution_provider(&self) -> VocoderExecutionProvider {
        self.vocoder.execution_provider()
    }

    #[must_use]
    pub fn vocoder_backend_label(&self) -> &'static str {
        self.vocoder.execution_provider_label()
    }

    #[must_use]
    pub fn speaker_encoder(&self) -> &SpeakerEncoder {
        &self.speaker_encoder
    }

    #[must_use]
    pub fn reference_codec(&self) -> Option<&ReferenceCodecEncoder> {
        self.reference_codec.as_ref()
    }

    /// Default `model_id` string embedded in native prompts (for debugging / interchange).
    pub const DEFAULT_VOICE_CLONE_MODEL_ID: &'static str = "Qwen3-TTS-12Hz-0.6B-Base";

    /// Build a [`VoiceClonePromptV2`] from reference WAV bytes (no Python, no pre-serialized CBOR).
    ///
    /// - [`VoiceCloneMode::XVectorOnly`]: uses the handcrafted [`SpeakerEncoder`] only.
    /// - [`VoiceCloneMode::Icl`]: requires `ref_text` and a loaded [`ReferenceCodecEncoder`]
    ///   (`qwen3-tts-reference-codec.onnx` + preprocess JSON in the model directory).
    pub fn create_voice_clone_prompt(
        &self,
        ref_wav_bytes: &[u8],
        ref_text: Option<&str>,
        mode: VoiceCloneMode,
    ) -> Result<VoiceClonePromptV2, Qwen3TtsError> {
        let dim = self.speaker_embedding_size();
        let n_cb = self.transformer.config().n_codebooks as usize;
        let model_id = Self::DEFAULT_VOICE_CLONE_MODEL_ID;
        let prompt = match mode {
            VoiceCloneMode::XVectorOnly => build_xvector_voice_clone_prompt(
                self.speaker_encoder(),
                dim,
                ref_wav_bytes,
                model_id,
            )?,
            VoiceCloneMode::Icl => {
                let codec = self.reference_codec.as_ref().ok_or_else(|| {
                    Qwen3TtsError::InvalidInput(
                        "ICL voice clone requires qwen3-tts-reference-codec.onnx and \
                         qwen3-tts-reference-codec-preprocess.json in the model directory (export from Python)"
                            .into(),
                    )
                })?;
                let text = ref_text.ok_or_else(|| {
                    Qwen3TtsError::InvalidInput(
                        "ICL voice clone requires ref_text (transcript of the reference clip)"
                            .into(),
                    )
                })?;
                build_icl_voice_clone_prompt(
                    self.speaker_encoder(),
                    codec,
                    dim,
                    n_cb,
                    ref_wav_bytes,
                    text,
                    model_id,
                )?
            }
        };
        self.validate_speaker_embedding(prompt.speaker_embedding())?;
        prompt.validate()?;
        Ok(prompt)
    }

    #[must_use]
    pub fn primary_backend_kind(&self) -> BackendKind {
        self.transformer.primary_backend_kind()
    }

    #[must_use]
    pub fn encode_for_tts(&self, text: &str) -> Vec<i32> {
        self.tokenizer.encode_for_tts(text)
    }

    pub fn encode_reference_speaker(&self, wav_bytes: &[u8]) -> Result<Vec<f32>, Qwen3TtsError> {
        self.speaker_encoder.encode_wav_bytes(wav_bytes)
    }

    #[must_use]
    pub fn speaker_embedding_size(&self) -> usize {
        self.transformer.config().hidden_size as usize
    }

    pub fn decode_voice_clone_prompt(
        &self,
        bytes: &[u8],
    ) -> Result<VoiceClonePromptV2, Qwen3TtsError> {
        let prompt = VoiceClonePromptV2::from_cbor_slice(bytes)?;
        self.validate_speaker_embedding(prompt.speaker_embedding())?;
        if let Some((_, codebooks)) = prompt.ref_code_shape()
            && codebooks != self.transformer.config().n_codebooks as usize
        {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "voice clone prompt ref_code must have {} codebooks per frame",
                self.transformer.config().n_codebooks
            )));
        }
        Ok(prompt)
    }

    pub fn synthesize_with_voice_clone_prompt(
        &self,
        req: &SynthesizeRequest,
        prompt: &VoiceClonePromptV2,
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.validate_speaker_embedding(prompt.speaker_embedding())?;
        self.synthesize_impl(req, None, Some(prompt), None)
    }

    pub fn synthesize(&self, req: &SynthesizeRequest) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.synthesize_impl(req, None, None, None)
    }

    pub fn synthesize_streaming<S>(
        &self,
        req: &SynthesizeRequest,
        sink: &mut S,
    ) -> Result<StreamingSynthesizeResult, Qwen3TtsError>
    where
        S: StreamingSynthesis + Send,
    {
        self.synthesize_streaming_impl(req, None, None, sink, None)
    }

    pub fn synthesize_with_voice_clone_prompt_streaming<S>(
        &self,
        req: &SynthesizeRequest,
        prompt: &VoiceClonePromptV2,
        sink: &mut S,
    ) -> Result<StreamingSynthesizeResult, Qwen3TtsError>
    where
        S: StreamingSynthesis + Send,
    {
        self.validate_speaker_embedding(prompt.speaker_embedding())?;
        self.synthesize_streaming_impl(req, None, Some(prompt), sink, None)
    }

    /// Same as [`Self::synthesize`], plus wall-clock timings per pipeline stage.
    pub fn synthesize_with_profile(
        &self,
        req: &SynthesizeRequest,
    ) -> Result<(SynthesizeResult, SynthesisStageTimings), Qwen3TtsError> {
        let mut timings = SynthesisStageTimings::default();
        let result = self.synthesize_impl(req, None, None, Some(&mut timings))?;
        Ok((result, timings))
    }

    /// Same as [`Self::synthesize_with_voice_clone_prompt`], plus stage timings.
    pub fn synthesize_with_voice_clone_prompt_profile(
        &self,
        req: &SynthesizeRequest,
        prompt: &VoiceClonePromptV2,
    ) -> Result<(SynthesizeResult, SynthesisStageTimings), Qwen3TtsError> {
        self.validate_speaker_embedding(prompt.speaker_embedding())?;
        let mut timings = SynthesisStageTimings::default();
        let result = self.synthesize_impl(req, None, Some(prompt), Some(&mut timings))?;
        Ok((result, timings))
    }

    fn synthesize_impl(
        &self,
        req: &SynthesizeRequest,
        speaker_embedding_override: Option<&[f32]>,
        voice_clone_prompt: Option<&VoiceClonePromptV2>,
        timings: Option<&mut SynthesisStageTimings>,
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        let prepared =
            self.prepare_synthesis(req, speaker_embedding_override, voice_clone_prompt)?;

        if req.vocoder_chunk_size > 0 {
            self.synthesize_pipelined(req, &prepared, timings)
        } else {
            self.synthesize_sequential(req, &prepared, timings)
        }
    }

    fn synthesize_streaming_impl<S>(
        &self,
        req: &SynthesizeRequest,
        speaker_embedding_override: Option<&[f32]>,
        voice_clone_prompt: Option<&VoiceClonePromptV2>,
        sink: &mut S,
        timings: Option<&mut SynthesisStageTimings>,
    ) -> Result<StreamingSynthesizeResult, Qwen3TtsError>
    where
        S: StreamingSynthesis + Send,
    {
        let prepared =
            self.prepare_synthesis(req, speaker_embedding_override, voice_clone_prompt)?;

        if req.vocoder_chunk_size > 0 {
            self.synthesize_pipelined_streaming(req, &prepared, sink, timings)
        } else {
            self.synthesize_sequential_streaming(req, &prepared, sink, timings)
        }
    }

    fn prepare_synthesis<'a>(
        &self,
        req: &SynthesizeRequest,
        speaker_embedding_override: Option<&'a [f32]>,
        voice_clone_prompt: Option<&'a VoiceClonePromptV2>,
    ) -> Result<PreparedSynthesis<'a>, Qwen3TtsError> {
        let speaker_encode = std::time::Duration::ZERO;
        let speaker_embedding = if let Some(speaker_embedding) = speaker_embedding_override {
            SpeakerEmbeddingStorage::Borrowed(speaker_embedding)
        } else if let Some(speaker_embedding) =
            voice_clone_prompt.map(VoiceClonePromptV2::speaker_embedding)
        {
            SpeakerEmbeddingStorage::Borrowed(speaker_embedding)
        } else {
            SpeakerEmbeddingStorage::Owned(vec![0.0f32; self.speaker_embedding_size()])
        };

        let t_tok = Instant::now();
        let prompt_frames = if let Some(prompt) = voice_clone_prompt {
            prompt
                .ref_code_shape()
                .map_or_else(Vec::new, |(frames, codebooks)| {
                    let values = prompt.ref_code_values().unwrap_or(&[]);
                    (0..frames)
                        .map(|frame_idx| {
                            let start = frame_idx * codebooks;
                            let end = start + codebooks;
                            values[start..end].to_vec()
                        })
                        .collect::<Vec<_>>()
                })
        } else {
            Vec::new()
        };
        let prefix_frame_count = prompt_frames.len();
        let tokenize = t_tok.elapsed();

        let t_prefill = Instant::now();
        let prepared_inputs = if let Some(prompt) = voice_clone_prompt {
            if prompt.icl_mode {
                let text_tokens = self.tokenizer.encode_for_tts(&req.text);
                let ref_text_tokens = self.tokenizer.encode_ref_for_tts(&prompt.ref_text);
                self.transformer.build_icl_prefill_inputs(
                    IclPrefillConditioning {
                        text_tokens: &text_tokens,
                        ref_text_tokens: &ref_text_tokens,
                        speaker_embd: Some(speaker_embedding.as_slice()),
                        ref_code_frames: &prompt_frames,
                        language_id: req.language_id,
                    },
                    req.thread_count,
                )?
            } else {
                let ref_codebook_0 = prompt_frames
                    .iter()
                    .filter_map(|frame| frame.first().copied())
                    .collect::<Vec<_>>();
                let text_tokens = self.tokenizer.encode_for_tts(&req.text);
                self.transformer.build_prefill_inputs(
                    PrefillConditioning {
                        text_tokens: &text_tokens,
                        speaker_embd: Some(speaker_embedding.as_slice()),
                        ref_codebook_0: &ref_codebook_0,
                        language_id: req.language_id,
                    },
                    req.thread_count,
                )?
            }
        } else {
            let text_tokens = self.tokenizer.encode_for_tts(&req.text);
            self.transformer.build_prefill_inputs(
                PrefillConditioning {
                    text_tokens: &text_tokens,
                    speaker_embd: Some(speaker_embedding.as_slice()),
                    ref_codebook_0: &[],
                    language_id: req.language_id,
                },
                req.thread_count,
            )?
        };
        let prefill_build = t_prefill.elapsed();

        Ok(PreparedSynthesis {
            prepared_inputs,
            prompt_frames,
            prefix_frame_count,
            speaker_encode,
            tokenize,
            prefill_build,
            _speaker_embedding: speaker_embedding,
        })
    }

    fn synthesize_sequential(
        &self,
        req: &SynthesizeRequest,
        prepared: &PreparedSynthesis<'_>,
        timings: Option<&mut SynthesisStageTimings>,
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        let t_roll = Instant::now();
        let codec_rollout = self.transformer.rollout_codec_frames_kv(
            &prepared.prepared_inputs.prefill_embd,
            &prepared.prepared_inputs.trailing_text_hidden,
            &prepared.prepared_inputs.tts_pad_embed,
            &prepared.prompt_frames,
            req.talker_kv_mode,
            req.thread_count,
            req.max_audio_frames,
            req.repetition_penalty,
            req.temperature,
            req.top_k,
            req.top_p,
        )?;
        let codec_rollout_dur = t_roll.elapsed();

        let generated_frames = codec_rollout
            .frames
            .len()
            .saturating_sub(prepared.prefix_frame_count);

        let t_post = Instant::now();
        let flattened_codes = codec_rollout
            .frames
            .iter()
            .flat_map(|frame| frame.codebook_tokens.iter().copied())
            .collect::<Vec<_>>();
        let flatten_dur = t_post.elapsed();

        let t_voc = Instant::now();
        let pcm_all = self.vocoder.decode(
            &flattened_codes,
            codec_rollout.frames.len(),
            req.thread_count,
        )?;
        let vocoder_decode = t_voc.elapsed();

        let t_trim = Instant::now();
        let pcm_f32 = if prepared.prefix_frame_count == 0 || codec_rollout.frames.is_empty() {
            pcm_all
        } else {
            let cut = prepared
                .prefix_frame_count
                .saturating_mul(pcm_all.len())
                .checked_div(codec_rollout.frames.len())
                .unwrap_or(0)
                .min(pcm_all.len());
            pcm_all[cut..].to_vec()
        };
        let post = t_trim.elapsed() + flatten_dur;

        let sample_rate_hz = self.vocoder.config().sample_rate as u32;
        if let Some(t) = timings {
            t.speaker_encode = prepared.speaker_encode;
            t.tokenize = prepared.tokenize;
            t.prefill_build = prepared.prefill_build;
            t.codec_rollout = codec_rollout_dur;
            t.vocoder_decode = vocoder_decode;
            t.post = post;
            t.codec_rollout_detail = codec_rollout.sub_timings;
            t.first_frame_latency = prepared.speaker_encode
                + prepared.tokenize
                + prepared.prefill_build
                + codec_rollout.first_frame_elapsed;
            t.generated_samples = pcm_f32.len();
            t.sample_rate_hz = sample_rate_hz;
        }

        Ok(SynthesizeResult {
            pcm_f32,
            sample_rate_hz,
            generated_frames,
        })
    }

    fn synthesize_sequential_streaming<S>(
        &self,
        req: &SynthesizeRequest,
        prepared: &PreparedSynthesis<'_>,
        sink: &mut S,
        timings: Option<&mut SynthesisStageTimings>,
    ) -> Result<StreamingSynthesizeResult, Qwen3TtsError>
    where
        S: StreamingSynthesis + Send,
    {
        let result = self.synthesize_sequential(req, prepared, timings)?;
        sink.push_pcm_chunk(&result.pcm_f32)?;
        Ok(StreamingSynthesizeResult {
            sample_rate_hz: result.sample_rate_hz,
            generated_frames: result.generated_frames,
            generated_samples: result.pcm_f32.len(),
        })
    }

    /// Pipeline transformer (GPU/main thread) with vocoder (CPU/background thread).
    ///
    /// The transformer generates frames autoregressively; every `chunk_size` generated
    /// frames the codebook tokens are sent to a vocoder worker that decodes them in
    /// parallel on CPU. When the transformer finishes the last chunk is flushed and
    /// all audio is concatenated.
    ///
    /// Chunk boundaries use [`crate::pipeline::vocoder_streaming::OverlapAddChunkDecoder`]:
    /// optional prefix frames are sent as `prefix_warmup_only` chunks (overlap state only, no output PCM).
    /// Overlap width defaults to **1** codec frame; set `QWEN3_TTS_VOCODER_OVERLAP_FRAMES` for a stronger fallback (e.g. `3`).
    fn synthesize_pipelined(
        &self,
        req: &SynthesizeRequest,
        prepared: &PreparedSynthesis<'_>,
        timings: Option<&mut SynthesisStageTimings>,
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        use std::sync::mpsc;

        let chunk_size = req.vocoder_chunk_size;
        let thread_count = req.thread_count;
        let vocoder_thread_count = if req.vocoder_thread_count > 0 {
            req.vocoder_thread_count
        } else {
            (thread_count / 2).max(1)
        };

        let (chunk_tx, chunk_rx) = mpsc::sync_channel::<VocoderChunk>(2);

        let t_pipeline_start = Instant::now();

        std::thread::scope(|s| {
            let vocoder = &self.vocoder;

            let vocoder_handle = s.spawn(
                move || -> Result<(Vec<f32>, std::time::Duration), Qwen3TtsError> {
                    let t_voc_start = Instant::now();
                    let mut all_pcm = Vec::<f32>::new();
                    let mut decoder =
                        crate::pipeline::vocoder_streaming::OverlapAddChunkDecoder::new(
                            vocoder,
                            chunk_size,
                            vocoder_thread_count,
                        );

                    while let Ok(chunk) = chunk_rx.recv() {
                        decoder.process_generated_chunk(vocoder, &chunk, &mut all_pcm)?;
                    }

                    Ok((all_pcm, t_voc_start.elapsed()))
                },
            );

            let t_roll = Instant::now();
            let codec_rollout = self.transformer.rollout_codec_frames_kv_streaming(
                &prepared.prepared_inputs.prefill_embd,
                &prepared.prepared_inputs.trailing_text_hidden,
                &prepared.prepared_inputs.tts_pad_embed,
                &prepared.prompt_frames,
                req.talker_kv_mode,
                thread_count,
                req.max_audio_frames,
                req.repetition_penalty,
                req.temperature,
                req.top_k,
                req.top_p,
                chunk_size,
                &chunk_tx,
            );
            let codec_rollout_dur = t_roll.elapsed();
            drop(chunk_tx);

            let codec_rollout = codec_rollout?;
            let generated_frames = codec_rollout
                .frames
                .len()
                .saturating_sub(prepared.prefix_frame_count);

            let (pcm_all, vocoder_decode) = vocoder_handle.join().unwrap()?;

            let pipeline_wall_clock = t_pipeline_start.elapsed();

            let t_trim = Instant::now();
            // Prefix audio is not accumulated when using prefix warmup chunks; output is generated-only.
            let pcm_f32 = pcm_all;
            let post = t_trim.elapsed();

            let sample_rate_hz = self.vocoder.config().sample_rate as u32;
            if let Some(t) = timings {
                t.speaker_encode = prepared.speaker_encode;
                t.tokenize = prepared.tokenize;
                t.prefill_build = prepared.prefill_build;
                t.codec_rollout = codec_rollout_dur;
                t.vocoder_decode = vocoder_decode;
                t.post = post;
                t.codec_rollout_detail = codec_rollout.sub_timings;
                let sequential_sum = codec_rollout_dur + vocoder_decode;
                t.pipeline_overlap = sequential_sum.saturating_sub(pipeline_wall_clock);
                t.first_frame_latency = prepared.speaker_encode
                    + prepared.tokenize
                    + prepared.prefill_build
                    + codec_rollout.first_frame_elapsed;
                t.generated_samples = pcm_f32.len();
                t.sample_rate_hz = sample_rate_hz;
            }

            Ok(SynthesizeResult {
                pcm_f32,
                sample_rate_hz,
                generated_frames,
            })
        })
    }

    fn synthesize_pipelined_streaming<S>(
        &self,
        req: &SynthesizeRequest,
        prepared: &PreparedSynthesis<'_>,
        sink: &mut S,
        timings: Option<&mut SynthesisStageTimings>,
    ) -> Result<StreamingSynthesizeResult, Qwen3TtsError>
    where
        S: StreamingSynthesis + Send,
    {
        use std::sync::mpsc;

        let chunk_size = req.vocoder_chunk_size;
        let thread_count = req.thread_count;
        let vocoder_thread_count = if req.vocoder_thread_count > 0 {
            req.vocoder_thread_count
        } else {
            (thread_count / 2).max(1)
        };

        let (chunk_tx, chunk_rx) = mpsc::sync_channel::<VocoderChunk>(2);

        let t_pipeline_start = Instant::now();

        std::thread::scope(|s| {
            let vocoder = &self.vocoder;

            let vocoder_handle = s.spawn(
                move || -> Result<(usize, std::time::Duration), Qwen3TtsError> {
                    let t_voc_start = Instant::now();
                    let mut total_samples = 0usize;
                    let mut all_pcm = Vec::<f32>::new();
                    let mut emitted_samples = 0usize;
                    let mut decoder =
                        crate::pipeline::vocoder_streaming::OverlapAddChunkDecoder::new(
                            vocoder,
                            chunk_size,
                            vocoder_thread_count,
                        );
                    let overlap_frames = decoder.overlap_frames();

                    while let Ok(chunk) = chunk_rx.recv() {
                        let len_before = all_pcm.len();
                        decoder.process_generated_chunk(vocoder, &chunk, &mut all_pcm)?;
                        if chunk.prefix_warmup_only {
                            continue;
                        }

                        let current_chunk_samples = all_pcm.len().saturating_sub(len_before);

                        let hold_back_frames = overlap_frames.min(chunk.n_frames);
                        let hold_back_samples = if chunk.n_frames == 0 {
                            0
                        } else {
                            current_chunk_samples
                                .saturating_mul(hold_back_frames)
                                .checked_div(chunk.n_frames)
                                .unwrap_or(0)
                                .min(all_pcm.len().saturating_sub(emitted_samples))
                        };
                        let finalized_end = all_pcm.len().saturating_sub(hold_back_samples);

                        if finalized_end > emitted_samples {
                            let finalized = &all_pcm[emitted_samples..finalized_end];
                            total_samples += finalized.len();
                            sink.push_pcm_chunk(finalized)?;
                            emitted_samples = finalized_end;
                        }
                    }

                    if all_pcm.len() > emitted_samples {
                        let finalized = &all_pcm[emitted_samples..];
                        total_samples += finalized.len();
                        sink.push_pcm_chunk(finalized)?;
                    }

                    Ok((total_samples, t_voc_start.elapsed()))
                },
            );

            let t_roll = Instant::now();
            let codec_rollout = self.transformer.rollout_codec_frames_kv_streaming(
                &prepared.prepared_inputs.prefill_embd,
                &prepared.prepared_inputs.trailing_text_hidden,
                &prepared.prepared_inputs.tts_pad_embed,
                &prepared.prompt_frames,
                req.talker_kv_mode,
                thread_count,
                req.max_audio_frames,
                req.repetition_penalty,
                req.temperature,
                req.top_k,
                req.top_p,
                chunk_size,
                &chunk_tx,
            );
            let codec_rollout_dur = t_roll.elapsed();
            drop(chunk_tx);

            let codec_rollout = codec_rollout?;
            let generated_frames = codec_rollout
                .frames
                .len()
                .saturating_sub(prepared.prefix_frame_count);

            let (generated_samples, vocoder_decode) = vocoder_handle.join().unwrap()?;

            let pipeline_wall_clock = t_pipeline_start.elapsed();
            let post = std::time::Duration::ZERO;
            let sample_rate_hz = self.vocoder.config().sample_rate as u32;
            if let Some(t) = timings {
                t.speaker_encode = prepared.speaker_encode;
                t.tokenize = prepared.tokenize;
                t.prefill_build = prepared.prefill_build;
                t.codec_rollout = codec_rollout_dur;
                t.vocoder_decode = vocoder_decode;
                t.post = post;
                t.codec_rollout_detail = codec_rollout.sub_timings;
                let sequential_sum = codec_rollout_dur + vocoder_decode;
                t.pipeline_overlap = sequential_sum.saturating_sub(pipeline_wall_clock);
                t.first_frame_latency = prepared.speaker_encode
                    + prepared.tokenize
                    + prepared.prefill_build
                    + codec_rollout.first_frame_elapsed;
                t.generated_samples = generated_samples;
                t.sample_rate_hz = sample_rate_hz;
            }

            Ok(StreamingSynthesizeResult {
                sample_rate_hz,
                generated_frames,
                generated_samples,
            })
        })
    }

    fn validate_speaker_embedding(&self, speaker_embedding: &[f32]) -> Result<(), Qwen3TtsError> {
        let expected = self.speaker_embedding_size();
        if speaker_embedding.len() != expected {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "speaker embedding must have {expected} elements"
            )));
        }
        Ok(())
    }
}

pub trait StreamingSynthesis {
    fn push_pcm_chunk(&mut self, pcm_f32: &[f32]) -> Result<(), Qwen3TtsError>;
}

impl<F> StreamingSynthesis for F
where
    F: FnMut(&[f32]) -> Result<(), Qwen3TtsError>,
{
    fn push_pcm_chunk(&mut self, pcm_f32: &[f32]) -> Result<(), Qwen3TtsError> {
        self(pcm_f32)
    }
}

mod qts_request_map;
mod qts_tts_backend;

pub use qts_request_map::{
    QtsVoiceCloneParams, synthesize_request_from_tts, voice_clone_params_from_tts,
};
pub use qts_tts_backend::{QtsTtsConfig, QtsTtsModel};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults() {
        let r = SynthesizeRequest::default();
        assert_eq!(r.temperature, 0.9);
        assert_eq!(r.top_k, 50);
        assert_eq!(r.language_id, 2050);
        assert_eq!(r.talker_kv_mode, TalkerKvMode::F16);
    }

    #[test]
    fn talker_kv_mode_parse_accepts_aliases() {
        assert_eq!(TalkerKvMode::parse("f16").unwrap(), TalkerKvMode::F16);
        assert_eq!(
            TalkerKvMode::parse("turboquant").unwrap(),
            TalkerKvMode::TurboQuant
        );
        assert_eq!(TalkerKvMode::parse("q8").unwrap(), TalkerKvMode::TurboQuant);
    }

    #[test]
    fn recommended_vocoder_chunk_matches_paper_packet() {
        assert_eq!(QTS12HZ_RECOMMENDED_VOCODER_CHUNK_FRAMES, 4);
    }
}
