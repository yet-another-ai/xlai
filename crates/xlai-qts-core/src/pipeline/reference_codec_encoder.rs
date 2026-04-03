//! ONNX Runtime wrapper for the exported Qwen3-TTS speech-tokenizer **encode** path (ICL `ref_code`).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;

use super::speaker_encoder::decode_wav_mono;
use crate::Qwen3TtsError;

fn ort_err(err: impl std::fmt::Display) -> Qwen3TtsError {
    Qwen3TtsError::Ort(err.to_string())
}

/// Matches the subset of Hugging Face `preprocessor_config.json` we need for Rust-side preprocessing.
#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceCodecPreprocessConfig {
    /// Model input sample rate (Hz), e.g. 24000.
    pub sampling_rate: u32,
    /// When true, apply per-utterance zero-mean / unit-variance normalization (Wav2Vec2-style).
    #[serde(default)]
    pub do_normalize: bool,
    /// RVQ depth (codebooks per frame), e.g. 16 for 12Hz Qwen3-TTS tokenizer.
    #[serde(default = "default_num_quantizers")]
    pub num_quantizers: u32,
}

fn default_num_quantizers() -> u32 {
    16
}

impl Default for ReferenceCodecPreprocessConfig {
    fn default() -> Self {
        Self {
            sampling_rate: 24_000,
            do_normalize: true,
            num_quantizers: 16,
        }
    }
}

/// Loads from `qwen3-tts-reference-codec-preprocess.json` next to the ONNX artifact.
#[derive(Debug)]
pub struct ReferenceCodecEncoder {
    session: Mutex<Session>,
    model_path: PathBuf,
    preprocess: ReferenceCodecPreprocessConfig,
}

impl ReferenceCodecEncoder {
    /// Load encoder + preprocessor config. The JSON path defaults to sibling of `onnx_path` with
    /// `-preprocess.json` suffix when `preprocess_json` is `None`.
    pub fn load_from_onnx(
        onnx_path: impl AsRef<Path>,
        preprocess_json: Option<&Path>,
    ) -> Result<Self, Qwen3TtsError> {
        let onnx_path = onnx_path.as_ref().to_path_buf();
        if !onnx_path.is_file() {
            return Err(Qwen3TtsError::ModelFile(onnx_path));
        }

        let json_path = preprocess_json
            .map(Path::to_path_buf)
            .unwrap_or_else(|| {
                onnx_path
                    .with_file_name("qwen3-tts-reference-codec-preprocess.json")
            });
        let preprocess: ReferenceCodecPreprocessConfig = if json_path.is_file() {
            let raw = std::fs::read_to_string(&json_path).map_err(|e| {
                Qwen3TtsError::InvalidInput(format!(
                    "failed to read reference codec preprocess config {}: {e}",
                    json_path.display()
                ))
            })?;
            serde_json::from_str(&raw).map_err(|e| {
                Qwen3TtsError::InvalidInput(format!(
                    "invalid reference codec preprocess JSON {}: {e}",
                    json_path.display()
                ))
            })?
        } else {
            return Err(Qwen3TtsError::ModelFile(json_path));
        };

        let _ = ort::init().commit();
        let mut builder = Session::builder().map_err(ort_err)?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?;
        let session = builder.commit_from_file(&onnx_path).map_err(ort_err)?;

        Ok(Self {
            session: Mutex::new(session),
            model_path: onnx_path,
            preprocess,
        })
    }

    #[must_use]
    pub fn preprocess_config(&self) -> &ReferenceCodecPreprocessConfig {
        &self.preprocess
    }

    /// Mono WAV bytes → `ref_code` values in row-major `[frames * num_quantizers]` layout.
    pub fn encode_wav_bytes(&self, wav_bytes: &[u8]) -> Result<Vec<i32>, Qwen3TtsError> {
        let (mono, rate) = decode_wav_mono(wav_bytes)?;
        let samples = resample_linear_internal(&mono, rate, self.preprocess.sampling_rate);
        if samples.is_empty() {
            return Err(Qwen3TtsError::InvalidInput(
                "reference audio is empty after resampling".into(),
            ));
        }
        let mut input_values = samples;
        if self.preprocess.do_normalize {
            z_normalize(&mut input_values);
        }
        let t = input_values.len();
        let mask = vec![1_i64; t];

        let input_iv = Tensor::from_array(([1usize, t], input_values)).map_err(ort_err)?;
        let input_mask = Tensor::from_array(([1usize, t], mask.clone())).map_err(ort_err)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| Qwen3TtsError::InvalidInput("reference codec session lock poisoned".into()))?;
        let in_names = session.inputs();
        if in_names.len() < 2 {
            return Err(Qwen3TtsError::InvalidOnnx(self.model_path.clone()));
        }
        let in0 = in_names[0].name().to_owned();
        let in1 = in_names[1].name().to_owned();
        let outs = session
            .run(ort::inputs![
                in0 => input_iv,
                in1 => input_mask,
            ])
            .map_err(ort_err)?;

        let first = outs
            .iter()
            .next()
            .ok_or_else(|| Qwen3TtsError::InvalidOnnx(self.model_path.clone()))?;
        let (_shape, data) = first.1.try_extract_tensor::<i64>().map_err(ort_err)?;
        Ok(data.iter().map(|&v| v as i32).collect())
    }

    /// Returns `(frames, codebooks)` for a successful encode (2D logical layout).
    pub fn encode_wav_bytes_shape(
        &self,
        wav_bytes: &[u8],
    ) -> Result<(usize, usize, Vec<i32>), Qwen3TtsError> {
        let values = self.encode_wav_bytes(wav_bytes)?;
        let q = self.preprocess.num_quantizers as usize;
        if q == 0 {
            return Err(Qwen3TtsError::InvalidInput(
                "reference codec reports zero quantizers".into(),
            ));
        }
        if values.len() % q != 0 {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "reference codec output length {} is not divisible by num_quantizers {}",
                values.len(),
                q
            )));
        }
        let frames = values.len() / q;
        Ok((frames, q, values))
    }
}

fn z_normalize(samples: &mut [f32]) {
    let n = samples.len() as f32;
    if n <= 0.0 {
        return;
    }
    let mean = samples.iter().copied().sum::<f32>() / n;
    let var = samples
        .iter()
        .map(|s| {
            let d = *s - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    let std = (var + 1e-7).sqrt();
    if std <= 1e-12 {
        return;
    }
    for s in samples.iter_mut() {
        *s = (*s - mean) / std;
    }
}

// Expose linear resample for reference codec (speaker_encoder keeps its own copy private).
pub(crate) fn resample_linear_internal(samples: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if input_rate == output_rate || samples.len() < 2 {
        return samples.to_vec();
    }
    let ratio = output_rate as f64 / input_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for idx in 0..out_len {
        let src_pos = idx as f64 / ratio;
        let src_idx = src_pos.floor() as usize;
        let frac = (src_pos - src_idx as f64) as f32;
        let left = samples[src_idx.min(samples.len() - 1)];
        let right = samples[(src_idx + 1).min(samples.len() - 1)];
        out.push(left + (right - left) * frac);
    }
    out
}
