//! WAV-only reference speaker encoder for tone-transfer conditioning.

use std::f32::consts::PI;
use std::io::Cursor;
use std::sync::Arc;

use hound::{SampleFormat, WavReader};
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::Qwen3TtsError;

#[derive(Debug, Clone)]
pub struct SpeakerEncoderConfig {
    pub sample_rate_hz: u32,
    pub frame_len: usize,
    pub hop_len: usize,
    pub fft_len: usize,
    pub mel_bins: usize,
    pub trim_amplitude: f32,
    pub min_duration_ms: u32,
    pub max_duration_ms: u32,
    pub output_rms: f32,
}

impl Default for SpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 16_000,
            frame_len: 400,
            hop_len: 160,
            fft_len: 512,
            mel_bins: 32,
            trim_amplitude: 0.02,
            min_duration_ms: 250,
            max_duration_ms: 8_000,
            output_rms: 0.25,
        }
    }
}

pub struct SpeakerEncoder {
    target_dim: usize,
    config: SpeakerEncoderConfig,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>,
}

impl SpeakerEncoder {
    pub fn new(target_dim: usize) -> Result<Self, Qwen3TtsError> {
        Self::with_config(target_dim, SpeakerEncoderConfig::default())
    }

    pub fn with_config(
        target_dim: usize,
        config: SpeakerEncoderConfig,
    ) -> Result<Self, Qwen3TtsError> {
        if target_dim == 0 {
            return Err(Qwen3TtsError::InvalidInput(
                "speaker encoder target dimension must be non-zero".into(),
            ));
        }
        if config.fft_len < config.frame_len || config.hop_len == 0 || config.mel_bins == 0 {
            return Err(Qwen3TtsError::InvalidInput(
                "speaker encoder config is invalid".into(),
            ));
        }

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(config.fft_len);
        let window = (0..config.frame_len)
            .map(|idx| 0.5 - 0.5 * (2.0 * PI * idx as f32 / config.frame_len as f32).cos())
            .collect::<Vec<_>>();
        let mel_filters = build_mel_filters(config.sample_rate_hz, config.fft_len, config.mel_bins);

        Ok(Self {
            target_dim,
            config,
            fft,
            window,
            mel_filters,
        })
    }

    pub fn target_dim(&self) -> usize {
        self.target_dim
    }

    pub fn encode_wav_bytes(&self, wav_bytes: &[u8]) -> Result<Vec<f32>, Qwen3TtsError> {
        let (mono, input_rate) = decode_wav_mono(wav_bytes)?;
        let preprocessed = self.preprocess(&mono, input_rate)?;
        let features = self.extract_features(&preprocessed)?;
        Ok(self.project_features(&features))
    }

    fn preprocess(&self, mono: &[f32], input_rate: u32) -> Result<Vec<f32>, Qwen3TtsError> {
        if mono.is_empty() {
            return Err(Qwen3TtsError::InvalidInput(
                "reference audio must contain at least one sample".into(),
            ));
        }
        if input_rate == 0 {
            return Err(Qwen3TtsError::InvalidInput(
                "reference audio has an invalid sample rate".into(),
            ));
        }

        let peak = mono
            .iter()
            .copied()
            .filter(|sample| sample.is_finite())
            .map(f32::abs)
            .fold(0.0f32, f32::max);
        if peak <= 1e-6 {
            return Err(Qwen3TtsError::InvalidInput(
                "reference audio is silent or degenerate".into(),
            ));
        }

        let normalized = mono
            .iter()
            .map(|sample| (sample / peak).clamp(-0.95, 0.95))
            .collect::<Vec<_>>();
        let trimmed = trim_silence(&normalized, self.config.trim_amplitude)?;
        let resampled = resample_linear(&trimmed, input_rate, self.config.sample_rate_hz);
        let cropped = crop_duration(
            &resampled,
            self.config.sample_rate_hz,
            self.config.min_duration_ms,
            self.config.max_duration_ms,
        )?;
        Ok(cropped)
    }

    fn extract_features(&self, samples: &[f32]) -> Result<Vec<f32>, Qwen3TtsError> {
        if samples.len() < self.config.frame_len {
            return Err(Qwen3TtsError::InvalidInput(
                "reference audio is too short after preprocessing".into(),
            ));
        }

        let mut mel_sum = vec![0.0f32; self.config.mel_bins];
        let mut mel_sq_sum = vec![0.0f32; self.config.mel_bins];
        let mut energy_sum = 0.0f32;
        let mut energy_sq_sum = 0.0f32;
        let mut zcr_sum = 0.0f32;
        let mut zcr_sq_sum = 0.0f32;
        let mut centroid_sum = 0.0f32;
        let mut centroid_sq_sum = 0.0f32;
        let mut frame_count = 0usize;

        let mut scratch = vec![Complex32::default(); self.config.fft_len];
        for start in (0..=samples.len() - self.config.frame_len).step_by(self.config.hop_len) {
            for value in &mut scratch {
                *value = Complex32::default();
            }
            let frame = &samples[start..start + self.config.frame_len];
            let mut prev_sign = frame[0].is_sign_positive();
            let mut zcr = 0.0f32;
            let mut energy = 0.0f32;
            for (idx, sample) in frame.iter().copied().enumerate() {
                energy += sample * sample;
                let sign = sample.is_sign_positive();
                if idx > 0 && sign != prev_sign {
                    zcr += 1.0;
                }
                prev_sign = sign;
                scratch[idx].re = sample * self.window[idx];
            }
            self.fft.process(&mut scratch);

            let mut power = vec![0.0f32; self.config.fft_len / 2 + 1];
            for (idx, bin) in scratch.iter().take(power.len()).enumerate() {
                power[idx] = bin.norm_sqr();
            }

            let mut mel = vec![0.0f32; self.config.mel_bins];
            for (mel_idx, filter) in self.mel_filters.iter().enumerate() {
                let value = filter
                    .iter()
                    .zip(power.iter())
                    .map(|(weight, pow)| weight * pow)
                    .sum::<f32>();
                mel[mel_idx] = (value + 1e-6).ln();
                mel_sum[mel_idx] += mel[mel_idx];
                mel_sq_sum[mel_idx] += mel[mel_idx] * mel[mel_idx];
            }

            energy /= frame.len() as f32;
            zcr /= frame.len() as f32;
            let centroid =
                spectral_centroid(&power, self.config.sample_rate_hz, self.config.fft_len);
            energy_sum += energy;
            energy_sq_sum += energy * energy;
            zcr_sum += zcr;
            zcr_sq_sum += zcr * zcr;
            centroid_sum += centroid;
            centroid_sq_sum += centroid * centroid;
            frame_count += 1;
        }

        if frame_count < 2 {
            return Err(Qwen3TtsError::InvalidInput(
                "reference audio does not contain enough voiced frames".into(),
            ));
        }

        let mut features = Vec::with_capacity(self.config.mel_bins * 2 + 6);
        for mel_idx in 0..self.config.mel_bins {
            let mean = mel_sum[mel_idx] / frame_count as f32;
            let variance = (mel_sq_sum[mel_idx] / frame_count as f32 - mean * mean).max(0.0);
            features.push(mean);
            features.push(variance.sqrt());
        }
        push_mean_std(&mut features, energy_sum, energy_sq_sum, frame_count);
        push_mean_std(&mut features, zcr_sum, zcr_sq_sum, frame_count);
        push_mean_std(&mut features, centroid_sum, centroid_sq_sum, frame_count);
        normalize_vector(&mut features);
        Ok(features)
    }

    fn project_features(&self, features: &[f32]) -> Vec<f32> {
        let mut projected = vec![0.0f32; self.target_dim];
        let norm = (features.len() as f32).sqrt();
        for (out_idx, slot) in projected.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (feat_idx, feature) in features.iter().copied().enumerate() {
                acc += feature * projection_weight(out_idx, feat_idx);
            }
            *slot = acc / norm;
        }

        let mean = projected.iter().sum::<f32>() / projected.len() as f32;
        for value in &mut projected {
            *value -= mean;
        }

        let rms = (projected.iter().map(|v| v * v).sum::<f32>() / projected.len() as f32).sqrt();
        if rms > 1e-6 {
            let scale = self.config.output_rms / rms;
            for value in &mut projected {
                *value *= scale;
            }
        }
        projected
    }
}

pub(crate) fn decode_wav_mono(wav_bytes: &[u8]) -> Result<(Vec<f32>, u32), Qwen3TtsError> {
    let mut reader = WavReader::new(Cursor::new(wav_bytes)).map_err(|err| {
        Qwen3TtsError::InvalidInput(format!("reference audio must be a valid WAV file: {err}"))
    })?;
    let spec = reader.spec();
    if spec.channels == 0 {
        return Err(Qwen3TtsError::InvalidInput(
            "reference WAV must declare at least one channel".into(),
        ));
    }
    let channels = spec.channels as usize;

    let interleaved = match spec.sample_format {
        SampleFormat::Float if spec.bits_per_sample == 32 => reader
            .samples::<f32>()
            .map(|sample| {
                sample.map_err(|err| {
                    Qwen3TtsError::InvalidInput(format!("failed to decode WAV float sample: {err}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        SampleFormat::Int if (1..=32).contains(&spec.bits_per_sample) => {
            let scale = ((1u64 << (spec.bits_per_sample - 1)) - 1) as f32;
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample
                        .map(|value| (value as f32 / scale).clamp(-1.0, 1.0))
                        .map_err(|err| {
                            Qwen3TtsError::InvalidInput(format!(
                                "failed to decode WAV integer sample: {err}"
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        _ => {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "unsupported WAV format: {:?} {}-bit",
                spec.sample_format, spec.bits_per_sample
            )));
        }
    };

    if interleaved.len() < channels {
        return Err(Qwen3TtsError::InvalidInput(
            "reference WAV does not contain enough audio frames".into(),
        ));
    }

    let mono = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect::<Vec<_>>();
    Ok((mono, spec.sample_rate))
}

fn trim_silence(samples: &[f32], threshold: f32) -> Result<Vec<f32>, Qwen3TtsError> {
    let first = samples.iter().position(|sample| sample.abs() >= threshold);
    let last = samples.iter().rposition(|sample| sample.abs() >= threshold);
    match (first, last) {
        (Some(start), Some(end)) if start <= end => Ok(samples[start..=end].to_vec()),
        _ => Err(Qwen3TtsError::InvalidInput(
            "reference audio does not contain enough voiced content".into(),
        )),
    }
}

fn resample_linear(samples: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
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

fn crop_duration(
    samples: &[f32],
    sample_rate_hz: u32,
    min_duration_ms: u32,
    max_duration_ms: u32,
) -> Result<Vec<f32>, Qwen3TtsError> {
    let min_len = (sample_rate_hz as usize * min_duration_ms as usize) / 1000;
    if samples.len() < min_len {
        return Err(Qwen3TtsError::InvalidInput(format!(
            "reference audio is too short; need at least {min_duration_ms} ms of voiced audio"
        )));
    }

    let max_len = (sample_rate_hz as usize * max_duration_ms as usize) / 1000;
    if samples.len() <= max_len {
        return Ok(samples.to_vec());
    }

    let start = (samples.len() - max_len) / 2;
    Ok(samples[start..start + max_len].to_vec())
}

fn build_mel_filters(sample_rate_hz: u32, fft_len: usize, mel_bins: usize) -> Vec<Vec<f32>> {
    let nyquist = sample_rate_hz as f32 / 2.0;
    let min_mel = hz_to_mel(20.0);
    let max_mel = hz_to_mel(nyquist.min(7_600.0));
    let mel_points = (0..mel_bins + 2)
        .map(|idx| {
            let t = idx as f32 / (mel_bins + 1) as f32;
            mel_to_hz(min_mel + t * (max_mel - min_mel))
        })
        .collect::<Vec<_>>();

    let fft_bins = fft_len / 2 + 1;
    let bin_freq = |bin: usize| bin as f32 * sample_rate_hz as f32 / fft_len as f32;
    let mut filters = vec![vec![0.0f32; fft_bins]; mel_bins];
    for mel_idx in 0..mel_bins {
        let left = mel_points[mel_idx];
        let center = mel_points[mel_idx + 1];
        let right = mel_points[mel_idx + 2];
        for (bin, weight_cell) in filters[mel_idx].iter_mut().enumerate().take(fft_bins) {
            let freq = bin_freq(bin);
            let weight = if freq >= left && freq < center {
                (freq - left) / (center - left + 1e-6)
            } else if freq >= center && freq <= right {
                (right - freq) / (right - center + 1e-6)
            } else {
                0.0
            };
            *weight_cell = weight.max(0.0);
        }
    }
    filters
}

fn spectral_centroid(power: &[f32], sample_rate_hz: u32, fft_len: usize) -> f32 {
    let bin_hz = sample_rate_hz as f32 / fft_len as f32;
    let weighted = power
        .iter()
        .enumerate()
        .map(|(idx, value)| idx as f32 * bin_hz * *value)
        .sum::<f32>();
    let total = power.iter().sum::<f32>();
    if total <= 1e-12 {
        0.0
    } else {
        weighted / total
    }
}

fn push_mean_std(features: &mut Vec<f32>, sum: f32, sq_sum: f32, count: usize) {
    let mean = sum / count as f32;
    let variance = (sq_sum / count as f32 - mean * mean).max(0.0);
    features.push(mean);
    features.push(variance.sqrt());
}

fn normalize_vector(values: &mut [f32]) {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    for value in values.iter_mut() {
        *value -= mean;
    }
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1e-6 {
        for value in values.iter_mut() {
            *value /= norm;
        }
    }
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn projection_weight(output_idx: usize, feature_idx: usize) -> f32 {
    let mut state = (output_idx as u64) << 32 | feature_idx as u64;
    state = state.wrapping_add(0x9e3779b97f4a7c15);
    state = (state ^ (state >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94d049bb133111eb);
    state ^= state >> 31;
    let normalized = ((state >> 40) as f32) / ((1u32 << 24) as f32);
    normalized * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use hound::{SampleFormat, WavSpec, WavWriter};

    use super::{PI, SpeakerEncoder};

    fn sine_wav(freq_hz: f32, sample_rate_hz: u32, seconds: f32) -> Vec<u8> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: sample_rate_hz,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut cursor = Cursor::new(Vec::new());
        let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
        let total_samples = (sample_rate_hz as f32 * seconds) as usize;
        for idx in 0..total_samples {
            let t = idx as f32 / sample_rate_hz as f32;
            let sample = (2.0 * PI * freq_hz * t).sin() * 0.5;
            writer
                .write_sample((sample * i16::MAX as f32) as i16)
                .unwrap();
        }
        writer.finalize().unwrap();
        cursor.into_inner()
    }

    #[test]
    fn rejects_invalid_wav() {
        let encoder = SpeakerEncoder::new(64).unwrap();
        let err = encoder.encode_wav_bytes(b"not a wav").unwrap_err();
        assert!(err.to_string().contains("valid WAV"));
    }

    #[test]
    fn rejects_too_short_audio() {
        let encoder = SpeakerEncoder::new(64).unwrap();
        let wav = sine_wav(220.0, 16_000, 0.05);
        let err = encoder.encode_wav_bytes(&wav).unwrap_err();
        assert!(err.to_string().contains("too short"));
    }

    #[test]
    fn encodes_reference_wav_to_target_dimension() {
        let encoder = SpeakerEncoder::new(128).unwrap();
        let wav = sine_wav(220.0, 16_000, 1.0);
        let embedding = encoder.encode_wav_bytes(&wav).unwrap();
        assert_eq!(embedding.len(), 128);
        assert!(embedding.iter().any(|value| value.abs() > 1e-4));
    }

    #[test]
    fn different_reference_audio_changes_embedding() {
        let encoder = SpeakerEncoder::new(128).unwrap();
        let low = encoder
            .encode_wav_bytes(&sine_wav(220.0, 16_000, 1.0))
            .unwrap();
        let high = encoder
            .encode_wav_bytes(&sine_wav(660.0, 16_000, 1.0))
            .unwrap();
        let diff = low
            .iter()
            .zip(high.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum::<f32>();
        assert!(diff > 1.0, "embedding diff too small: {diff}");
    }
}
