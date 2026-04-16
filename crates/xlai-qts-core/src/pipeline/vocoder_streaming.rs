//! Overlap-add stitching for pipelined QTS 12Hz vocoder chunks (stateless ONNX decode).
//!
//! Qwen3-TTS-12Hz targets a causal codec; chunking the vocoder for throughput still benefits from
//! a small temporal overlap between chunks. See [`QTS12HZ_RECOMMENDED_VOCODER_CHUNK_FRAMES`] and
//! `docs/qts/vocoder-streaming.md`.

use crate::Qwen3TtsError;
use crate::pipeline::tts_transformer::VocoderChunk;
use crate::pipeline::vocoder::{Vocoder, VocoderGraphTemplate};

/// Recommended vocoder packet size in codec frames for 12.5 Hz streaming (matches Qwen3-TTS paper: 4 tokens ≈ 320 ms).
pub const QTS12HZ_RECOMMENDED_VOCODER_CHUNK_FRAMES: usize = 4;

/// Overlap between consecutive vocoder windows (codec frames). Default `1` for 12Hz; set `QWEN3_TTS_VOCODER_OVERLAP_FRAMES` (e.g. `3`) for a stronger quality fallback.
#[must_use]
pub fn overlap_frames_for_qts12hz() -> usize {
    std::env::var("QWEN3_TTS_VOCODER_OVERLAP_FRAMES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .min(8)
}

/// Carries overlap state and scratch for [`OverlapAddChunkDecoder::process_generated_chunk`].
pub struct OverlapAddChunkDecoder {
    overlap_frames: usize,
    n_codebooks: usize,
    vocoder_thread_count: usize,
    prev_codes: Vec<i32>,
    prev_n_frames: usize,
    combined_scratch: Vec<i32>,
    template: Option<VocoderGraphTemplate>,
    standard_decode_frames: usize,
}

impl OverlapAddChunkDecoder {
    #[must_use]
    pub fn overlap_frames(&self) -> usize {
        self.overlap_frames
    }

    #[must_use]
    pub fn new(vocoder: &Vocoder, chunk_size: usize, vocoder_thread_count: usize) -> Self {
        let overlap_frames = overlap_frames_for_qts12hz();
        let chunk_size = chunk_size.max(1);
        let standard_decode_frames = overlap_frames.saturating_add(chunk_size).max(1);
        let _ = vocoder;
        Self {
            overlap_frames,
            n_codebooks: vocoder.config().n_codebooks as usize,
            vocoder_thread_count,
            prev_codes: Vec::new(),
            prev_n_frames: 0,
            combined_scratch: Vec::new(),
            template: None,
            standard_decode_frames,
        }
    }

    /// Apply a prefix-only chunk: updates overlap tail without decoding or emitting PCM.
    pub fn absorb_prefix_warmup(&mut self, chunk: &VocoderChunk) {
        debug_assert!(chunk.prefix_warmup_only);
        self.prev_codes.clone_from(&chunk.codes);
        self.prev_n_frames = chunk.n_frames;
    }

    /// Decode one non-prefix chunk and merge into `all_pcm` (overlap-add when context exists).
    pub fn process_generated_chunk(
        &mut self,
        vocoder: &Vocoder,
        chunk: &VocoderChunk,
        all_pcm: &mut Vec<f32>,
    ) -> Result<(), Qwen3TtsError> {
        if chunk.prefix_warmup_only {
            self.absorb_prefix_warmup(chunk);
            return Ok(());
        }

        let ctx_frames = self.overlap_frames.min(self.prev_n_frames);

        let (codes_slice, decode_frames) = if ctx_frames > 0 {
            self.combined_scratch.clear();
            self.combined_scratch
                .reserve(ctx_frames * self.n_codebooks + chunk.codes.len());
            let ctx_start = self.prev_codes.len() - ctx_frames * self.n_codebooks;
            self.combined_scratch
                .extend_from_slice(&self.prev_codes[ctx_start..]);
            self.combined_scratch.extend_from_slice(&chunk.codes);
            (
                self.combined_scratch.as_slice(),
                ctx_frames + chunk.n_frames,
            )
        } else {
            (chunk.codes.as_slice(), chunk.n_frames)
        };

        let audio = if decode_frames == self.standard_decode_frames {
            if self.template.is_none() {
                self.template = Some(vocoder.build_decode_template(self.standard_decode_frames)?);
            }
            vocoder.decode_with_template(
                self.template.as_mut().ok_or_else(|| {
                    Qwen3TtsError::InvalidInput("vocoder template missing".into())
                })?,
                codes_slice,
                self.vocoder_thread_count,
            )?
        } else {
            vocoder.decode(codes_slice, decode_frames, self.vocoder_thread_count)?
        };

        if ctx_frames > 0 && !all_pcm.is_empty() {
            let total_frames = ctx_frames + chunk.n_frames;
            let overlap_samples = (audio.len() * ctx_frames / total_frames).min(all_pcm.len());
            let start = all_pcm.len() - overlap_samples;
            for i in 0..overlap_samples {
                let t = (i as f32 + 0.5) / overlap_samples as f32;
                all_pcm[start + i] = all_pcm[start + i] * (1.0 - t) + audio[i] * t;
            }
            all_pcm.extend_from_slice(&audio[overlap_samples..]);
        } else if ctx_frames > 0 && all_pcm.is_empty() {
            // First generated chunk after prefix warmup: drop context portion of the decode only.
            let total_frames = ctx_frames + chunk.n_frames;
            let overlap_samples = audio.len() * ctx_frames / total_frames;
            all_pcm.extend_from_slice(&audio[overlap_samples..]);
        } else {
            all_pcm.extend_from_slice(&audio);
        }

        self.prev_codes.clone_from(&chunk.codes);
        self.prev_n_frames = chunk.n_frames;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommended_chunk_is_four() {
        assert_eq!(QTS12HZ_RECOMMENDED_VOCODER_CHUNK_FRAMES, 4);
    }
}
