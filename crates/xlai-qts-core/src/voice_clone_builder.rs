//! Build [`VoiceClonePromptV2`](crate::VoiceClonePromptV2) from raw reference WAV bytes (Rust-native).

use crate::pipeline::reference_codec_encoder::ReferenceCodecEncoder;
use crate::pipeline::speaker_encoder::SpeakerEncoderConfig;
use crate::voice_clone_prompt::{TensorF32, TensorI32, VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION, VoiceClonePromptV2};
use crate::Qwen3TtsError;

/// How reference audio conditions synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VoiceCloneMode {
    /// Speaker embedding only (`x_vector_only_mode`); no `ref_code`.
    #[default]
    XVectorOnly,
    /// In-context learning: requires `ref_text` and `ref_code` from [`ReferenceCodecEncoder`].
    Icl,
}

impl VoiceCloneMode {
    pub fn parse(value: &str) -> Result<Self, Qwen3TtsError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "xvector" | "x_vector" | "x-vector" | "xvector_only" | "x_vector_only" => {
                Ok(Self::XVectorOnly)
            }
            "icl" => Ok(Self::Icl),
            other => Err(Qwen3TtsError::InvalidInput(format!(
                "unknown voice clone mode '{other}' (expected xvector or icl)"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::XVectorOnly => "xvector",
            Self::Icl => "icl",
        }
    }
}

/// Build an x-vector-only prompt using the engine's handcrafted [`SpeakerEncoder`](crate::SpeakerEncoder).
pub fn build_xvector_voice_clone_prompt(
    speaker_encoder: &crate::pipeline::speaker_encoder::SpeakerEncoder,
    speaker_embedding_dim: usize,
    ref_wav_bytes: &[u8],
    model_id: &str,
) -> Result<VoiceClonePromptV2, Qwen3TtsError> {
    let emb = speaker_encoder.encode_wav_bytes(ref_wav_bytes)?;
    if emb.len() != speaker_embedding_dim {
        return Err(Qwen3TtsError::InvalidInput(format!(
            "speaker embedding length {} does not match expected {}",
            emb.len(),
            speaker_embedding_dim
        )));
    }
    Ok(VoiceClonePromptV2 {
        schema_version: VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
        source: "xlai-qts-core/native".into(),
        model_id: model_id.to_owned(),
        speaker_encoder_sample_rate_hz: SpeakerEncoderConfig::default().sample_rate_hz,
        x_vector_only_mode: true,
        icl_mode: false,
        ref_text: String::new(),
        ref_code: None,
        ref_spk_embedding: Some(TensorF32 {
            shape: vec![speaker_embedding_dim as u32],
            values: emb,
        }),
    })
}

/// Build an ICL prompt: `ref_code` from ONNX [`ReferenceCodecEncoder`], embedding from [`SpeakerEncoder`](crate::SpeakerEncoder).
pub fn build_icl_voice_clone_prompt(
    speaker_encoder: &crate::pipeline::speaker_encoder::SpeakerEncoder,
    reference_codec: &ReferenceCodecEncoder,
    speaker_embedding_dim: usize,
    n_codebooks: usize,
    ref_wav_bytes: &[u8],
    ref_text: &str,
    model_id: &str,
) -> Result<VoiceClonePromptV2, Qwen3TtsError> {
    if ref_text.trim().is_empty() {
        return Err(Qwen3TtsError::InvalidInput(
            "ICL voice clone requires non-empty ref_text".into(),
        ));
    }
    let emb = speaker_encoder.encode_wav_bytes(ref_wav_bytes)?;
    if emb.len() != speaker_embedding_dim {
        return Err(Qwen3TtsError::InvalidInput(format!(
            "speaker embedding length {} does not match expected {}",
            emb.len(),
            speaker_embedding_dim
        )));
    }
    let (frames, codebooks, values) = reference_codec.encode_wav_bytes_shape(ref_wav_bytes)?;
    if codebooks != n_codebooks {
        return Err(Qwen3TtsError::InvalidInput(format!(
            "reference codec produced {codebooks} codebooks but the talker expects {n_codebooks}"
        )));
    }
    Ok(VoiceClonePromptV2 {
        schema_version: VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
        source: "xlai-qts-core/native".into(),
        model_id: model_id.to_owned(),
        speaker_encoder_sample_rate_hz: SpeakerEncoderConfig::default().sample_rate_hz,
        x_vector_only_mode: false,
        icl_mode: true,
        ref_text: ref_text.to_owned(),
        ref_code: Some(TensorI32 {
            shape: vec![frames as u32, codebooks as u32],
            values,
        }),
        ref_spk_embedding: Some(TensorF32 {
            shape: vec![speaker_embedding_dim as u32],
            values: emb,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::VoiceCloneMode;

    #[test]
    fn voice_clone_mode_parse() {
        assert_eq!(
            VoiceCloneMode::parse("xvector").unwrap(),
            VoiceCloneMode::XVectorOnly
        );
        assert_eq!(VoiceCloneMode::parse("ICL").unwrap(), VoiceCloneMode::Icl);
        assert!(VoiceCloneMode::parse("bogus").is_err());
    }
}
