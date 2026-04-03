use crate::Qwen3TtsError;
use serde::{Deserialize, Serialize};
use std::io::Cursor;

pub const VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorI32 {
    pub shape: Vec<u32>,
    pub values: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorF32 {
    pub shape: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceClonePromptV2 {
    pub schema_version: u32,
    pub source: String,
    pub model_id: String,
    pub speaker_encoder_sample_rate_hz: u32,
    pub x_vector_only_mode: bool,
    pub icl_mode: bool,
    pub ref_text: String,
    pub ref_code: Option<TensorI32>,
    pub ref_spk_embedding: Option<TensorF32>,
}

fn validate_shape(shape: &[u32], values_len: usize, field_name: &str) -> Result<(), Qwen3TtsError> {
    if shape.is_empty() {
        return Err(Qwen3TtsError::InvalidInput(format!(
            "{field_name}.shape must not be empty"
        )));
    }
    let element_count = shape.iter().copied().try_fold(1usize, |acc, dim| {
        let dim = usize::try_from(dim).map_err(|_| {
            Qwen3TtsError::InvalidInput(format!("{field_name}.shape contains out-of-range dim"))
        })?;
        if dim == 0 {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "{field_name}.shape must not contain zero"
            )));
        }
        acc.checked_mul(dim).ok_or_else(|| {
            Qwen3TtsError::InvalidInput(format!("{field_name}.shape overflows element count"))
        })
    })?;
    if element_count != values_len {
        return Err(Qwen3TtsError::InvalidInput(format!(
            "{field_name}.values length {values_len} does not match shape product {element_count}"
        )));
    }
    Ok(())
}

impl TensorI32 {
    fn validate(&self, field_name: &str) -> Result<(), Qwen3TtsError> {
        validate_shape(&self.shape, self.values.len(), field_name)
    }
}

impl TensorF32 {
    fn validate(&self, field_name: &str) -> Result<(), Qwen3TtsError> {
        validate_shape(&self.shape, self.values.len(), field_name)
    }
}

impl VoiceClonePromptV2 {
    pub fn from_cbor_slice(bytes: &[u8]) -> Result<Self, Qwen3TtsError> {
        let prompt: Self = ciborium::from_reader(Cursor::new(bytes)).map_err(|err| {
            Qwen3TtsError::InvalidInput(format!("invalid voice clone prompt CBOR: {err}"))
        })?;
        prompt.validate()?;
        Ok(prompt)
    }

    pub fn to_cbor_vec(&self) -> Result<Vec<u8>, Qwen3TtsError> {
        self.validate()?;
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes).map_err(|err| {
            Qwen3TtsError::InvalidInput(format!("failed to encode voice clone prompt as CBOR: {err}"))
        })?;
        Ok(bytes)
    }

    pub fn validate(&self) -> Result<(), Qwen3TtsError> {
        if self.schema_version != VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "unsupported voice clone prompt schema_version: {}",
                self.schema_version
            )));
        }

        let speaker = self.ref_spk_embedding.as_ref().ok_or_else(|| {
            Qwen3TtsError::InvalidInput("voice clone prompt ref_spk_embedding is required".into())
        })?;
        speaker.validate("voice clone prompt ref_spk_embedding")?;
        if speaker.shape.len() != 1 {
            return Err(Qwen3TtsError::InvalidInput(
                "voice clone prompt ref_spk_embedding must be a 1D tensor".into(),
            ));
        }

        match (self.x_vector_only_mode, self.icl_mode) {
            (true, false) => {
                if let Some(ref_code) = &self.ref_code {
                    ref_code.validate("voice clone prompt ref_code")?;
                    if !ref_code.values.is_empty() {
                        return Err(Qwen3TtsError::InvalidInput(
                            "voice clone prompt ref_code must be empty in x-vector-only mode"
                                .into(),
                        ));
                    }
                }
            }
            (false, true) => {
                if self.ref_text.trim().is_empty() {
                    return Err(Qwen3TtsError::InvalidInput(
                        "voice clone prompt ref_text is required in icl_mode".into(),
                    ));
                }
                let ref_code = self.ref_code.as_ref().ok_or_else(|| {
                    Qwen3TtsError::InvalidInput(
                        "voice clone prompt ref_code is required in icl_mode".into(),
                    )
                })?;
                ref_code.validate("voice clone prompt ref_code")?;
                if ref_code.shape.len() != 2 {
                    return Err(Qwen3TtsError::InvalidInput(
                        "voice clone prompt ref_code must be a 2D tensor [frames, codebooks]"
                            .into(),
                    ));
                }
            }
            _ => {
                return Err(Qwen3TtsError::InvalidInput(
                    "voice clone prompt must use exactly one of x_vector_only_mode or icl_mode"
                        .into(),
                ));
            }
        }

        Ok(())
    }

    #[must_use]
    pub fn speaker_embedding(&self) -> &[f32] {
        self.ref_spk_embedding
            .as_ref()
            .map(|tensor| tensor.values.as_slice())
            .unwrap_or(&[])
    }

    #[must_use]
    pub fn speaker_embedding_dim(&self) -> Option<usize> {
        self.ref_spk_embedding
            .as_ref()
            .and_then(|tensor| tensor.shape.first().copied())
            .map(|dim| dim as usize)
    }

    #[must_use]
    pub fn ref_code_shape(&self) -> Option<(usize, usize)> {
        self.ref_code.as_ref().and_then(|tensor| {
            if tensor.shape.len() == 2 {
                Some((tensor.shape[0] as usize, tensor.shape[1] as usize))
            } else {
                None
            }
        })
    }

    #[must_use]
    pub fn ref_code_values(&self) -> Option<&[i32]> {
        self.ref_code
            .as_ref()
            .map(|tensor| tensor.values.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::{TensorF32, TensorI32, VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION, VoiceClonePromptV2};

    #[test]
    fn parses_prompt_cbor() {
        let prompt = VoiceClonePromptV2 {
            schema_version: VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
            source: "unit-test".into(),
            model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-Base".into(),
            speaker_encoder_sample_rate_hz: 24_000,
            x_vector_only_mode: false,
            icl_mode: true,
            ref_text: "hello".into(),
            ref_code: Some(TensorI32 {
                shape: vec![2, 3].into_iter().map(|v| v as u32).collect(),
                values: vec![1, 2, 3, 4, 5, 6],
            }),
            ref_spk_embedding: Some(TensorF32 {
                shape: vec![4],
                values: vec![0.1, 0.2, 0.3, 0.4],
            }),
        };
        let bytes = prompt.to_cbor_vec().unwrap();
        let parsed = VoiceClonePromptV2::from_cbor_slice(&bytes).unwrap();
        assert_eq!(parsed, prompt);
    }

    #[test]
    fn rejects_bad_schema_version() {
        let prompt = VoiceClonePromptV2 {
            schema_version: 999,
            source: String::new(),
            model_id: String::new(),
            speaker_encoder_sample_rate_hz: 0,
            x_vector_only_mode: true,
            icl_mode: false,
            ref_text: String::new(),
            ref_code: None,
            ref_spk_embedding: Some(TensorF32 {
                shape: vec![2],
                values: vec![0.1, 0.2],
            }),
        };
        let err = prompt.to_cbor_vec().unwrap_err();
        assert!(
            err.to_string()
                .contains("unsupported voice clone prompt schema_version")
        );
    }

    #[test]
    fn rejects_missing_icl_fields() {
        let prompt = VoiceClonePromptV2 {
            schema_version: VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
            source: String::new(),
            model_id: String::new(),
            speaker_encoder_sample_rate_hz: 0,
            x_vector_only_mode: false,
            icl_mode: true,
            ref_text: String::new(),
            ref_code: None,
            ref_spk_embedding: Some(TensorF32 {
                shape: vec![2],
                values: vec![0.1, 0.2],
            }),
        };
        let err = prompt.validate().unwrap_err();
        assert!(err.to_string().contains("ref_text is required in icl_mode"));
    }

    /// Golden `.cbor` files are optional (often gitignored); this covers the same invariants via encode/decode.
    #[test]
    fn roundtrips_xvector_only_voice_clone_prompt() {
        let prompt = VoiceClonePromptV2 {
            schema_version: VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
            source: "unit-test".into(),
            model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-Base".into(),
            speaker_encoder_sample_rate_hz: 24_000,
            x_vector_only_mode: true,
            icl_mode: false,
            ref_text: String::new(),
            ref_code: None,
            ref_spk_embedding: Some(TensorF32 {
                shape: vec![1024],
                values: vec![0.0; 1024],
            }),
        };
        let bytes = prompt.to_cbor_vec().unwrap();
        let parsed = VoiceClonePromptV2::from_cbor_slice(&bytes).unwrap();
        assert!(parsed.x_vector_only_mode);
        assert!(!parsed.icl_mode);
        assert_eq!(parsed.ref_text, "");
        assert_eq!(parsed.speaker_embedding_dim(), Some(1024));
        assert_eq!(parsed.ref_code_shape(), None);
        assert!(parsed.ref_code_values().is_none());
    }

    #[test]
    fn roundtrips_icl_voice_clone_prompt() {
        let n_frames = 105u32;
        let n_codebooks = 16u32;
        let values_len = (n_frames * n_codebooks) as usize;
        let prompt = VoiceClonePromptV2 {
            schema_version: VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
            source: "unit-test".into(),
            model_id: "Qwen/Qwen3-TTS-12Hz-0.6B-Base".into(),
            speaker_encoder_sample_rate_hz: 24_000,
            x_vector_only_mode: false,
            icl_mode: true,
            ref_text: "hello".into(),
            ref_code: Some(TensorI32 {
                shape: vec![n_frames, n_codebooks],
                values: vec![0_i32; values_len],
            }),
            ref_spk_embedding: Some(TensorF32 {
                shape: vec![1024],
                values: vec![0.0; 1024],
            }),
        };
        let bytes = prompt.to_cbor_vec().unwrap();
        let parsed = VoiceClonePromptV2::from_cbor_slice(&bytes).unwrap();
        assert!(!parsed.x_vector_only_mode);
        assert!(parsed.icl_mode);
        assert_eq!(parsed.speaker_embedding_dim(), Some(1024));
        assert_eq!(parsed.ref_code_shape(), Some((105, 16)));
        assert!(!parsed.ref_text.is_empty());
        assert_eq!(
            parsed.ref_code_values().map(|values| values.len()),
            Some(105 * 16)
        );
    }
}
