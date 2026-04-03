//! Map [`xlai_core::TtsRequest`] to [`xlai_qts_core::SynthesizeRequest`].

use serde_json::Value;
use xlai_core::{
    ErrorKind, MediaSource, Metadata, TtsRequest, VoiceReferenceSample, VoiceSpec, XlaiError,
};
use xlai_qts_core::{SynthesizeRequest, TalkerKvMode, VoiceCloneMode};

/// Parsed voice-clone inputs for QTS (first reference sample only in this release).
#[derive(Debug, Clone)]
pub struct QtsVoiceCloneParams {
    pub ref_wav: Vec<u8>,
    pub ref_text: Option<String>,
    pub mode: VoiceCloneMode,
}

/// Extract inline reference WAV + mode from [`VoiceSpec::Clone`].
///
/// # Errors
///
/// Returns [`XlaiError`] when references are missing, decoding fails, or mode metadata is invalid.
pub fn voice_clone_params_from_tts(
    request: &TtsRequest,
) -> Result<Option<QtsVoiceCloneParams>, XlaiError> {
    let VoiceSpec::Clone { references } = &request.voice else {
        return Ok(None);
    };
    if references.is_empty() {
        return Err(XlaiError::new(
            ErrorKind::Validation,
            "voice clone requires at least one VoiceReferenceSample",
        ));
    }
    let sample = &references[0];
    let ref_wav = voice_reference_wav_bytes(sample)?;
    let ref_text = sample.transcript.clone();
    let mode = resolve_voice_clone_mode(&request.metadata, ref_text.as_deref())?;
    Ok(Some(QtsVoiceCloneParams {
        ref_wav,
        ref_text,
        mode,
    }))
}

fn resolve_voice_clone_mode(
    meta: &Metadata,
    transcript: Option<&str>,
) -> Result<VoiceCloneMode, XlaiError> {
    if let Some(raw) = meta.get("xlai.qts.voice_clone_mode").and_then(json_str) {
        return VoiceCloneMode::parse(raw)
            .map_err(|e| XlaiError::new(ErrorKind::Validation, e.to_string()));
    }
    let has_text = transcript.map(|t| !t.trim().is_empty()).unwrap_or(false);
    Ok(if has_text {
        VoiceCloneMode::Icl
    } else {
        VoiceCloneMode::XVectorOnly
    })
}

fn voice_reference_wav_bytes(sample: &VoiceReferenceSample) -> Result<Vec<u8>, XlaiError> {
    if let Some(mt) = &sample.mime_type {
        let lower = mt.to_ascii_lowercase();
        if !(lower.contains("wav") || lower.contains("wave")) {
            return Err(XlaiError::new(
                ErrorKind::Validation,
                format!("voice clone reference audio must be WAV (got mime {mt})"),
            ));
        }
    }
    match &sample.audio {
        MediaSource::InlineData { data, .. } => Ok(data.clone()),
        MediaSource::Url { .. } => Err(XlaiError::new(
            ErrorKind::Unsupported,
            "voice clone with URL audio references is not supported by xlai-backend-qts",
        )),
    }
}

fn json_str(v: &Value) -> Option<&str> {
    v.as_str()
}

/// Build a QTS [`SynthesizeRequest`] from an xlai [`TtsRequest`].
///
/// # Errors
///
/// Returns [`XlaiError`] if required metadata is invalid.
pub fn synthesize_request_from_tts(request: &TtsRequest) -> Result<SynthesizeRequest, XlaiError> {
    let mut sr = SynthesizeRequest {
        text: request.input.clone(),
        ..SynthesizeRequest::default()
    };

    if let Some(n) = meta_u32(&request.metadata, "xlai.qts.thread_count") {
        sr.thread_count = n as usize;
    }
    if let Some(n) = meta_u32(&request.metadata, "xlai.qts.max_audio_frames") {
        sr.max_audio_frames = n as usize;
    }
    if let Some(t) = meta_f32(&request.metadata, "xlai.qts.temperature") {
        sr.temperature = t;
    }
    if let Some(t) = meta_f32(&request.metadata, "xlai.qts.top_p") {
        sr.top_p = t;
    }
    if let Some(k) = meta_i32(&request.metadata, "xlai.qts.top_k") {
        sr.top_k = k;
    }
    if let Some(r) = meta_f32(&request.metadata, "xlai.qts.repetition_penalty") {
        sr.repetition_penalty = r;
    }
    if let Some(id) = meta_i32(&request.metadata, "xlai.qts.language_id") {
        sr.language_id = id;
    }
    if let Some(n) = meta_u32(&request.metadata, "xlai.qts.vocoder_thread_count") {
        sr.vocoder_thread_count = n as usize;
    }
    if let Some(n) = meta_u32(&request.metadata, "xlai.qts.vocoder_chunk_size") {
        sr.vocoder_chunk_size = n as usize;
    }
    if let Some(s) = meta_str(&request.metadata, "xlai.qts.talker_kv_mode") {
        sr.talker_kv_mode = TalkerKvMode::parse(&s)
            .map_err(|e| XlaiError::new(ErrorKind::Validation, e.to_string()))?;
    }

    Ok(sr)
}

fn meta_u32(meta: &Metadata, key: &str) -> Option<u32> {
    meta.get(key).and_then(json_u32)
}

fn meta_i32(meta: &Metadata, key: &str) -> Option<i32> {
    meta.get(key).and_then(json_i32)
}

fn meta_f32(meta: &Metadata, key: &str) -> Option<f32> {
    meta.get(key).and_then(json_f32)
}

fn meta_str(meta: &Metadata, key: &str) -> Option<String> {
    meta.get(key).and_then(|v| v.as_str().map(str::to_owned))
}

fn json_u32(v: &Value) -> Option<u32> {
    if let Some(n) = v.as_u64() {
        return u32::try_from(n).ok();
    }
    v.as_str()?.parse().ok()
}

fn json_i32(v: &Value) -> Option<i32> {
    if let Some(n) = v.as_i64() {
        return i32::try_from(n).ok();
    }
    v.as_str()?.parse().ok()
}

fn json_f32(v: &Value) -> Option<f32> {
    if let Some(n) = v.as_f64() {
        return Some(n as f32);
    }
    v.as_str()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlai_core::VoiceSpec;

    #[test]
    fn clone_empty_references_errors() {
        let req = TtsRequest {
            model: None,
            input: "hi".to_owned(),
            voice: VoiceSpec::Clone { references: vec![] },
            response_format: None,
            speed: None,
            instructions: None,
            delivery: Default::default(),
            metadata: Metadata::default(),
        };
        assert!(matches!(
            voice_clone_params_from_tts(&req),
            Err(ref e) if e.kind == ErrorKind::Validation
        ));
    }

    #[test]
    fn synthesize_request_accepts_clone_voice() {
        let req = TtsRequest {
            model: None,
            input: "hi".to_owned(),
            voice: VoiceSpec::Clone { references: vec![] },
            response_format: None,
            speed: None,
            instructions: None,
            delivery: Default::default(),
            metadata: Metadata::default(),
        };
        assert!(synthesize_request_from_tts(&req).is_ok());
    }
}
