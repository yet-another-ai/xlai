//! Map [`xlai_core::TtsRequest`] to [`xlai_qts_core::SynthesizeRequest`].

use serde_json::Value;
use xlai_core::{ErrorKind, Metadata, TtsRequest, VoiceSpec, XlaiError};
use xlai_qts_core::{SynthesizeRequest, TalkerKvMode};

/// Build a QTS [`SynthesizeRequest`] from an xlai [`TtsRequest`].
///
/// # Errors
///
/// Returns [`XlaiError`] if voice cloning is requested or required metadata is invalid.
pub fn synthesize_request_from_tts(request: &TtsRequest) -> Result<SynthesizeRequest, XlaiError> {
    if matches!(request.voice, VoiceSpec::Clone { .. }) {
        return Err(XlaiError::new(
            ErrorKind::Unsupported,
            "voice cloning is not supported by xlai-backend-qts in this release; see phase-2 Rust-native clone plan",
        ));
    }

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
    fn clone_voice_rejected() {
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
            synthesize_request_from_tts(&req),
            Err(ref e) if e.kind == ErrorKind::Unsupported
        ));
    }
}
