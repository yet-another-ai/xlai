use std::collections::BTreeMap;

use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::json;
use xlai_core::{MediaSource, TranscriptionRequest};

use crate::transcription::{OpenAiTranscriptionRequest, OpenAiTranscriptionResponse};

use super::common::test_config;

#[test]
fn transcription_request_uses_configured_transcription_model_and_decodes_audio() {
    let config = test_config().with_transcription_model("gpt-4o-mini-transcribe");
    let request = TranscriptionRequest {
        model: None,
        audio: MediaSource::InlineData {
            mime_type: "audio/wav".to_owned(),
            data: STANDARD.decode("UklGRg==").unwrap(),
        },
        mime_type: None,
        filename: Some("sample.wav".to_owned()),
        language: Some("en".to_owned()),
        prompt: Some("Speaker is concise.".to_owned()),
        temperature: Some(0.2),
        metadata: BTreeMap::new(),
    };

    let payload = OpenAiTranscriptionRequest::from_core_request(&config, request);
    assert!(payload.is_ok(), "build transcription request");
    let Ok(payload) = payload else {
        return;
    };

    assert_eq!(payload.model, "gpt-4o-mini-transcribe");
    assert_eq!(payload.filename, "sample.wav");
    assert_eq!(payload.mime_type, "audio/wav");
    assert_eq!(payload.audio_bytes, STANDARD.decode("UklGRg==").unwrap());
}

#[test]
fn transcription_request_rejects_url_audio_sources() {
    let config = test_config();
    let request = TranscriptionRequest {
        model: None,
        audio: MediaSource::Url {
            url: "https://example.com/audio.wav".to_owned(),
        },
        mime_type: Some("audio/wav".to_owned()),
        filename: None,
        language: None,
        prompt: None,
        temperature: None,
        metadata: BTreeMap::new(),
    };

    let result = OpenAiTranscriptionRequest::from_core_request(&config, request);
    assert!(result.is_err(), "url-based audio should be rejected");
    let Err(error) = result else {
        return;
    };
    assert_eq!(error.kind, xlai_core::ErrorKind::Unsupported);
    assert!(error.message.contains("inline audio"));
}

#[test]
fn transcription_response_preserves_provider_metadata() {
    let response: Result<OpenAiTranscriptionResponse, _> = serde_json::from_value(json!({
        "text": "hello world",
        "language": "en",
        "usage": {
            "total_tokens": 42
        }
    }));
    assert!(response.is_ok(), "deserialize transcription response");
    let Ok(response) = response else {
        return;
    };

    let response = response.into_core_response();
    assert_eq!(response.text, "hello world");
    assert_eq!(response.metadata.get("language"), Some(&json!("en")));
    assert_eq!(
        response.metadata.get("usage"),
        Some(&json!({
            "total_tokens": 42
        }))
    );
}
