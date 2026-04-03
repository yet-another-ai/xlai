use std::collections::BTreeMap;

use serde_json::json;
use xlai_core::{
    ErrorKind, MediaSource, TtsAudioFormat, TtsDeliveryMode, TtsRequest, VoiceReferenceSample,
    VoiceSpec,
};

use crate::tts::{
    ParsedSpeechSse, build_speech_json_body, openai_model_supports_speech_sse,
    parse_speech_sse_data, resolved_tts_model,
};

use super::common::test_config;

fn sample_tts_request() -> TtsRequest {
    TtsRequest {
        model: None,
        input: "Hello.".to_owned(),
        voice: VoiceSpec::Preset {
            name: "alloy".to_owned(),
        },
        response_format: Some(TtsAudioFormat::Mp3),
        speed: Some(1.25),
        instructions: None,
        delivery: TtsDeliveryMode::Unary,
        metadata: BTreeMap::new(),
    }
}

#[test]
fn tts_json_body_uses_explicit_model_and_preset_voice() {
    let config = test_config().with_tts_model("tts-from-config");
    let mut request = sample_tts_request();
    request.model = Some("tts-explicit".to_owned());

    let body_result = build_speech_json_body(&config, &request, None);
    assert!(body_result.is_ok(), "build body: {:?}", body_result.err());
    let body = body_result.unwrap_or_else(|_| json!({}));
    assert_eq!(body["model"], json!("tts-explicit"));
    assert_eq!(body["voice"], json!("alloy"));
    assert_eq!(body["response_format"], json!("mp3"));
    assert_eq!(body["speed"], json!(1.25));
    assert!(body.get("stream_format").is_none());
}

#[test]
fn tts_json_body_falls_back_to_chat_model_when_no_tts_model() {
    let config = test_config();
    let request = sample_tts_request();
    let body_result = build_speech_json_body(&config, &request, None);
    assert!(body_result.is_ok(), "build body: {:?}", body_result.err());
    let body = body_result.unwrap_or_else(|_| json!({}));
    assert_eq!(body["model"], json!("gpt-test"));
}

#[test]
fn tts_json_body_includes_stream_format_sse() {
    let config = test_config();
    let request = sample_tts_request();
    let body_result = build_speech_json_body(&config, &request, Some("sse"));
    assert!(body_result.is_ok(), "build body: {:?}", body_result.err());
    let body = body_result.unwrap_or_else(|_| json!({}));
    assert_eq!(body["stream_format"], json!("sse"));
}

#[test]
fn tts_json_body_maps_provider_ref_voice_to_id_object() {
    let config = test_config();
    let request = TtsRequest {
        model: None,
        input: "Hi".to_owned(),
        voice: VoiceSpec::ProviderRef {
            id: "voice_abc".to_owned(),
            provider: None,
        },
        response_format: None,
        speed: None,
        instructions: None,
        delivery: TtsDeliveryMode::Unary,
        metadata: BTreeMap::new(),
    };
    let body_result = build_speech_json_body(&config, &request, None);
    assert!(body_result.is_ok(), "build body: {:?}", body_result.err());
    let body = body_result.unwrap_or_else(|_| json!({}));
    assert_eq!(body["voice"], json!({ "id": "voice_abc" }));
}

#[test]
fn tts_rejects_clone_voice_spec() {
    let config = test_config();
    let request = TtsRequest {
        model: None,
        input: "Hi".to_owned(),
        voice: VoiceSpec::Clone {
            references: vec![VoiceReferenceSample {
                audio: MediaSource::InlineData {
                    mime_type: "audio/wav".to_owned(),
                    data_base64: "UklGRg==".to_owned(),
                },
                mime_type: None,
                transcript: None,
                weight: None,
                metadata: BTreeMap::new(),
            }],
        },
        response_format: None,
        speed: None,
        instructions: None,
        delivery: TtsDeliveryMode::Unary,
        metadata: BTreeMap::new(),
    };
    let result = build_speech_json_body(&config, &request, None);
    assert!(result.is_err(), "clone voice should be rejected");
    let Err(err) = result else {
        return;
    };
    assert_eq!(err.kind, ErrorKind::Unsupported);
}

#[test]
fn tts_rejects_non_openai_voice_provider() {
    let config = test_config();
    let request = TtsRequest {
        model: None,
        input: "Hi".to_owned(),
        voice: VoiceSpec::ProviderRef {
            id: "v1".to_owned(),
            provider: Some("acme".to_owned()),
        },
        response_format: None,
        speed: None,
        instructions: None,
        delivery: TtsDeliveryMode::Unary,
        metadata: BTreeMap::new(),
    };
    let result = build_speech_json_body(&config, &request, None);
    assert!(result.is_err(), "non-openai provider should be rejected");
    let Err(err) = result else {
        return;
    };
    assert_eq!(err.kind, ErrorKind::Unsupported);
}

#[test]
fn resolved_tts_model_prefers_request_then_config_then_chat_model() {
    let config = test_config().with_tts_model("cfg-tts");
    let mut request = sample_tts_request();
    assert_eq!(resolved_tts_model(&config, &request), "cfg-tts");
    request.model = Some("req-tts".to_owned());
    assert_eq!(resolved_tts_model(&config, &request), "req-tts");
}

#[test]
fn openai_sse_model_support_flags_match_documentation() {
    assert!(!openai_model_supports_speech_sse("tts-1"));
    assert!(!openai_model_supports_speech_sse("tts-1-hd"));
    assert!(openai_model_supports_speech_sse("gpt-4o-mini-tts"));
}

#[test]
fn parse_speech_sse_accepts_audio_and_delta_fields() {
    let d1 = parse_speech_sse_data(r#"{"type":"speech.audio.delta","audio":"YWI="}"#);
    assert!(d1.is_ok(), "{:?}", d1.err());
    assert_eq!(
        d1.unwrap_or(ParsedSpeechSse::Ignored),
        ParsedSpeechSse::DeltaBase64("YWI=".to_owned())
    );
    let d2 = parse_speech_sse_data(r#"{"type":"speech.audio.delta","delta":"eGk="}"#);
    assert!(d2.is_ok(), "{:?}", d2.err());
    assert_eq!(
        d2.unwrap_or(ParsedSpeechSse::Ignored),
        ParsedSpeechSse::DeltaBase64("eGk=".to_owned())
    );
    let done = parse_speech_sse_data(r#"{"type":"speech.audio.done"}"#);
    assert!(done.is_ok(), "{:?}", done.err());
    assert_eq!(
        done.unwrap_or(ParsedSpeechSse::Ignored),
        ParsedSpeechSse::Done
    );
    let ign = parse_speech_sse_data(r#"{"type":"other"}"#);
    assert!(ign.is_ok(), "{:?}", ign.err());
    assert_eq!(
        ign.unwrap_or(ParsedSpeechSse::Done),
        ParsedSpeechSse::Ignored
    );
}
