use std::collections::BTreeMap;

use serde_json::json;
use xlai_core::{
    ChatContent, ChatMessage, ChatRequest, ContentPart, ErrorKind, MediaSource, MessageRole,
    StructuredOutput, StructuredOutputFormat, TranscriptionRequest, XlaiError,
};

use crate::OpenAiConfig;
use crate::request::OpenAiChatRequest;
use crate::transcription::{OpenAiTranscriptionRequest, OpenAiTranscriptionResponse};

fn test_config() -> OpenAiConfig {
    OpenAiConfig {
        base_url: "https://api.openai.com/v1".to_owned(),
        api_key: "k".to_owned(),
        model: "gpt-test".to_owned(),
        transcription_model: None,
    }
}

#[test]
fn serializes_multimodal_user_message_as_content_array() {
    let config = test_config();
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::from_parts(vec![
                ContentPart::Text {
                    text: "Describe the image.".to_owned(),
                },
                ContentPart::Image {
                    source: MediaSource::Url {
                        url: "https://example.com/a.png".to_owned(),
                    },
                    mime_type: None,
                    detail: None,
                },
            ]),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }],
        available_tools: Vec::new(),
        structured_output: None,
        metadata: BTreeMap::new(),
        temperature: None,
        max_output_tokens: None,
    };

    let payload = OpenAiChatRequest::from_core_request(&config, request, false);
    assert!(payload.is_ok(), "build payload");
    let Ok(payload) = payload else {
        return;
    };
    let serialized = serde_json::to_value(&payload);
    assert!(serialized.is_ok(), "serialize payload");
    let Ok(v) = serialized else {
        return;
    };
    let content = &v["messages"][0]["content"];
    assert!(content.is_array());
    assert_eq!(content[0]["type"], json!("text"));
    assert_eq!(content[1]["type"], json!("image_url"));
}

#[test]
fn serializes_plain_text_user_message_as_string_content() {
    let config = test_config();
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::text("hello"),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }],
        available_tools: Vec::new(),
        structured_output: None,
        metadata: BTreeMap::new(),
        temperature: None,
        max_output_tokens: None,
    };

    let payload = OpenAiChatRequest::from_core_request(&config, request, false);
    assert!(payload.is_ok(), "build payload");
    let Ok(payload) = payload else {
        return;
    };
    let serialized = serde_json::to_value(&payload);
    assert!(serialized.is_ok(), "serialize payload");
    let Ok(v) = serialized else {
        return;
    };
    assert_eq!(v["messages"][0]["content"], json!("hello"));
}

#[test]
fn serializes_inline_audio_as_file_content_part() {
    let config = test_config();
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::from_parts(vec![ContentPart::Audio {
                source: MediaSource::InlineData {
                    mime_type: "audio/wav".to_owned(),
                    data_base64: "UklGRg==".to_owned(),
                },
                mime_type: Some("audio/wav".to_owned()),
            }]),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }],
        available_tools: Vec::new(),
        structured_output: None,
        metadata: BTreeMap::new(),
        temperature: None,
        max_output_tokens: None,
    };

    let payload = OpenAiChatRequest::from_core_request(&config, request, false);
    assert!(payload.is_ok(), "build payload");
    let Ok(payload) = payload else {
        return;
    };
    let serialized = serde_json::to_value(&payload);
    assert!(serialized.is_ok(), "serialize payload");
    let Ok(v) = serialized else {
        return;
    };
    assert_eq!(v["messages"][0]["content"][0]["type"], json!("file"));
}

#[test]
fn serializes_json_schema_structured_output_as_response_format() {
    let config = test_config();
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::text("Return a person object."),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }],
        available_tools: Vec::new(),
        structured_output: Some(
            StructuredOutput::json_schema(json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                },
                "required": ["name"],
                "additionalProperties": false
            }))
            .with_name("person")
            .with_description("A simple person object"),
        ),
        metadata: BTreeMap::new(),
        temperature: None,
        max_output_tokens: None,
    };

    let payload = OpenAiChatRequest::from_core_request(&config, request, false);
    assert!(payload.is_ok(), "build payload");
    let Ok(payload) = payload else {
        return;
    };
    let serialized = serde_json::to_value(&payload);
    assert!(serialized.is_ok(), "serialize payload");
    let Ok(v) = serialized else {
        return;
    };
    assert_eq!(v["response_format"]["type"], json!("json_schema"));
    assert_eq!(v["response_format"]["json_schema"]["name"], json!("person"));
    assert_eq!(
        v["response_format"]["json_schema"]["schema"]["type"],
        json!("object")
    );
}

#[test]
fn json_schema_structured_output_defaults_response_format_name() {
    let config = test_config();
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::text("Return a person object."),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }],
        available_tools: Vec::new(),
        structured_output: Some(StructuredOutput::json_schema(json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"],
            "additionalProperties": false
        }))),
        metadata: BTreeMap::new(),
        temperature: None,
        max_output_tokens: None,
    };

    let payload = OpenAiChatRequest::from_core_request(&config, request, false);
    assert!(payload.is_ok(), "build payload");
    let Ok(payload) = payload else {
        return;
    };
    let serialized = serde_json::to_value(&payload);
    assert!(serialized.is_ok(), "serialize payload");
    let Ok(v) = serialized else {
        return;
    };
    assert_eq!(
        v["response_format"]["json_schema"]["name"],
        json!("structured_output")
    );
    assert!(v["response_format"]["json_schema"]["description"].is_null());
    assert_eq!(v["response_format"]["json_schema"]["strict"], json!(true));
}

#[test]
fn rejects_lark_structured_output_for_openai_compatible_backend() {
    let config = test_config();
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::text("Return a record."),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }],
        available_tools: Vec::new(),
        structured_output: Some(StructuredOutput {
            name: Some("record".to_owned()),
            description: None,
            format: StructuredOutputFormat::LarkGrammar {
                grammar: "start: NAME\nNAME: /[a-z]+/".to_owned(),
            },
        }),
        metadata: BTreeMap::new(),
        temperature: None,
        max_output_tokens: None,
    };

    let result = OpenAiChatRequest::from_core_request(&config, request, false);
    assert!(matches!(
        result,
        Err(XlaiError {
            kind: ErrorKind::Unsupported,
            message,
        }) if message.contains("Lark grammar")
    ));
}

#[test]
fn transcription_request_uses_configured_transcription_model_and_decodes_audio() {
    let config = test_config().with_transcription_model("gpt-4o-mini-transcribe");
    let request = TranscriptionRequest {
        model: None,
        audio: MediaSource::InlineData {
            mime_type: "audio/wav".to_owned(),
            data_base64: "UklGRg==".to_owned(),
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
    assert_eq!(payload.audio_bytes, b"RIFF".to_vec());
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

#[test]
fn provider_error_message_surfaces_openai_quota_details() {
    let message = crate::format_provider_error_message(
        reqwest::StatusCode::TOO_MANY_REQUESTS,
        Some("req_123"),
        r#"{
                "error": {
                    "message": "You exceeded your current quota.",
                    "type": "insufficient_quota",
                    "param": "model",
                    "code": "insufficient_quota"
                }
            }"#,
    );

    assert!(message.contains("429 Too Many Requests"));
    assert!(message.contains("request_id=req_123"));
    assert!(message.contains("You exceeded your current quota."));
    assert!(message.contains("type=insufficient_quota"));
    assert!(message.contains("code=insufficient_quota"));
    assert!(message.contains("param=model"));
}

#[test]
fn provider_error_message_falls_back_to_raw_body_for_non_json_errors() {
    let message = crate::format_provider_error_message(
        reqwest::StatusCode::BAD_GATEWAY,
        None,
        "upstream gateway timeout",
    );

    assert!(message.contains("502 Bad Gateway"));
    assert!(message.contains("upstream gateway timeout"));
}
