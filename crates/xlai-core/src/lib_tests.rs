use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::json;

use crate::{
    ChatContent, ChatMessage, ChatRequest, ChatRetryPolicy, ContentPart, ErrorKind, MediaSource,
    MessageRole, ReasoningEffort, StructuredOutput, StructuredOutputFormat, ToolCall, TtsChunk,
    TtsResponse, XlaiError,
};

#[test]
fn chat_message_deserializes_missing_metadata_as_empty_map() {
    let result: Result<ChatMessage, _> = serde_json::from_value(json!({
        "role": "System",
        "content": "Preserve this reminder.",
        "tool_name": null,
        "tool_call_id": null
    }));
    assert!(
        result.is_ok(),
        "chat message without metadata should deserialize"
    );
    let Ok(message) = result else {
        return;
    };

    assert_eq!(message.role, MessageRole::System);
    assert!(message.metadata.is_empty());
}

#[test]
fn chat_message_supports_structured_metadata_values() {
    let message = ChatMessage {
        role: MessageRole::System,
        content: ChatContent::text("Preserve this reminder."),
        tool_name: None,
        tool_call_id: None,
        metadata: [(
            "reminder".to_owned(),
            json!({
                "kind": "system_reminder",
                "editable": true,
                "tags": ["session", "mutable"]
            }),
        )]
        .into_iter()
        .collect(),
    };

    assert_eq!(
        message.metadata.get("reminder"),
        Some(&json!({
            "kind": "system_reminder",
            "editable": true,
            "tags": ["session", "mutable"]
        }))
    );
}

#[test]
fn chat_message_round_trips_assistant_tool_calls_via_metadata() {
    let message = ChatMessage {
        role: MessageRole::Assistant,
        content: ChatContent::empty(),
        tool_name: None,
        tool_call_id: None,
        metadata: Default::default(),
    }
    .with_assistant_tool_calls(&[ToolCall {
        id: "call_1".to_owned(),
        tool_name: "skill".to_owned(),
        arguments: json!({ "skill_id": "review.code" }),
    }]);

    let tool_calls = message.assistant_tool_calls();
    assert!(
        tool_calls.is_some(),
        "expected assistant tool calls metadata"
    );
    let Some(tool_calls) = tool_calls else {
        return;
    };
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_1");
    assert_eq!(tool_calls[0].tool_name, "skill");
    assert_eq!(
        tool_calls[0].arguments,
        json!({ "skill_id": "review.code" })
    );
}

#[test]
fn chat_content_serializes_single_text_as_plain_string() {
    let c = ChatContent::text("hello");
    let result = serde_json::to_value(&c);
    assert!(result.is_ok(), "serialize");
    let Ok(v) = result else {
        return;
    };
    assert_eq!(v, json!("hello"));
}

#[test]
fn chat_content_round_trips_multimodal_parts() {
    let c = ChatContent::from_parts(vec![
        ContentPart::Text {
            text: "Describe:".to_owned(),
        },
        ContentPart::Image {
            source: MediaSource::Url {
                url: "https://example.com/a.png".to_owned(),
            },
            mime_type: Some("image/png".to_owned()),
            detail: None,
        },
    ]);
    let serialized = serde_json::to_value(&c);
    assert!(serialized.is_ok(), "serialize");
    let Ok(v) = serialized else {
        return;
    };
    let deserialized: Result<ChatContent, _> = serde_json::from_value(v);
    assert!(deserialized.is_ok(), "deserialize");
    let Ok(back) = deserialized else {
        return;
    };
    assert_eq!(back, c);
}

#[test]
fn chat_content_round_trips_audio_parts() {
    let decoded = STANDARD.decode("UklGRg==");
    assert!(decoded.is_ok(), "decode fixture base64");
    let Ok(audio_bytes) = decoded else {
        return;
    };
    let c = ChatContent::from_parts(vec![ContentPart::Audio {
        source: MediaSource::InlineData {
            mime_type: "audio/wav".to_owned(),
            data: audio_bytes,
        },
        mime_type: Some("audio/wav".to_owned()),
    }]);
    let serialized = serde_json::to_value(&c);
    assert!(serialized.is_ok(), "serialize");
    let Ok(v) = serialized else {
        return;
    };
    let deserialized: Result<ChatContent, _> = serde_json::from_value(v);
    assert!(deserialized.is_ok(), "deserialize");
    let Ok(back) = deserialized else {
        return;
    };
    assert_eq!(back, c);
}

#[test]
fn structured_output_round_trips_json_schema_format() {
    let output = StructuredOutput::json_schema(json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" }
        },
        "required": ["name"],
        "additionalProperties": false
    }))
    .with_name("person")
    .with_description("A person object");

    let serialized = serde_json::to_value(&output);
    assert!(serialized.is_ok(), "serialize");
    let Ok(v) = serialized else {
        return;
    };
    assert_eq!(v["type"], json!("json_schema"));
    assert_eq!(v["schema"]["type"], json!("object"));

    let deserialized: Result<StructuredOutput, _> = serde_json::from_value(v);
    assert!(deserialized.is_ok(), "deserialize");
    let Ok(back) = deserialized else {
        return;
    };
    assert_eq!(back, output);
}

#[test]
fn structured_output_round_trips_lark_grammar_format() {
    let output = StructuredOutput::lark_grammar("start: NAME\nNAME: /[a-z]+/")
        .with_name("record")
        .with_description("A simple lark grammar");

    let serialized = serde_json::to_value(&output);
    assert!(serialized.is_ok(), "serialize");
    let Ok(v) = serialized else {
        return;
    };
    assert_eq!(v["type"], json!("lark_grammar"));
    assert_eq!(v["grammar"], json!("start: NAME\nNAME: /[a-z]+/"));

    let deserialized: Result<StructuredOutput, _> = serde_json::from_value(v);
    assert!(deserialized.is_ok(), "deserialize");
    let Ok(back) = deserialized else {
        return;
    };
    assert_eq!(back, output);
    assert!(matches!(
        back.format,
        StructuredOutputFormat::LarkGrammar { .. }
    ));
}

#[test]
fn reasoning_effort_round_trips_snake_case_json() {
    let serialized = serde_json::to_value(ReasoningEffort::Medium);
    assert!(serialized.is_ok(), "serialize");
    let Ok(v) = serialized else {
        return;
    };
    assert_eq!(v, json!("medium"));

    let deserialized: Result<ReasoningEffort, _> = serde_json::from_value(v);
    assert!(deserialized.is_ok(), "deserialize");
    let Ok(back) = deserialized else {
        return;
    };
    assert_eq!(back, ReasoningEffort::Medium);
}

#[test]
fn xlai_error_deserializes_legacy_json_without_optional_fields() {
    let v = json!({"kind": "Provider", "message": "bad"});
    let result: Result<XlaiError, _> = serde_json::from_value(v);
    assert!(result.is_ok(), "deserialize legacy error JSON");
    let Ok(e) = result else {
        return;
    };
    assert_eq!(e.kind, ErrorKind::Provider);
    assert_eq!(e.message, "bad");
    assert!(e.http_status.is_none());
    assert!(e.request_id.is_none());
}

#[test]
fn xlai_error_round_trips_optional_fields() {
    let e = XlaiError::new(ErrorKind::Provider, "x")
        .with_http_status(429)
        .with_request_id("r1")
        .with_provider_code("rate_limit")
        .with_retryable(true);
    let serialized = serde_json::to_value(&e);
    assert!(serialized.is_ok(), "serialize");
    let Ok(v) = serialized else {
        return;
    };
    let deserialized: Result<XlaiError, _> = serde_json::from_value(v);
    assert!(deserialized.is_ok(), "deserialize");
    let Ok(back) = deserialized else {
        return;
    };
    assert_eq!(back, e);
}

#[test]
fn inline_media_json_serializes_data_as_base64_string() {
    let src = MediaSource::InlineData {
        mime_type: "audio/wav".to_owned(),
        data: vec![0, 1, 2, 255],
    };
    let serialized = serde_json::to_value(&src);
    assert!(serialized.is_ok(), "serialize");
    let Ok(v) = serialized else {
        return;
    };
    let obj = v.as_object();
    assert!(
        obj.is_some(),
        "expected serialized media source to be a JSON object"
    );
    let Some(obj) = obj else {
        return;
    };
    assert!(obj.contains_key("data"), "JSON should use `data` key");
    assert_eq!(obj["data"], json!(STANDARD.encode([0u8, 1, 2, 255])));
    let deserialized: Result<MediaSource, _> = serde_json::from_value(v);
    assert!(deserialized.is_ok(), "deserialize");
    let Ok(back) = deserialized else {
        return;
    };
    assert_eq!(back, src);
}

#[test]
fn tts_response_cbor_roundtrip_smaller_than_json_for_binary() {
    let pcm: Vec<u8> = (0u8..=200).collect();
    let response = TtsResponse {
        audio: MediaSource::InlineData {
            mime_type: "audio/wav".to_owned(),
            data: pcm.clone(),
        },
        mime_type: "audio/wav".to_owned(),
        metadata: Default::default(),
    };
    let cbor = response.to_cbor_vec();
    assert!(cbor.is_ok(), "cbor encode");
    let Ok(cbor) = cbor else {
        return;
    };
    let json = serde_json::to_vec(&response);
    assert!(json.is_ok(), "json encode");
    let Ok(json) = json else {
        return;
    };
    assert!(
        cbor.len() < json.len(),
        "CBOR should be smaller than JSON base64 for this payload: cbor={} json={}",
        cbor.len(),
        json.len()
    );
    let decoded = TtsResponse::from_cbor_slice(&cbor);
    assert!(decoded.is_ok(), "cbor decode");
    let Ok(back) = decoded else {
        return;
    };
    assert_eq!(back, response);
}

#[test]
fn tts_chunk_cbor_roundtrip() {
    let chunk = TtsChunk::AudioDelta {
        data: vec![1, 2, 3],
    };
    let encoded = chunk.to_cbor_vec();
    assert!(encoded.is_ok(), "encode chunk as cbor");
    let Ok(bytes) = encoded else {
        return;
    };
    let decoded = TtsChunk::from_cbor_slice(&bytes);
    assert!(decoded.is_ok(), "decode chunk from cbor");
    let Ok(back) = decoded else {
        return;
    };
    assert_eq!(back, chunk);
}

#[test]
fn chat_request_deserializes_without_retry_policy() {
    let v = json!({
        "model": null,
        "system_prompt": null,
        "messages": [],
        "available_tools": [],
        "metadata": {},
        "temperature": null,
        "max_output_tokens": null
    });
    let parsed = serde_json::from_value::<ChatRequest>(v);
    assert!(
        parsed.is_ok(),
        "deserialize ChatRequest without retry_policy"
    );
    let Ok(req) = parsed else {
        return;
    };
    assert!(req.retry_policy.is_none());
}

#[test]
fn chat_request_round_trips_retry_policy() {
    let req = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![],
        available_tools: vec![],
        structured_output: None,
        metadata: Default::default(),
        temperature: None,
        max_output_tokens: None,
        reasoning_effort: None,
        retry_policy: Some(
            ChatRetryPolicy::default()
                .with_max_retries(1)
                .with_initial_backoff_ms(100),
        ),
    };
    let serialized = serde_json::to_value(&req);
    assert!(
        serialized.is_ok(),
        "serialize ChatRequest with retry_policy"
    );
    let Ok(v) = serialized else {
        return;
    };
    let parsed = serde_json::from_value::<ChatRequest>(v);
    assert!(parsed.is_ok(), "deserialize ChatRequest with retry_policy");
    let Ok(back) = parsed else {
        return;
    };
    assert_eq!(back, req);
}
