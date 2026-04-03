use serde_json::json;

use crate::{
    ChatContent, ChatMessage, ContentPart, ErrorKind, MediaSource, MessageRole, StructuredOutput,
    StructuredOutputFormat, XlaiError,
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
    let c = ChatContent::from_parts(vec![ContentPart::Audio {
        source: MediaSource::InlineData {
            mime_type: "audio/wav".to_owned(),
            data_base64: "UklGRg==".to_owned(),
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
