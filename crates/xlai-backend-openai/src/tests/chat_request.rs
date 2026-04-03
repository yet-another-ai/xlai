use std::collections::BTreeMap;

use serde_json::json;
use xlai_core::{
    ChatContent, ChatMessage, ChatRequest, ContentPart, ErrorKind, MediaSource, MessageRole,
    StructuredOutput, StructuredOutputFormat, XlaiError,
};

use crate::request::OpenAiChatRequest;

use super::common::test_config;

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
            ..
        }) if message.contains("Lark grammar")
    ));
}
