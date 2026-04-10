use std::collections::BTreeMap;

use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::json;
use xlai_core::{
    ChatContent, ChatMessage, ChatRequest, ContentPart, ErrorKind, MediaSource, MessageRole,
    ReasoningEffort, StructuredOutput, StructuredOutputFormat, ToolCall, ToolDefinition,
    ToolSchema, XlaiError,
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
        reasoning_effort: None,
        retry_policy: None,
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
    let content = &v["input"][0]["content"];
    assert!(content.is_array());
    assert_eq!(v["input"][0]["type"], json!("message"));
    assert_eq!(v["input"][0]["role"], json!("user"));
    assert_eq!(content[0]["type"], json!("input_text"));
    assert_eq!(content[1]["type"], json!("input_image"));
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
        reasoning_effort: None,
        retry_policy: None,
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
    assert_eq!(v["input"][0]["type"], json!("message"));
    assert_eq!(v["input"][0]["role"], json!("user"));
    assert_eq!(v["input"][0]["content"][0]["type"], json!("input_text"));
    assert_eq!(v["input"][0]["content"][0]["text"], json!("hello"));
}

#[test]
fn serializes_inline_audio_as_file_content_part() {
    let config = test_config();
    let decoded = STANDARD.decode("UklGRg==");
    assert!(decoded.is_ok(), "decode fixture base64");
    let Ok(audio_bytes) = decoded else {
        return;
    };
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::from_parts(vec![ContentPart::Audio {
                source: MediaSource::InlineData {
                    mime_type: "audio/wav".to_owned(),
                    data: audio_bytes,
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
        reasoning_effort: None,
        retry_policy: None,
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
    assert_eq!(v["input"][0]["content"][0]["type"], json!("input_file"));
}

#[test]
fn serializes_json_schema_structured_output_as_text_format() {
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
        reasoning_effort: None,
        retry_policy: None,
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
    assert_eq!(v["text"]["format"]["type"], json!("json_schema"));
    assert_eq!(v["text"]["format"]["name"], json!("person"));
    assert_eq!(v["text"]["format"]["schema"]["type"], json!("object"));
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
        reasoning_effort: None,
        retry_policy: None,
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
    assert_eq!(v["text"]["format"]["name"], json!("structured_output"));
    assert!(v["text"]["format"]["description"].is_null());
    assert_eq!(v["text"]["format"]["strict"], json!(true));
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
        reasoning_effort: None,
        retry_policy: None,
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

#[test]
fn serializes_assistant_tool_calls_for_follow_up_rounds() {
    let config = test_config();
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![
            ChatMessage {
                role: MessageRole::Assistant,
                content: ChatContent::empty(),
                tool_name: None,
                tool_call_id: None,
                metadata: BTreeMap::new(),
            }
            .with_assistant_tool_calls(&[ToolCall {
                id: "call_1".to_owned(),
                tool_name: "skill".to_owned(),
                arguments: json!({ "skill_id": "review.code" }),
            }]),
            ChatMessage {
                role: MessageRole::Tool,
                content: ChatContent::text("resolved"),
                tool_name: Some("skill".to_owned()),
                tool_call_id: Some("call_1".to_owned()),
                metadata: BTreeMap::new(),
            },
        ],
        available_tools: Vec::new(),
        structured_output: None,
        metadata: BTreeMap::new(),
        temperature: None,
        max_output_tokens: None,
        reasoning_effort: None,
        retry_policy: None,
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

    assert_eq!(v["input"][0]["type"], json!("function_call"));
    assert_eq!(v["input"][0]["call_id"], json!("call_1"));
    assert_eq!(v["input"][0]["name"], json!("skill"));
    assert_eq!(
        v["input"][0]["arguments"],
        json!(r#"{"skill_id":"review.code"}"#)
    );
    assert_eq!(v["input"][1]["type"], json!("function_call_output"));
    assert_eq!(v["input"][1]["call_id"], json!("call_1"));
    assert_eq!(v["input"][1]["output"], json!("resolved"));
}

#[test]
fn serializes_reasoning_effort_for_openai_requests() {
    let config = test_config();
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::text("Think carefully."),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }],
        available_tools: Vec::new(),
        structured_output: None,
        metadata: BTreeMap::new(),
        temperature: None,
        max_output_tokens: None,
        reasoning_effort: Some(ReasoningEffort::High),
        retry_policy: None,
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
    assert_eq!(v["reasoning"]["effort"], json!("high"));
}

#[test]
fn serializes_nested_tool_schema_for_openai_tools() {
    let config = test_config();
    let tool = ToolDefinition::new(
        "lookup_weather",
        "Lookup weather for multiple cities",
        ToolSchema::object(
            BTreeMap::from([
                (
                    "cities".to_owned(),
                    ToolSchema::array(Some(ToolSchema::object(
                        BTreeMap::from([(
                            "name".to_owned(),
                            ToolSchema::string().with_description("City name"),
                        )]),
                        vec!["name".to_owned()],
                    )))
                    .with_description("Cities to check"),
                ),
                (
                    "include_forecast".to_owned(),
                    ToolSchema::boolean().with_description("Whether to include forecast data"),
                ),
            ]),
            vec!["cities".to_owned()],
        ),
    );
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::text("Check the weather."),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }],
        available_tools: vec![tool],
        structured_output: None,
        metadata: BTreeMap::new(),
        temperature: None,
        max_output_tokens: None,
        reasoning_effort: None,
        retry_policy: None,
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
    assert_eq!(v["tools"][0]["parameters"]["type"], json!("object"));
    assert_eq!(
        v["tools"][0]["parameters"]["properties"]["cities"]["type"],
        json!("array")
    );
    assert_eq!(
        v["tools"][0]["parameters"]["properties"]["cities"]["items"]["properties"]["name"]["type"],
        json!("string")
    );
    assert_eq!(v["tools"][0]["parameters"]["required"], json!(["cities"]));
}
