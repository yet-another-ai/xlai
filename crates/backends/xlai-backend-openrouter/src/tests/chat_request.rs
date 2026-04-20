use std::collections::BTreeMap;

use serde_json::json;
use xlai_core::{
    ChatContent, ChatMessage, ChatRequest, MessageRole, ReasoningEffort, StructuredOutput,
};

use crate::request::OpenRouterChatRequest;

use super::common::test_config;

#[test]
fn serializes_plain_text_user_message_for_responses_api() {
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
        temperature: Some(0.3),
        max_output_tokens: Some(42),
        reasoning_effort: Some(ReasoningEffort::Medium),
        retry_policy: None,
        ..Default::default()
    };

    let payload = OpenRouterChatRequest::from_core_request(&config, request, false)
        .expect("payload should build");
    let value = serde_json::to_value(payload).expect("payload should serialize");

    assert_eq!(value["model"], json!("openai/gpt-4.1"));
    assert_eq!(value["input"][0]["type"], json!("message"));
    assert_eq!(value["input"][0]["role"], json!("user"));
    assert_eq!(value["input"][0]["content"][0]["type"], json!("input_text"));
    assert_eq!(value["input"][0]["content"][0]["text"], json!("hello"));
    let temperature = value["temperature"]
        .as_f64()
        .expect("temperature should serialize as a number");
    assert!((temperature - 0.3).abs() < 1e-6);
    assert_eq!(value["max_output_tokens"], json!(42));
    assert_eq!(value["reasoning"]["effort"], json!("medium"));
}

#[test]
fn serializes_json_schema_structured_output() {
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
            .with_name("person"),
        ),
        metadata: BTreeMap::new(),
        temperature: None,
        max_output_tokens: None,
        reasoning_effort: None,
        retry_policy: None,
        ..Default::default()
    };

    let payload = OpenRouterChatRequest::from_core_request(&config, request, true)
        .expect("payload should build");
    let value = serde_json::to_value(payload).expect("payload should serialize");

    assert_eq!(value["text"]["format"]["type"], json!("json_schema"));
    assert_eq!(value["text"]["format"]["name"], json!("person"));
    assert_eq!(value["stream"], json!(true));
}
