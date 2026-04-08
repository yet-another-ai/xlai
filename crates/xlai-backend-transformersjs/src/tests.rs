#![allow(clippy::expect_used, clippy::panic)]

use serde_json::json;
use xlai_core::{
    ChatMessage, ChatModel, ChatRequest, ErrorKind, MessageRole, StructuredOutput,
    ToolCallExecutionMode, ToolDefinition, ToolParameter, ToolParameterType,
};

use crate::TransformersJsChatModel;
use crate::TransformersJsConfig;
use xlai_local_common::{
    LocalChatPrepareOptions, PreparedLocalChatRequest, parse_tool_response, tool_response_schema,
    validate_structured_output,
};

#[test]
fn native_chat_model_generate_is_unsupported() {
    let model = TransformersJsChatModel::new(TransformersJsConfig::new("m"));
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let err = rt
        .block_on(model.generate(ChatRequest {
            model: None,
            system_prompt: None,
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: xlai_core::ChatContent::text("hi"),
                tool_name: None,
                tool_call_id: None,
                metadata: Default::default(),
            }],
            available_tools: Vec::new(),
            structured_output: None,
            metadata: Default::default(),
            temperature: None,
            max_output_tokens: None,
            reasoning_effort: None,
            retry_policy: None,
        }))
        .expect_err("wasm-only backend");

    assert_eq!(err.kind, ErrorKind::Unsupported);
}

#[test]
fn prepared_rejects_tools_plus_structured() {
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: xlai_core::ChatContent::text("x"),
            tool_name: None,
            tool_call_id: None,
            metadata: Default::default(),
        }],
        available_tools: vec![ToolDefinition {
            name: "t".into(),
            description: "d".into(),
            parameters: vec![],
            execution_mode: ToolCallExecutionMode::Concurrent,
        }],
        structured_output: Some(StructuredOutput::json_schema(json!({"type": "object"}))),
        metadata: Default::default(),
        temperature: None,
        max_output_tokens: None,
        reasoning_effort: None,
        retry_policy: None,
    };

    let opts = LocalChatPrepareOptions {
        default_temperature: 0.7,
        default_max_output_tokens: 32,
        expected_model_name: Some("hf/model".into()),
    };

    let prepared = PreparedLocalChatRequest::from_chat_request(request, &opts).expect("prepare");
    let err = prepared.validate_common().expect_err("tools+structured");
    assert_eq!(err.kind, ErrorKind::Unsupported);
}

#[test]
fn tool_envelope_roundtrip() {
    let tools = vec![ToolDefinition {
        name: "get_weather".into(),
        description: "w".into(),
        parameters: vec![ToolParameter {
            name: "city".into(),
            description: "c".into(),
            kind: ToolParameterType::String,
            required: true,
        }],
        execution_mode: ToolCallExecutionMode::Concurrent,
    }];
    let schema = tool_response_schema(&tools);
    let raw = json!({
        "assistant_response": null,
        "tool_calls": [{
            "name": "get_weather",
            "arguments": { "city": "Paris" }
        }]
    })
    .to_string();

    match parse_tool_response(&raw, &tools).expect("parse") {
        xlai_local_common::ToolResponse::ToolCalls(calls) => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].tool_name, "get_weather");
        }
        _ => panic!("expected tool calls"),
    }

    assert!(
        schema.get("properties").is_some(),
        "schema should look like JSON Schema"
    );
}

#[test]
fn structured_json_validates_after_generation() {
    let structured = StructuredOutput::json_schema(json!({
        "type": "object",
        "properties": { "n": { "type": "number" } },
        "required": ["n"]
    }));
    validate_structured_output(&structured, "{\"n\":1}").expect("valid");
    let err = validate_structured_output(&structured, "{\"n\":\"x\"}").expect_err("invalid");
    assert_eq!(err.kind, ErrorKind::Provider);
}
