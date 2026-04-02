#![allow(clippy::expect_used, clippy::panic)]

use serde_json::json;
use xlai_core::{
    ChatContent, ChatMessage, ContentPart, ErrorKind, MessageRole, StructuredOutput,
    StructuredOutputFormat, ToolCallExecutionMode, ToolDefinition, ToolParameter,
    ToolParameterType, XlaiError,
};

use crate::LlamaCppConfig;
use crate::prompt::{
    prompt_messages_with_constraints, render_manual_prompt, validate_structured_output,
    validate_structured_output_schema,
};
use crate::request::{
    PreparedRequest, PromptMessage, PromptRole, extract_text_content, validate_prepared_for_llama,
};
use xlai_local_common::{ToolResponse, parse_tool_response, tool_response_schema};

#[test]
fn manual_prompt_renderer_appends_assistant_turn() {
    let prompt = render_manual_prompt(&[
        PromptMessage {
            role: PromptRole::System,
            content: "Be concise.".to_owned(),
        },
        PromptMessage {
            role: PromptRole::User,
            content: "Say hello".to_owned(),
        },
    ])
    .expect("render manual prompt");

    assert!(prompt.contains("System: Be concise."));
    assert!(prompt.contains("User: Say hello"));
    assert!(prompt.ends_with("Assistant:"));
}

#[test]
fn prepared_request_allows_tool_calls() {
    let config = LlamaCppConfig::new("/tmp/model.gguf");
    let request = PreparedRequest {
        messages: vec![PromptMessage {
            role: PromptRole::User,
            content: "hi".to_owned(),
        }],
        available_tools: vec![ToolDefinition {
            name: "lookup_weather".to_owned(),
            description: "Lookup weather".to_owned(),
            parameters: vec![ToolParameter {
                name: "city".to_owned(),
                description: "City".to_owned(),
                kind: ToolParameterType::String,
                required: true,
            }],
            execution_mode: ToolCallExecutionMode::Concurrent,
        }],
        structured_output: None,
        temperature: 0.8,
        max_output_tokens: 64,
    };

    let result = validate_prepared_for_llama(&request, &config);
    assert!(result.is_ok(), "tool definitions should now be accepted");
}

#[test]
fn text_extraction_rejects_multimodal_content() {
    let message = ChatMessage {
        role: MessageRole::User,
        content: ChatContent::from_parts(vec![
            ContentPart::Text {
                text: "describe this".to_owned(),
            },
            ContentPart::Image {
                source: xlai_core::MediaSource::Url {
                    url: "https://example.com/image.png".to_owned(),
                },
                mime_type: None,
                detail: None,
            },
        ]),
        tool_name: None,
        tool_call_id: None,
        metadata: Default::default(),
    };

    let result = extract_text_content(&message);
    assert!(matches!(
        result,
        Err(XlaiError {
            kind: ErrorKind::Unsupported,
            ..
        })
    ));
}

#[test]
fn text_extraction_formats_tool_result_messages() {
    let message = ChatMessage {
        role: MessageRole::Tool,
        content: ChatContent::text("weather for Paris: sunny"),
        tool_name: Some("lookup_weather".to_owned()),
        tool_call_id: Some("tool_1".to_owned()),
        metadata: Default::default(),
    };

    let result = extract_text_content(&message);
    assert_eq!(
        result.as_deref(),
        Ok("Tool: lookup_weather\nCall ID: tool_1\nResult:\nweather for Paris: sunny")
    );
}

#[test]
fn structured_output_instruction_is_added_as_system_message() {
    let prepared = PreparedRequest {
        messages: vec![PromptMessage {
            role: PromptRole::User,
            content: "Return a person".to_owned(),
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
            .with_description("Simple person object"),
        ),
        temperature: 0.8,
        max_output_tokens: 64,
    };

    let messages = prompt_messages_with_constraints(&prepared);
    assert!(messages.is_ok(), "build prompt messages");
    let Ok(messages) = messages else {
        return;
    };

    assert_eq!(messages[0].role, PromptRole::System);
    assert!(messages[0].content.contains("Schema name: person"));
    assert!(messages[0].content.contains("\"required\": ["));
    assert_eq!(messages[1].role, PromptRole::User);
}

#[test]
fn lark_structured_output_instruction_is_added_as_system_message() {
    let prepared = PreparedRequest {
        messages: vec![PromptMessage {
            role: PromptRole::User,
            content: "Return a name".to_owned(),
        }],
        available_tools: Vec::new(),
        structured_output: Some(
            StructuredOutput::lark_grammar("start: NAME\nNAME: /[A-Z][a-z]+/")
                .with_name("person_name")
                .with_description("Single capitalized name"),
        ),
        temperature: 0.8,
        max_output_tokens: 64,
    };

    let messages = prompt_messages_with_constraints(&prepared);
    assert!(messages.is_ok(), "build prompt messages");
    let Ok(messages) = messages else {
        return;
    };

    assert_eq!(messages[0].role, PromptRole::System);
    assert!(messages[0].content.contains("Lark Grammar"));
    assert!(messages[0].content.contains("start: NAME"));
    assert!(!messages[0].content.contains("JSON Schema"));
    assert_eq!(messages[1].role, PromptRole::User);
}

#[test]
fn tool_instruction_is_added_as_system_message() {
    let prepared = PreparedRequest {
        messages: vec![PromptMessage {
            role: PromptRole::User,
            content: "What is the weather?".to_owned(),
        }],
        available_tools: vec![ToolDefinition {
            name: "lookup_weather".to_owned(),
            description: "Lookup weather by city".to_owned(),
            parameters: vec![ToolParameter {
                name: "city".to_owned(),
                description: "City name".to_owned(),
                kind: ToolParameterType::String,
                required: true,
            }],
            execution_mode: ToolCallExecutionMode::Concurrent,
        }],
        structured_output: None,
        temperature: 0.8,
        max_output_tokens: 64,
    };

    let messages = prompt_messages_with_constraints(&prepared);
    assert!(messages.is_ok(), "build prompt messages");
    let Ok(messages) = messages else {
        return;
    };

    assert_eq!(messages[0].role, PromptRole::System);
    assert!(
        messages[0]
            .content
            .contains("You may answer directly or request tool execution")
    );
    assert!(messages[0].content.contains("lookup_weather"));
    assert!(messages[0].content.contains("`assistant_response`"));
    assert_eq!(messages[1].role, PromptRole::User);
}

#[test]
fn prepared_request_rejects_combined_tools_and_structured_output() {
    let config = LlamaCppConfig::new("/tmp/model.gguf");
    let request = PreparedRequest {
        messages: vec![PromptMessage {
            role: PromptRole::User,
            content: "hi".to_owned(),
        }],
        available_tools: vec![ToolDefinition {
            name: "lookup_weather".to_owned(),
            description: "Lookup weather".to_owned(),
            parameters: vec![ToolParameter {
                name: "city".to_owned(),
                description: "City".to_owned(),
                kind: ToolParameterType::String,
                required: true,
            }],
            execution_mode: ToolCallExecutionMode::Concurrent,
        }],
        structured_output: Some(StructuredOutput::json_schema(json!({
            "type": "object"
        }))),
        temperature: 0.8,
        max_output_tokens: 64,
    };

    let result = validate_prepared_for_llama(&request, &config);
    assert!(matches!(
        result,
        Err(XlaiError {
            kind: ErrorKind::Unsupported,
            message,
        }) if message.contains("cannot be combined with tool calling")
    ));
}

#[test]
fn structured_output_validation_rejects_non_json_output() {
    let structured_output = StructuredOutput::json_schema(json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" }
        },
        "required": ["name"],
        "additionalProperties": false
    }));

    let result = validate_structured_output(&structured_output, "not json");
    assert!(matches!(
        result,
        Err(XlaiError {
            kind: ErrorKind::Provider,
            message,
        }) if message.contains("valid JSON")
    ));
}

#[test]
fn structured_output_validation_rejects_schema_mismatch() {
    let structured_output = StructuredOutput::json_schema(json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" }
        },
        "required": ["name"],
        "additionalProperties": false
    }));

    let result = validate_structured_output(&structured_output, r#"{"age": 42}"#);
    assert!(matches!(
        result,
        Err(XlaiError {
            kind: ErrorKind::Provider,
            message,
        }) if message.contains("did not match")
    ));
}

#[test]
fn lark_structured_output_validation_rejects_empty_grammar() {
    let structured_output = StructuredOutput {
        name: None,
        description: None,
        format: StructuredOutputFormat::LarkGrammar {
            grammar: "   ".to_owned(),
        },
    };

    let result = validate_structured_output_schema(&structured_output);
    assert!(matches!(
        result,
        Err(XlaiError {
            kind: ErrorKind::Validation,
            message,
        }) if message.contains("must not be empty")
    ));
}

#[test]
fn lark_structured_output_validation_accepts_non_json_output() {
    let structured_output = StructuredOutput::lark_grammar("start: NAME\nNAME: /[a-z]+/");
    let result = validate_structured_output(&structured_output, "alice");
    assert!(
        result.is_ok(),
        "lark output should not require JSON validation"
    );
}

#[test]
fn tool_response_schema_contains_tool_variants() {
    let schema = tool_response_schema(&[ToolDefinition {
        name: "lookup_weather".to_owned(),
        description: "Lookup weather".to_owned(),
        parameters: vec![ToolParameter {
            name: "city".to_owned(),
            description: "City".to_owned(),
            kind: ToolParameterType::String,
            required: true,
        }],
        execution_mode: ToolCallExecutionMode::Concurrent,
    }]);

    assert_eq!(
        schema["properties"]["tool_calls"]["items"]["oneOf"][0]["properties"]["name"]["const"],
        json!("lookup_weather")
    );
}

#[test]
fn tool_response_parser_returns_tool_calls() {
    let tools = vec![ToolDefinition {
        name: "lookup_weather".to_owned(),
        description: "Lookup weather".to_owned(),
        parameters: vec![ToolParameter {
            name: "city".to_owned(),
            description: "City".to_owned(),
            kind: ToolParameterType::String,
            required: true,
        }],
        execution_mode: ToolCallExecutionMode::Concurrent,
    }];

    let result = parse_tool_response(
        r#"{
            "assistant_response": null,
            "tool_calls": [
                {
                    "name": "lookup_weather",
                    "arguments": { "city": "Paris" }
                }
            ]
        }"#,
        &tools,
    );
    assert!(result.is_ok(), "parse tool response");
    let Ok(result) = result else {
        return;
    };

    assert!(matches!(result, ToolResponse::ToolCalls(_)));
    let ToolResponse::ToolCalls(calls) = result else {
        return;
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "local_tool_call_1");
    assert_eq!(calls[0].tool_name, "lookup_weather");
    assert_eq!(calls[0].arguments, json!({ "city": "Paris" }));
}

#[test]
fn tool_response_parser_returns_final_answer() {
    let result = parse_tool_response(
        r#"{
            "assistant_response": "Paris is sunny.",
            "tool_calls": []
        }"#,
        &[ToolDefinition {
            name: "lookup_weather".to_owned(),
            description: "Lookup weather".to_owned(),
            parameters: vec![ToolParameter {
                name: "city".to_owned(),
                description: "City".to_owned(),
                kind: ToolParameterType::String,
                required: true,
            }],
            execution_mode: ToolCallExecutionMode::Concurrent,
        }],
    );
    assert!(result.is_ok(), "parse final response");
    let Ok(result) = result else {
        return;
    };

    assert!(matches!(result, ToolResponse::AssistantMessage(_)));
    let ToolResponse::AssistantMessage(text) = result else {
        return;
    };
    assert_eq!(text, "Paris is sunny.");
}
