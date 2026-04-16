use serde_json::json;
use xlai_core::{
    ChatContent, ChatMessage, ContentPart, ErrorKind, MessageRole, StructuredOutput,
    ToolCallExecutionMode, ToolDefinition, ToolParameter, ToolParameterType, XlaiError,
};

use crate::LlamaCppConfig;
use crate::prompt::{prompt_messages_with_constraints, render_manual_prompt};
use crate::request::{
    PreparedRequest, PromptMessage, PromptRole, extract_text_content, validate_prepared_for_llama,
};

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
            input_schema: None,
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
        execution: None,
        cancellation: None,
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
        execution: None,
        cancellation: None,
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
        execution: None,
        cancellation: None,
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
            input_schema: None,
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
        execution: None,
        cancellation: None,
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
