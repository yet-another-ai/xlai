use serde_json::json;
use xlai_core::{
    ChatContent, ChatMessage, ContentPart, ErrorKind, MessageRole, StructuredOutput,
    StructuredOutputFormat, ToolCallExecutionMode, ToolDefinition, ToolParameter,
    ToolParameterType, XlaiError,
};

use crate::LlamaCppConfig;
use crate::prompt::{
    prompt_messages_with_structured_output, render_manual_prompt, validate_structured_output,
    validate_structured_output_schema,
};
use crate::request::{PreparedRequest, PromptMessage, PromptRole, extract_text_content};

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
    ]);

    assert!(prompt.contains("System: Be concise."));
    assert!(prompt.contains("User: Say hello"));
    assert!(prompt.ends_with("Assistant:"));
}

#[test]
fn prepared_request_rejects_tool_calls_for_now() {
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

    let result = request.validate_against(&config);
    assert!(matches!(
        result,
        Err(XlaiError {
            kind: ErrorKind::Unsupported,
            message,
        }) if message.contains("tool calling")
    ));
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

    let messages = prompt_messages_with_structured_output(&prepared);
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

    let messages = prompt_messages_with_structured_output(&prepared);
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
