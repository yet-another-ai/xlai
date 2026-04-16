use serde_json::json;
use xlai_core::{
    ErrorKind, StructuredOutput, StructuredOutputFormat, ToolCallExecutionMode, ToolDefinition,
    ToolParameter, ToolParameterType, XlaiError,
};

use crate::LlamaCppConfig;
use crate::prompt::{validate_structured_output, validate_structured_output_schema};
use crate::request::{PreparedRequest, PromptMessage, PromptRole, validate_prepared_for_llama};

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
            input_schema: None,
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
        execution: None,
        cancellation: None,
    };

    let result = validate_prepared_for_llama(&request, &config);
    assert!(matches!(
        result,
        Err(XlaiError {
            kind: ErrorKind::Unsupported,
            message,
            ..
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
            ..
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
            ..
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
            ..
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
