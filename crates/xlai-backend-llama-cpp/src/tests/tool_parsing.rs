use serde_json::json;
use xlai_core::{ToolCallExecutionMode, ToolDefinition, ToolParameter, ToolParameterType};

use xlai_local_common::{ToolResponse, parse_tool_response, tool_response_schema};

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
