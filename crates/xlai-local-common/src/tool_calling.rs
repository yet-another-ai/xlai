use serde_json::{Map, Value, json};
use tera::Context;
use xlai_core::{ErrorKind, ToolCall, ToolDefinition, ToolParameterType, XlaiError};

use crate::prompt_store::EmbeddedPromptStore;

pub enum ToolResponse {
    AssistantMessage(String),
    ToolCalls(Vec<ToolCall>),
}

pub fn tool_call_instruction(tools: &[ToolDefinition]) -> Result<String, XlaiError> {
    let tool_specs = tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool_arguments_schema(tool),
            })
        })
        .collect::<Vec<_>>();
    let tool_specs = serde_json::to_string_pretty(&tool_specs).map_err(|error| {
        XlaiError::new(
            ErrorKind::Validation,
            format!("tool definitions could not be serialized: {error}"),
        )
    })?;

    let mut context = Context::new();
    context.insert("tool_specs", &tool_specs);
    EmbeddedPromptStore::render("system/tool-calling.md", &context)
}

pub fn tool_response_schema(tools: &[ToolDefinition]) -> Value {
    let variants = tools
        .iter()
        .map(|tool| {
            json!({
                "type": "object",
                "properties": {
                    "name": {
                        "const": tool.name,
                    },
                    "arguments": tool_arguments_schema(tool),
                },
                "required": ["name", "arguments"],
                "additionalProperties": false,
            })
        })
        .collect::<Vec<_>>();

    json!({
        "type": "object",
        "properties": {
            "assistant_response": {
                "type": ["string", "null"],
            },
            "tool_calls": {
                "type": "array",
                "items": {
                    "oneOf": variants,
                },
            },
        },
        "required": ["assistant_response", "tool_calls"],
        "additionalProperties": false,
    })
}

pub fn parse_tool_response(
    generated: &str,
    tools: &[ToolDefinition],
) -> Result<ToolResponse, XlaiError> {
    let generated = generated.trim();
    let value: Value = serde_json::from_str(generated).map_err(|error| {
        XlaiError::new(
            ErrorKind::Provider,
            format!("tool-calling output was not valid JSON: {error}"),
        )
    })?;

    let schema = tool_response_schema(tools);
    let validator = jsonschema::validator_for(&schema).map_err(|error| {
        XlaiError::new(
            ErrorKind::Validation,
            format!("tool-calling schema is invalid: {error}"),
        )
    })?;
    if let Err(error) = validator.validate(&value) {
        return Err(XlaiError::new(
            ErrorKind::Provider,
            format!("tool-calling output did not match the required schema: {error}"),
        ));
    }

    let envelope = value.as_object().ok_or_else(|| {
        XlaiError::new(
            ErrorKind::Provider,
            "tool-calling output must be a JSON object",
        )
    })?;
    let assistant_response = envelope
        .get("assistant_response")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let tool_calls = envelope
        .get("tool_calls")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            XlaiError::new(
                ErrorKind::Provider,
                "tool-calling output must include a `tool_calls` array",
            )
        })?;

    if tool_calls.is_empty() {
        let assistant_response = assistant_response.unwrap_or_default();
        if assistant_response.trim().is_empty() {
            return Err(XlaiError::new(
                ErrorKind::Provider,
                "tool-calling output contained neither a final assistant response nor any tool calls",
            ));
        }
        return Ok(ToolResponse::AssistantMessage(assistant_response));
    }

    if assistant_response
        .as_deref()
        .is_some_and(|text| !text.trim().is_empty())
    {
        return Err(XlaiError::new(
            ErrorKind::Provider,
            "tool-calling output may not include both assistant text and tool calls in the same turn",
        ));
    }

    Ok(ToolResponse::ToolCalls(
        tool_calls
            .iter()
            .enumerate()
            .map(|(index, call)| ToolCall {
                id: format!("local_tool_call_{}", index + 1),
                tool_name: call
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                arguments: call.get("arguments").cloned().unwrap_or(Value::Null),
            })
            .collect(),
    ))
}

fn tool_arguments_schema(tool: &ToolDefinition) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();

    for parameter in &tool.parameters {
        properties.insert(
            parameter.name.clone(),
            json!({
                "type": tool_parameter_kind(parameter.kind),
                "description": parameter.description,
            }),
        );
        if parameter.required {
            required.push(parameter.name.clone());
        }
    }

    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

const fn tool_parameter_kind(kind: ToolParameterType) -> &'static str {
    match kind {
        ToolParameterType::String => "string",
        ToolParameterType::Number => "number",
        ToolParameterType::Integer => "integer",
        ToolParameterType::Boolean => "boolean",
        ToolParameterType::Array => "array",
        ToolParameterType::Object => "object",
    }
}
