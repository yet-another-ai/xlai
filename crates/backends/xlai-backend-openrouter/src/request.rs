use base64::{Engine, engine::general_purpose::STANDARD};
use serde::Serialize;
use serde_json::Value;
use xlai_core::{
    ChatMessage, ChatRequest, ContentPart, ErrorKind, ImageDetail, MediaSource, MessageRole,
    StructuredOutputFormat, ToolCall, ToolDefinition, XlaiError,
};

use crate::OpenRouterConfig;
use crate::response::response_output_items_from_message;

#[derive(Serialize)]
pub(crate) struct OpenRouterChatRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    input: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<OpenRouterReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenRouterTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<OpenRouterTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

impl OpenRouterChatRequest {
    pub(crate) fn from_core_request(
        config: &OpenRouterConfig,
        request: ChatRequest,
        stream: bool,
    ) -> Result<Self, XlaiError> {
        let text = match request
            .structured_output
            .as_ref()
            .map(|output| &output.format)
        {
            None => None,
            Some(StructuredOutputFormat::JsonSchema { schema }) => Some(OpenRouterTextConfig {
                format: Some(OpenRouterTextFormat::JsonSchema {
                    name: request
                        .structured_output
                        .as_ref()
                        .and_then(|output| output.name.clone())
                        .unwrap_or_else(|| "structured_output".to_owned()),
                    description: request
                        .structured_output
                        .as_ref()
                        .and_then(|output| output.description.clone()),
                    schema: schema.clone(),
                    strict: Some(true),
                }),
            }),
            Some(StructuredOutputFormat::LarkGrammar { .. }) => {
                return Err(XlaiError::new(
                    ErrorKind::Unsupported,
                    "openrouter backend does not support Lark grammar structured output",
                ));
            }
        };
        let tools = (!request.available_tools.is_empty()).then(|| {
            request
                .available_tools
                .iter()
                .map(OpenRouterTool::from)
                .collect::<Vec<_>>()
        });

        Ok(Self {
            model: request.model.unwrap_or_else(|| config.model.clone()),
            instructions: request.system_prompt,
            input: request
                .messages
                .into_iter()
                .flat_map(openrouter_request_input_items)
                .collect(),
            temperature: request.temperature,
            max_output_tokens: request.max_output_tokens,
            reasoning: request
                .reasoning_effort
                .map(|effort| OpenRouterReasoningConfig {
                    effort: reasoning_effort_openrouter(effort),
                }),
            tool_choice: tools.as_ref().map(|_| "auto"),
            tools,
            text,
            stream: stream.then_some(true),
        })
    }
}

fn openrouter_request_input_items(message: ChatMessage) -> Vec<Value> {
    if let Some(items) = response_output_items_from_message(&message) {
        return items;
    }

    match message.role {
        MessageRole::System => vec![openrouter_request_message_item(
            "system",
            &message.content,
            false,
        )],
        MessageRole::User => vec![openrouter_request_message_item(
            "user",
            &message.content,
            false,
        )],
        MessageRole::Assistant => {
            if let Some(tool_calls) = message.assistant_tool_calls()
                && !tool_calls.is_empty()
            {
                return openrouter_request_function_calls(&tool_calls);
            }
            vec![openrouter_request_message_item(
                "assistant",
                &message.content,
                true,
            )]
        }
        MessageRole::Tool => vec![serde_json::json!({
            "type": "function_call_output",
            "call_id": message.tool_call_id,
            "output": message.content.text_parts_concatenated(),
        })],
    }
}

fn openrouter_request_message_item(
    role: &str,
    content: &xlai_core::ChatContent,
    assistant_output: bool,
) -> Value {
    serde_json::json!({
        "type": "message",
        "role": role,
        "content": content
            .parts
            .iter()
            .map(|part| openrouter_request_part_value(part, assistant_output))
            .collect::<Vec<_>>(),
    })
}

fn openrouter_request_function_calls(tool_calls: &[ToolCall]) -> Vec<Value> {
    tool_calls
        .iter()
        .map(|tool_call| {
            serde_json::json!({
                "type": "function_call",
                "call_id": tool_call.id,
                "name": tool_call.tool_name,
                "arguments": tool_call.arguments.to_string(),
            })
        })
        .collect()
}

fn openrouter_request_part_value(part: &ContentPart, assistant_output: bool) -> Value {
    match part {
        ContentPart::Text { text } => serde_json::json!({
            "type": if assistant_output { "output_text" } else { "input_text" },
            "text": text,
        }),
        ContentPart::Image {
            source,
            mime_type,
            detail,
        } => {
            let url = match source {
                MediaSource::Url { url } => url.clone(),
                MediaSource::InlineData {
                    mime_type: inline_mime,
                    data,
                } => {
                    let mime = mime_type.as_deref().unwrap_or(inline_mime.as_str());
                    format!("data:{mime};base64,{}", STANDARD.encode(data))
                }
            };
            let mut item = serde_json::json!({
                "type": "input_image",
                "image_url": url,
            });
            if let Some(detail) = detail {
                item["detail"] = Value::String(image_detail_openrouter(*detail).to_owned());
            }
            item
        }
        ContentPart::Audio { source, mime_type } => match source {
            MediaSource::Url { url } => serde_json::json!({
                "type": if assistant_output { "output_text" } else { "input_text" },
                "text": format!("(Attached audio URL: {url})"),
            }),
            MediaSource::InlineData {
                mime_type: inline_mime,
                data,
            } => {
                let mime = mime_type.as_deref().unwrap_or(inline_mime.as_str());
                serde_json::json!({
                    "type": "input_file",
                    "filename": "audio",
                    "file_data": format!("data:{mime};base64,{}", STANDARD.encode(data)),
                })
            }
        },
        ContentPart::File {
            source,
            mime_type,
            filename,
        } => match source {
            MediaSource::Url { url } => serde_json::json!({
                "type": "input_file",
                "file_url": url,
                "filename": filename.clone().unwrap_or_else(|| "attachment".to_owned()),
            }),
            MediaSource::InlineData {
                mime_type: inline_mime,
                data,
            } => {
                let mime = mime_type.as_deref().unwrap_or(inline_mime.as_str());
                let filename = filename.clone().unwrap_or_else(|| "attachment".to_owned());
                serde_json::json!({
                    "type": "input_file",
                    "filename": filename,
                    "file_data": format!("data:{mime};base64,{}", STANDARD.encode(data)),
                })
            }
        },
    }
}

const fn image_detail_openrouter(detail: ImageDetail) -> &'static str {
    match detail {
        ImageDetail::Auto => "auto",
        ImageDetail::Low => "low",
        ImageDetail::High => "high",
    }
}

#[derive(Serialize)]
struct OpenRouterReasoningConfig {
    effort: &'static str,
}

#[derive(Serialize)]
struct OpenRouterTextConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<OpenRouterTextFormat>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenRouterTextFormat {
    JsonSchema {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        schema: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

#[derive(Serialize)]
struct OpenRouterTool {
    #[serde(rename = "type")]
    kind: &'static str,
    name: String,
    description: String,
    parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

impl From<&ToolDefinition> for OpenRouterTool {
    fn from(tool: &ToolDefinition) -> Self {
        Self {
            kind: "function",
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool_json_schema(tool),
            strict: None,
        }
    }
}

fn tool_json_schema(tool: &ToolDefinition) -> Value {
    tool.resolved_input_schema().json_schema()
}

const fn reasoning_effort_openrouter(effort: xlai_core::ReasoningEffort) -> &'static str {
    match effort {
        xlai_core::ReasoningEffort::Low => "low",
        xlai_core::ReasoningEffort::Medium => "medium",
        xlai_core::ReasoningEffort::High => "high",
    }
}
