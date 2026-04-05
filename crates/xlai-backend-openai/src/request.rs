use base64::{Engine, engine::general_purpose::STANDARD};
use serde::Serialize;
use serde_json::{Map, Value};
use xlai_core::{
    ChatContent, ChatMessage, ChatRequest, ContentPart, ErrorKind, ImageDetail, MediaSource,
    MessageRole, StructuredOutputFormat, ToolDefinition, ToolParameterType, XlaiError,
};

use crate::OpenAiConfig;

#[derive(Serialize)]
pub(crate) struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiRequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OpenAiResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

impl OpenAiChatRequest {
    pub(crate) fn from_core_request(
        config: &OpenAiConfig,
        request: ChatRequest,
        stream: bool,
    ) -> Result<Self, XlaiError> {
        let response_format = match request
            .structured_output
            .as_ref()
            .map(|output| &output.format)
        {
            None => None,
            Some(StructuredOutputFormat::JsonSchema { schema }) => Some(OpenAiResponseFormat {
                kind: "json_schema",
                json_schema: Some(OpenAiJsonSchemaResponseFormat {
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
                    "openai-compatible backend does not support Lark grammar structured output",
                ));
            }
        };
        let tools = (!request.available_tools.is_empty()).then(|| {
            request
                .available_tools
                .iter()
                .map(OpenAiTool::from)
                .collect::<Vec<_>>()
        });

        Ok(Self {
            model: request.model.unwrap_or_else(|| config.model.clone()),
            messages: request
                .messages
                .into_iter()
                .map(OpenAiRequestMessage::from)
                .collect(),
            temperature: request.temperature,
            max_tokens: request.max_output_tokens,
            tool_choice: tools.as_ref().map(|_| "auto"),
            tools,
            response_format,
            stream: stream.then_some(true),
        })
    }
}

#[derive(Serialize)]
struct OpenAiResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<OpenAiJsonSchemaResponseFormat>,
}

#[derive(Serialize)]
struct OpenAiJsonSchemaResponseFormat {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Serialize)]
struct OpenAiRequestMessage {
    role: &'static str,
    content: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<ChatMessage> for OpenAiRequestMessage {
    fn from(message: ChatMessage) -> Self {
        let role = match message.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };

        Self {
            role,
            content: openai_request_content_value(&message),
            name: message.tool_name,
            tool_call_id: message.tool_call_id,
        }
    }
}

fn openai_request_content_value(message: &ChatMessage) -> Value {
    if message.role == MessageRole::Tool {
        return Value::String(message.content.text_parts_concatenated());
    }
    chat_content_to_openai_request_value(&message.content)
}

fn chat_content_to_openai_request_value(content: &ChatContent) -> Value {
    if let Some(text) = content.as_single_text() {
        return Value::String(text.to_owned());
    }
    let parts: Vec<Value> = content
        .parts
        .iter()
        .map(openai_request_part_value)
        .collect();
    Value::Array(parts)
}

fn openai_request_part_value(part: &ContentPart) -> Value {
    match part {
        ContentPart::Text { text } => serde_json::json!({
            "type": "text",
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
            let mut image_url = serde_json::json!({ "url": url });
            if let Some(d) = detail {
                image_url["detail"] = Value::String(image_detail_openai(*d).to_owned());
            }
            serde_json::json!({
                "type": "image_url",
                "image_url": image_url,
            })
        }
        ContentPart::Audio { source, mime_type } => match source {
            MediaSource::Url { url } => serde_json::json!({
                "type": "text",
                "text": format!("(Attached audio URL: {url})"),
            }),
            MediaSource::InlineData {
                mime_type: inline_mime,
                data,
            } => {
                let mime = mime_type.as_deref().unwrap_or(inline_mime.as_str());
                serde_json::json!({
                    "type": "file",
                    "file": {
                        "filename": "audio",
                        "file_data": format!("data:{mime};base64,{}", STANDARD.encode(data)),
                    },
                })
            }
        },
        ContentPart::File {
            source,
            mime_type,
            filename,
        } => match source {
            MediaSource::Url { url } => serde_json::json!({
                "type": "text",
                "text": format!("(Attached file URL: {url})"),
            }),
            MediaSource::InlineData {
                mime_type: inline_mime,
                data,
            } => {
                let mime = mime_type.as_deref().unwrap_or(inline_mime.as_str());
                let fname = filename.clone().unwrap_or_else(|| "attachment".to_owned());
                serde_json::json!({
                    "type": "file",
                    "file": {
                        "filename": fname,
                        "file_data": format!("data:{mime};base64,{}", STANDARD.encode(data)),
                    },
                })
            }
        },
    }
}

const fn image_detail_openai(detail: ImageDetail) -> &'static str {
    match detail {
        ImageDetail::Auto => "auto",
        ImageDetail::Low => "low",
        ImageDetail::High => "high",
    }
}

#[derive(Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunctionDefinition,
}

impl From<&ToolDefinition> for OpenAiTool {
    fn from(tool: &ToolDefinition) -> Self {
        Self {
            kind: "function",
            function: OpenAiFunctionDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool_json_schema(tool),
            },
        }
    }
}

#[derive(Serialize)]
struct OpenAiFunctionDefinition {
    name: String,
    description: String,
    parameters: Value,
}

fn tool_json_schema(tool: &ToolDefinition) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();

    for parameter in &tool.parameters {
        properties.insert(
            parameter.name.clone(),
            serde_json::json!({
                "type": tool_parameter_kind(parameter.kind),
                "description": parameter.description,
            }),
        );

        if parameter.required {
            required.push(parameter.name.clone());
        }
    }

    serde_json::json!({
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
