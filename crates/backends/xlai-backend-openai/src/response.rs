use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use xlai_core::{
    ChatContent, ChatMessage, ChatResponse, ContentPart, ErrorKind, FinishReason, MediaSource,
    MessageRole, TokenUsage, ToolCall, XlaiError,
};

pub(crate) const OPENAI_RESPONSE_OUTPUT_METADATA_KEY: &str = "openai_response_output";

#[derive(Deserialize)]
pub(crate) struct OpenAiChatResponse {
    #[serde(default)]
    output: Vec<Value>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    incomplete_details: Option<OpenAiIncompleteDetails>,
}

impl OpenAiChatResponse {
    pub(crate) fn into_core_response(self) -> Result<ChatResponse, XlaiError> {
        let (content, tool_calls) = openai_response_output_to_chat(&self.output)?;
        let message = attach_response_output_items(
            ChatMessage {
                role: MessageRole::Assistant,
                content,
                tool_name: None,
                tool_call_id: None,
                metadata: BTreeMap::new(),
            }
            .with_assistant_tool_calls(&tool_calls),
            &self.output,
        );

        let has_tool_calls = !tool_calls.is_empty();
        Ok(ChatResponse {
            message,
            tool_calls,
            usage: self.usage.map(Into::into),
            finish_reason: finish_reason_from_api(
                self.status.as_deref(),
                self.incomplete_details
                    .as_ref()
                    .and_then(|d| d.reason.as_deref()),
                has_tool_calls,
            ),
            metadata: BTreeMap::new(),
        })
    }
}

#[derive(Deserialize)]
pub(crate) struct OpenAiErrorEnvelope {
    pub(crate) error: OpenAiErrorDetail,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiErrorDetail {
    pub(crate) message: String,
    #[serde(rename = "type")]
    pub(crate) kind: Option<String>,
    #[serde(default)]
    pub(crate) param: Option<String>,
    #[serde(default)]
    pub(crate) code: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiIncompleteDetails {
    #[serde(default)]
    reason: Option<String>,
}

pub(crate) fn attach_response_output_items(
    mut message: ChatMessage,
    output: &[Value],
) -> ChatMessage {
    if !output.is_empty() {
        message.metadata.insert(
            OPENAI_RESPONSE_OUTPUT_METADATA_KEY.to_owned(),
            Value::Array(output.to_vec()),
        );
    }
    message
}

pub(crate) fn response_output_items_from_message(message: &ChatMessage) -> Option<Vec<Value>> {
    message
        .metadata
        .get(OPENAI_RESPONSE_OUTPUT_METADATA_KEY)
        .and_then(|value| {
            let Value::Array(items) = value else {
                return None;
            };
            Some(items.clone())
        })
}

pub(crate) fn openai_response_output_to_chat(
    output: &[Value],
) -> Result<(ChatContent, Vec<ToolCall>), XlaiError> {
    let mut parts: Vec<ContentPart> = Vec::new();
    let mut tool_calls = Vec::new();

    for item in output {
        let Some(obj) = item.as_object() else {
            continue;
        };
        match obj.get("type").and_then(Value::as_str) {
            Some("message") => {
                let content_items = obj
                    .get("content")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                parts.extend(
                    content_items
                        .iter()
                        .filter_map(parse_openai_response_content_part),
                );
            }
            Some("function_call") => {
                tool_calls.push(parse_openai_function_call_item(obj)?);
            }
            _ => {}
        }
    }

    let content = if parts.is_empty() {
        ChatContent::text("")
    } else {
        ChatContent::from_parts(parts)
    };
    Ok((content, tool_calls))
}

fn parse_openai_response_content_part(value: &Value) -> Option<ContentPart> {
    let obj = value.as_object()?;
    match obj.get("type")?.as_str()? {
        "output_text" | "input_text" => {
            let text = obj.get("text")?.as_str()?.to_owned();
            Some(ContentPart::Text { text })
        }
        "input_image" => {
            let url = obj.get("image_url")?.as_str()?.to_owned();
            Some(ContentPart::Image {
                source: MediaSource::Url { url },
                mime_type: None,
                detail: None,
            })
        }
        "refusal" => {
            let text = obj.get("refusal")?.as_str()?.to_owned();
            Some(ContentPart::Text { text })
        }
        _ => None,
    }
}

fn parse_openai_function_call_item(
    obj: &serde_json::Map<String, Value>,
) -> Result<ToolCall, XlaiError> {
    let arguments_raw = obj.get("arguments").and_then(Value::as_str).unwrap_or("{}");
    let arguments = serde_json::from_str(arguments_raw).map_err(|error| {
        XlaiError::new(
            ErrorKind::Provider,
            format!("failed to parse tool call arguments: {error}"),
        )
    })?;
    Ok(ToolCall {
        id: obj
            .get("call_id")
            .and_then(Value::as_str)
            .or_else(|| obj.get("id").and_then(Value::as_str))
            .unwrap_or("tool_call_0")
            .to_owned(),
        tool_name: obj
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        arguments,
    })
}

#[derive(Deserialize)]
struct OpenAiUsage {
    #[serde(rename = "prompt_tokens", alias = "input_tokens")]
    prompt: u32,
    #[serde(rename = "completion_tokens", alias = "output_tokens")]
    completion: u32,
    #[serde(rename = "total_tokens")]
    total: u32,
}

impl From<OpenAiUsage> for TokenUsage {
    fn from(value: OpenAiUsage) -> Self {
        Self {
            input_tokens: value.prompt,
            output_tokens: value.completion,
            total_tokens: value.total,
        }
    }
}

pub(crate) fn finish_reason_from_api(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
    has_tool_calls: bool,
) -> FinishReason {
    if has_tool_calls {
        return FinishReason::ToolCalls;
    }
    match (status, incomplete_reason) {
        (_, Some("max_output_tokens")) => FinishReason::Length,
        (Some("incomplete"), _) => FinishReason::Length,
        (Some("completed"), _) => FinishReason::Completed,
        _ => FinishReason::Completed,
    }
}
