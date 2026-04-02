use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use xlai_core::{
    ChatContent, ChatMessage, ChatResponse, ContentPart, ErrorKind, FinishReason, MediaSource,
    MessageRole, TokenUsage, ToolCall, XlaiError,
};

#[derive(Deserialize)]
pub(crate) struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

impl OpenAiChatResponse {
    pub(crate) fn into_core_response(self) -> Result<ChatResponse, XlaiError> {
        let choice = self.choices.into_iter().next().ok_or_else(|| {
            XlaiError::new(
                ErrorKind::Provider,
                "openai-compatible response contained no choices",
            )
        })?;

        let message = ChatMessage {
            role: MessageRole::Assistant,
            content: openai_response_content_to_chat_content(choice.message.content.as_ref()),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        };

        let tool_calls = choice
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(ToolCall::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ChatResponse {
            message,
            tool_calls,
            usage: self.usage.map(Into::into),
            finish_reason: finish_reason_from_api(choice.finish_reason.as_deref()),
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
struct OpenAiChoice {
    message: OpenAiResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiResponseMessage {
    #[serde(default)]
    content: Option<Value>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

fn openai_response_content_to_chat_content(value: Option<&Value>) -> ChatContent {
    match value {
        None | Some(Value::Null) => ChatContent::text(""),
        Some(Value::String(s)) => ChatContent::text(s.clone()),
        Some(Value::Array(items)) => {
            let parts: Vec<ContentPart> = items
                .iter()
                .filter_map(parse_openai_response_content_part)
                .collect();
            if parts.is_empty() {
                ChatContent::text("")
            } else {
                ChatContent::from_parts(parts)
            }
        }
        Some(other) => ChatContent::text(other.to_string()),
    }
}

fn parse_openai_response_content_part(value: &Value) -> Option<ContentPart> {
    let obj = value.as_object()?;
    let ty = obj.get("type")?.as_str()?;
    match ty {
        "text" => {
            let text = obj.get("text")?.as_str()?.to_owned();
            Some(ContentPart::Text { text })
        }
        "image_url" => {
            let url = obj.get("image_url")?.get("url")?.as_str()?.to_owned();
            Some(ContentPart::Image {
                source: MediaSource::Url { url },
                mime_type: None,
                detail: None,
            })
        }
        _ => None,
    }
}

#[derive(Deserialize)]
struct OpenAiToolCall {
    id: String,
    function: OpenAiFunctionCall,
}

impl TryFrom<OpenAiToolCall> for ToolCall {
    type Error = XlaiError;

    fn try_from(value: OpenAiToolCall) -> Result<Self, Self::Error> {
        let arguments = serde_json::from_str(&value.function.arguments).map_err(|error| {
            XlaiError::new(
                ErrorKind::Provider,
                format!("failed to parse tool call arguments: {error}"),
            )
        })?;

        Ok(Self {
            id: value.id,
            tool_name: value.function.name,
            arguments,
        })
    }
}

#[derive(Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    #[serde(rename = "prompt_tokens")]
    prompt: u32,
    #[serde(rename = "completion_tokens")]
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

pub(crate) fn finish_reason_from_api(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        Some("stop") => FinishReason::Stopped,
        _ => FinishReason::Completed,
    }
}
