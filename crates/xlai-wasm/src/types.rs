//! Serde DTOs and small label helpers shared across wasm bindings.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use xlai_core::{ChatContent, ChatResponse, FinishReason, FsEntry, FsEntryKind, MessageRole, TokenUsage};

pub(crate) const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
pub(crate) const DEFAULT_OPENAI_MODEL: &str = "gpt-4.1-mini";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WasmChatRequest {
    pub(crate) prompt: String,
    pub(crate) api_key: String,
    #[serde(default)]
    pub(crate) base_url: Option<String>,
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) system_prompt: Option<String>,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
    #[serde(default)]
    pub(crate) max_output_tokens: Option<u32>,
    /// When set, used as the user message body instead of plain-text [`Self::prompt`].
    #[serde(default)]
    pub(crate) content: Option<ChatContent>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WasmAgentRequest {
    pub(crate) prompt: String,
    pub(crate) api_key: String,
    #[serde(default)]
    pub(crate) base_url: Option<String>,
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) system_prompt: Option<String>,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
    #[serde(default)]
    pub(crate) max_output_tokens: Option<u32>,
    #[serde(default)]
    pub(crate) content: Option<ChatContent>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WasmChatSessionOptions {
    pub(crate) api_key: String,
    #[serde(default)]
    pub(crate) base_url: Option<String>,
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) system_prompt: Option<String>,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
    #[serde(default)]
    pub(crate) max_output_tokens: Option<u32>,
}

impl From<WasmChatRequest> for WasmChatSessionOptions {
    fn from(value: WasmChatRequest) -> Self {
        Self {
            api_key: value.api_key,
            base_url: value.base_url,
            model: value.model,
            system_prompt: value.system_prompt,
            temperature: value.temperature,
            max_output_tokens: value.max_output_tokens,
        }
    }
}

impl From<WasmAgentRequest> for WasmChatSessionOptions {
    fn from(value: WasmAgentRequest) -> Self {
        Self {
            api_key: value.api_key,
            base_url: value.base_url,
            model: value.model,
            system_prompt: value.system_prompt,
            temperature: value.temperature,
            max_output_tokens: value.max_output_tokens,
        }
    }
}

/// Options for browser sessions backed by `transformers.js` (wasm only).
#[cfg(target_arch = "wasm32")]
pub(crate) struct WasmTransformersSessionOptions {
    /// Hugging Face / Xenova model id passed to the JS adapter.
    pub(crate) model_id: String,
    /// JavaScript object with `async generate(request)` (see `xlai-backend-transformersjs`).
    pub(crate) adapter: wasm_bindgen::JsValue,
    pub(crate) system_prompt: Option<String>,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_output_tokens: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WasmChatResponse {
    pub(crate) message: WasmChatMessage,
    pub(crate) finish_reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<WasmChatUsage>,
}

#[derive(Serialize)]
pub(crate) struct WasmChatMessage {
    pub(crate) role: &'static str,
    /// Structured [`ChatContent`] as JSON (plain string or `{ "parts": [...] }`).
    pub(crate) content: JsonValue,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WasmChatUsage {
    pub(crate) input_tokens: u32,
    pub(crate) output_tokens: u32,
    pub(crate) total_tokens: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WasmFsEntry {
    pub(crate) path: String,
    pub(crate) kind: &'static str,
}

#[cfg(target_arch = "wasm32")]
#[derive(Deserialize)]
pub(crate) struct WasmFsEntryInput {
    pub(crate) path: String,
    pub(crate) kind: String,
}

impl From<ChatResponse> for WasmChatResponse {
    fn from(response: ChatResponse) -> Self {
        let content = serde_json::to_value(&response.message.content).unwrap_or_else(|_| {
            JsonValue::String(response.message.content.text_parts_concatenated())
        });
        Self {
            message: WasmChatMessage {
                role: message_role_label(response.message.role),
                content,
            },
            finish_reason: finish_reason_label(response.finish_reason),
            usage: response.usage.map(Into::into),
        }
    }
}

impl From<TokenUsage> for WasmChatUsage {
    fn from(usage: TokenUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

impl From<FsEntry> for WasmFsEntry {
    fn from(entry: FsEntry) -> Self {
        Self {
            path: entry.path.as_str().to_owned(),
            kind: fs_entry_kind_label(entry.kind),
        }
    }
}

pub(crate) const fn finish_reason_label(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Completed => "completed",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::Length => "length",
        FinishReason::Stopped => "stopped",
    }
}

pub(crate) const fn message_role_label(role: MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

pub(crate) const fn fs_entry_kind_label(kind: FsEntryKind) -> &'static str {
    match kind {
        FsEntryKind::File => "file",
        FsEntryKind::Directory => "directory",
    }
}
