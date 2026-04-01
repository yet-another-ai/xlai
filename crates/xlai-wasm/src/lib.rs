pub use xlai_backend_openai::{OpenAiChatModel, OpenAiConfig};
pub use xlai_core as core;
pub use xlai_runtime::{
    Chat, ChatExecutionEvent, DirectoryFileSystem, FileSystem, FsEntry, FsEntryKind, FsPath,
    MemoryFileSystem, ReadableFileSystem, RuntimeBuilder, ToolCallExecutionMode,
    WritableFileSystem, XlaiRuntime,
};

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::{JsValue, wasm_bindgen};
use xlai_core::{ChatResponse, FinishReason, MessageRole, TokenUsage};

const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_OPENAI_MODEL: &str = "gpt-4.1-mini";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WasmChatRequest {
    prompt: String,
    api_key: String,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WasmChatResponse {
    message: WasmChatMessage,
    finish_reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<WasmChatUsage>,
}

#[derive(Serialize)]
struct WasmChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WasmChatUsage {
    input_tokens: u32,
    output_tokens: u32,
    total_tokens: u32,
}

impl From<ChatResponse> for WasmChatResponse {
    fn from(response: ChatResponse) -> Self {
        Self {
            message: WasmChatMessage {
                role: message_role_label(response.message.role),
                content: response.message.content,
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

#[must_use]
pub fn builder() -> RuntimeBuilder {
    RuntimeBuilder::new()
}

#[wasm_bindgen]
#[must_use]
pub fn package_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

#[wasm_bindgen]
pub async fn chat(options: JsValue) -> Result<JsValue, JsValue> {
    let options: WasmChatRequest = serde_wasm_bindgen::from_value(options).map_err(js_error)?;

    let runtime = RuntimeBuilder::new()
        .with_chat_backend(OpenAiConfig::new(
            options
                .base_url
                .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_owned()),
            options.api_key,
            options
                .model
                .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_owned()),
        ))
        .build()
        .map_err(js_error)?;

    let mut chat = runtime.chat_session();

    if let Some(system_prompt) = options.system_prompt {
        chat = chat.with_system_prompt(system_prompt);
    }

    if let Some(temperature) = options.temperature {
        chat = chat.with_temperature(temperature);
    }

    if let Some(max_output_tokens) = options.max_output_tokens {
        chat = chat.with_max_output_tokens(max_output_tokens);
    }

    let response = chat.prompt(options.prompt).await.map_err(js_error)?;
    serde_wasm_bindgen::to_value(&WasmChatResponse::from(response)).map_err(js_error)
}

fn js_error(error: impl ToString) -> JsValue {
    JsValue::from_str(&error.to_string())
}

const fn finish_reason_label(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Completed => "completed",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::Length => "length",
        FinishReason::Stopped => "stopped",
    }
}

const fn message_role_label(role: MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use xlai_core::{ChatMessage, FinishReason, MessageRole, TokenUsage};

    use super::WasmChatResponse;

    #[test]
    fn wasm_chat_response_uses_js_friendly_field_values() {
        let response = WasmChatResponse::from(xlai_core::ChatResponse {
            message: ChatMessage {
                role: MessageRole::Assistant,
                content: "hello from wasm".to_owned(),
                tool_name: None,
                tool_call_id: None,
                metadata: BTreeMap::new(),
            },
            tool_calls: Vec::new(),
            usage: Some(TokenUsage {
                input_tokens: 5,
                output_tokens: 7,
                total_tokens: 12,
            }),
            finish_reason: FinishReason::Stopped,
            metadata: BTreeMap::new(),
        });

        assert_eq!(response.message.role, "assistant");
        assert_eq!(response.message.content, "hello from wasm");
        assert_eq!(response.finish_reason, "stopped");

        assert!(response.usage.is_some());
        let Some(usage) = response.usage else {
            return;
        };

        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.total_tokens, 12);
    }
}
