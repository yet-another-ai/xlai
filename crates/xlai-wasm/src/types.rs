//! Serde DTOs and small label helpers shared across wasm bindings.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use xlai_core::{
    ChatContent, ChatResponse, ChatRetryPolicy, FinishReason, FsEntry, FsEntryKind,
    ImageGenerationBackground, ImageGenerationOutputFormat, ImageGenerationQuality, MessageRole,
    ReasoningEffort, TokenUsage, TtsAudioFormat, TtsDeliveryMode, VoiceSpec,
};

pub(crate) const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
pub(crate) const DEFAULT_OPENAI_MODEL: &str = "gpt-4.1-mini";

/// Retry policy from JS (`retryPolicy`), mapped to [`ChatRetryPolicy`].
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WasmChatRetryPolicy {
    #[serde(default)]
    pub(crate) enabled: Option<bool>,
    #[serde(default)]
    pub(crate) max_retries: Option<u32>,
    #[serde(default)]
    pub(crate) initial_backoff_ms: Option<u64>,
    #[serde(default)]
    pub(crate) max_backoff_ms: Option<u64>,
}

impl From<WasmChatRetryPolicy> for ChatRetryPolicy {
    fn from(wasm: WasmChatRetryPolicy) -> Self {
        let mut p = ChatRetryPolicy::default();
        if let Some(enabled) = wasm.enabled {
            p.enabled = enabled;
        }
        if let Some(max_retries) = wasm.max_retries {
            p.max_retries = max_retries;
        }
        if let Some(ms) = wasm.initial_backoff_ms {
            p.initial_backoff_ms = ms;
        }
        if let Some(ms) = wasm.max_backoff_ms {
            p.max_backoff_ms = ms;
        }
        p
    }
}

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
    #[serde(default)]
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
    /// When set, used as the user message body instead of plain-text [`Self::prompt`].
    #[serde(default)]
    pub(crate) content: Option<ChatContent>,
    #[serde(default)]
    pub(crate) retry_policy: Option<WasmChatRetryPolicy>,
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
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub(crate) content: Option<ChatContent>,
    #[serde(default)]
    pub(crate) retry_policy: Option<WasmChatRetryPolicy>,
}

/// Optional local QTS config on chat/agent sessions (`manifest` only for now).
#[cfg(feature = "qts")]
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WasmQtsSessionConfig {
    #[serde(default)]
    pub(crate) manifest: Option<crate::qts_browser::QtsModelManifest>,
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
    #[serde(default)]
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub(crate) retry_policy: Option<WasmChatRetryPolicy>,
    /// When set, the runtime behind this session includes local QTS (`QtsBrowserTtsModel`).
    #[cfg(feature = "qts")]
    #[serde(default)]
    pub(crate) qts: Option<WasmQtsSessionConfig>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WasmTtsCallOptions {
    pub(crate) input: String,
    pub(crate) api_key: String,
    #[serde(default)]
    pub(crate) base_url: Option<String>,
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) tts_model: Option<String>,
    pub(crate) voice: VoiceSpec,
    #[serde(default)]
    pub(crate) response_format: Option<TtsAudioFormat>,
    #[serde(default)]
    pub(crate) speed: Option<f32>,
    #[serde(default)]
    pub(crate) instructions: Option<String>,
    #[serde(default)]
    pub(crate) delivery: Option<TtsDeliveryMode>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WasmImageGenerationCallOptions {
    pub(crate) prompt: String,
    pub(crate) api_key: String,
    #[serde(default)]
    pub(crate) base_url: Option<String>,
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) image_model: Option<String>,
    #[serde(default)]
    pub(crate) size: Option<String>,
    #[serde(default)]
    pub(crate) quality: Option<ImageGenerationQuality>,
    #[serde(default)]
    pub(crate) background: Option<ImageGenerationBackground>,
    #[serde(default)]
    pub(crate) output_format: Option<ImageGenerationOutputFormat>,
    #[serde(default)]
    pub(crate) count: Option<u32>,
}

/// Local QTS TTS call options (no OpenAI `apiKey`; see `docs/qts-wasm-browser-runtime.md`).
#[cfg(feature = "qts")]
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WasmQtsTtsCallOptions {
    pub(crate) input: String,
    pub(crate) voice: VoiceSpec,
    #[serde(default)]
    pub(crate) response_format: Option<TtsAudioFormat>,
    #[serde(default)]
    pub(crate) delivery: Option<TtsDeliveryMode>,
    #[serde(default)]
    pub(crate) manifest: Option<crate::qts_browser::QtsModelManifest>,
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
            reasoning_effort: value.reasoning_effort,
            retry_policy: value.retry_policy,
            #[cfg(feature = "qts")]
            qts: None,
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
            reasoning_effort: value.reasoning_effort,
            retry_policy: value.retry_policy,
            #[cfg(feature = "qts")]
            qts: None,
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
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
    pub(crate) retry_policy: Option<WasmChatRetryPolicy>,
    #[cfg(feature = "qts")]
    pub(crate) qts: Option<WasmQtsSessionConfig>,
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
