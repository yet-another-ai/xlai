//! WebAssembly bindings for xlai (OpenAI-compatible chat, optional transformers.js).
//!
//! Public types match `xlai-facade` / `xlai-native`; internal modules also use `xlai_core` / `xlai_runtime` directly.

pub use xlai_facade::builder;
pub use xlai_facade::core;
pub use xlai_facade::{
    Agent, Chat, ChatExecutionEvent, DirectoryFileSystem, FileSystem, FsEntry, FsEntryKind, FsPath,
    McpRegistry, MemoryFileSystem, OpenAiChatModel, OpenAiConfig, OpenAiTranscriptionModel,
    OpenAiTtsModel, ReadableFileSystem, RuntimeBuilder, ToolCallExecutionMode, WritableFileSystem,
    XlaiRuntime,
};
#[cfg(target_arch = "wasm32")]
pub use xlai_facade::{TransformersJsBundle, TransformersJsChatModel, TransformersJsConfig};

mod agent_session;
mod api;
#[cfg(feature = "qts")]
mod backend_qts_browser;
mod chat_session;
mod factory;
mod memory_fs;
#[cfg(feature = "qts")]
mod qts_browser;
#[cfg(feature = "qts")]
mod tts_runtime;
mod types;
mod wasm_helpers;

#[cfg(target_arch = "wasm32")]
mod js_file_system;

pub use agent_session::WasmAgentSession;
pub use api::{
    agent, chat, create_agent_session, create_agent_session_with_memory_file_system,
    create_chat_session, create_chat_session_with_memory_file_system, package_version, tts,
    tts_stream,
};
#[cfg(target_arch = "wasm32")]
pub use api::{
    create_agent_session_with_file_system, create_transformers_agent_session,
    create_transformers_agent_session_with_file_system,
    create_transformers_agent_session_with_memory_file_system, create_transformers_chat_session,
    create_transformers_chat_session_with_file_system,
    create_transformers_chat_session_with_memory_file_system,
};
#[cfg(feature = "qts")]
pub use api::{
    qts_browser_tts, qts_browser_tts_capabilities, qts_browser_tts_stream,
    validate_qts_model_manifest,
};
pub use chat_session::WasmChatSession;
pub use memory_fs::WasmMemoryFileSystem;
#[cfg(feature = "qts")]
pub use tts_runtime::{WasmLocalTtsRuntime, create_local_tts_runtime};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;
    use xlai_core::{
        ChatContent, ChatMessage, FinishReason, FsEntry, FsEntryKind, FsPath, MessageRole,
        TokenUsage,
    };

    use crate::factory::create_agent_session_inner;
    #[cfg(feature = "qts")]
    use crate::factory::create_chat_session_inner;
    #[cfg(feature = "qts")]
    use crate::types::WasmQtsSessionConfig;
    use crate::types::{
        WasmAgentRequest, WasmChatRequest, WasmChatResponse, WasmChatSessionOptions, WasmFsEntry,
    };

    #[test]
    fn wasm_chat_response_uses_js_friendly_field_values() {
        let response = WasmChatResponse::from(xlai_core::ChatResponse {
            message: ChatMessage {
                role: MessageRole::Assistant,
                content: ChatContent::text("hello from wasm"),
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
        assert_eq!(response.message.content, json!("hello from wasm"));
        assert_eq!(response.finish_reason, "stopped");

        assert!(response.usage.is_some());
        let Some(usage) = response.usage else {
            return;
        };

        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.total_tokens, 12);
    }

    #[test]
    fn wasm_fs_entry_uses_js_friendly_field_values() {
        let entry = WasmFsEntry::from(FsEntry {
            path: FsPath::from("/docs/readme.md"),
            kind: FsEntryKind::File,
        });

        assert_eq!(entry.path, "/docs/readme.md");
        assert_eq!(entry.kind, "file");
    }

    #[test]
    fn wasm_chat_request_deserializes_optional_multimodal_content() {
        let raw = serde_json::json!({
            "prompt": "fallback",
            "apiKey": "k",
            "content": {
                "parts": [
                    {"type": "text", "text": "Describe this."},
                    {
                        "type": "image",
                        "source": {"kind": "url", "url": "https://example.com/i.png"},
                        "mime_type": null,
                        "detail": null
                    }
                ]
            }
        });
        let result: Result<WasmChatRequest, _> = serde_json::from_value(raw);
        assert!(result.is_ok(), "deserialize WasmChatRequest");
        let Ok(req) = result else {
            return;
        };
        assert!(req.content.is_some());
        let Some(content) = req.content.as_ref() else {
            return;
        };
        assert_eq!(content.parts.len(), 2);
    }

    #[test]
    fn wasm_agent_request_maps_into_session_options() {
        let options = WasmChatSessionOptions::from(WasmAgentRequest {
            prompt: "Hello".to_owned(),
            api_key: "test-key".to_owned(),
            base_url: Some("https://example.com/v1".to_owned()),
            model: Some("gpt-test".to_owned()),
            system_prompt: Some("Be concise.".to_owned()),
            temperature: Some(0.2),
            max_output_tokens: Some(512),
            content: None,
            agent_loop: Some(false),
            retry_policy: None,
        });

        assert_eq!(options.api_key, "test-key");
        assert_eq!(options.base_url.as_deref(), Some("https://example.com/v1"));
        assert_eq!(options.model.as_deref(), Some("gpt-test"));
        assert_eq!(options.system_prompt.as_deref(), Some("Be concise."));
        assert_eq!(options.temperature, Some(0.2));
        assert_eq!(options.max_output_tokens, Some(512));
        assert_eq!(options.agent_loop, Some(false));
    }

    #[test]
    fn wasm_agent_session_can_be_created_from_options() {
        let session = create_agent_session_inner(
            WasmChatSessionOptions {
                api_key: "test-key".to_owned(),
                base_url: Some("https://example.com/v1".to_owned()),
                model: Some("gpt-test".to_owned()),
                system_prompt: Some("Use tools.".to_owned()),
                temperature: Some(0.1),
                max_output_tokens: Some(256),
                agent_loop: None,
                retry_policy: None,
                #[cfg(feature = "qts")]
                qts: None,
            },
            None,
        );

        assert!(session.is_ok());
    }

    #[cfg(feature = "qts")]
    #[test]
    fn chat_session_with_qts_config_builds() {
        let session = create_chat_session_inner(
            WasmChatSessionOptions {
                api_key: "test-key".to_owned(),
                base_url: Some("https://example.com/v1".to_owned()),
                model: Some("gpt-test".to_owned()),
                system_prompt: None,
                temperature: None,
                max_output_tokens: None,
                agent_loop: None,
                retry_policy: None,
                qts: Some(WasmQtsSessionConfig::default()),
            },
            None,
        );
        assert!(session.is_ok());
    }

    #[cfg(feature = "qts")]
    #[test]
    fn tts_only_runtime_builds_via_factory() {
        let runtime = crate::factory::qts_runtime::build_runtime_tts_only(None);
        assert!(runtime.is_ok());
    }
}
