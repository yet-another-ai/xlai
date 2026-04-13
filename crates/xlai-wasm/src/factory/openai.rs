use std::sync::Arc;

use wasm_bindgen::JsValue;
use xlai_backend_openai::OpenAiConfig;
use xlai_runtime::{FileSystem, MemoryFileSystem, RuntimeBuilder};

use crate::agent_session::WasmAgentSession;
use crate::chat_session::WasmChatSession;
use crate::types::{DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_MODEL, WasmChatSessionOptions};
use crate::wasm_helpers::js_error;

pub(crate) fn create_chat_session_inner(
    options: WasmChatSessionOptions,
    file_system: Option<Arc<MemoryFileSystem>>,
) -> Result<WasmChatSession, JsValue> {
    let file_system = file_system.map(|file_system| -> Arc<dyn FileSystem> { file_system });
    create_chat_session_with_dyn_file_system(options, file_system)
}

pub(crate) fn create_agent_session_inner(
    options: WasmChatSessionOptions,
    file_system: Option<Arc<MemoryFileSystem>>,
) -> Result<WasmAgentSession, JsValue> {
    let file_system = file_system.map(|file_system| -> Arc<dyn FileSystem> { file_system });
    create_agent_session_with_dyn_file_system(options, file_system)
}

pub(crate) fn create_chat_session_with_dyn_file_system(
    options: WasmChatSessionOptions,
    file_system: Option<Arc<dyn FileSystem>>,
) -> Result<WasmChatSession, JsValue> {
    let mut runtime_builder = RuntimeBuilder::new().with_chat_backend(OpenAiConfig::new(
        options
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_owned()),
        options.api_key.clone(),
        options
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_owned()),
    ));

    if let Some(file_system) = file_system {
        runtime_builder = runtime_builder.with_file_system(file_system);
    }

    #[cfg(feature = "qts")]
    {
        runtime_builder =
            crate::factory::qts_runtime::apply_qts_session_to_builder(runtime_builder, &options)?;
    }

    let runtime = runtime_builder.build().map_err(js_error)?;

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

    if let Some(reasoning_effort) = options.reasoning_effort {
        chat = chat.with_reasoning_effort(reasoning_effort);
    }

    if let Some(ref rp) = options.retry_policy {
        chat = chat.with_retry_policy(Some(rp.clone().into()));
    }

    Ok(WasmChatSession { inner: chat })
}

pub(crate) fn create_agent_session_with_dyn_file_system(
    options: WasmChatSessionOptions,
    file_system: Option<Arc<dyn FileSystem>>,
) -> Result<WasmAgentSession, JsValue> {
    let mut runtime_builder = RuntimeBuilder::new().with_chat_backend(OpenAiConfig::new(
        options
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_owned()),
        options.api_key.clone(),
        options
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_owned()),
    ));

    if let Some(file_system) = file_system {
        runtime_builder = runtime_builder.with_file_system(file_system);
    }

    #[cfg(feature = "qts")]
    {
        runtime_builder =
            crate::factory::qts_runtime::apply_qts_session_to_builder(runtime_builder, &options)?;
    }

    let runtime = runtime_builder.build().map_err(js_error)?;

    let mut agent = runtime.agent_session().map_err(js_error)?;

    if let Some(system_prompt) = options.system_prompt {
        agent = agent.with_system_prompt(system_prompt);
    }

    if let Some(temperature) = options.temperature {
        agent = agent.with_temperature(temperature);
    }

    if let Some(max_output_tokens) = options.max_output_tokens {
        agent = agent.with_max_output_tokens(max_output_tokens);
    }

    if let Some(reasoning_effort) = options.reasoning_effort {
        agent = agent.with_reasoning_effort(reasoning_effort);
    }

    if let Some(ref rp) = options.retry_policy {
        agent = agent.with_retry_policy(Some(rp.clone().into()));
    }

    Ok(WasmAgentSession { inner: agent })
}
