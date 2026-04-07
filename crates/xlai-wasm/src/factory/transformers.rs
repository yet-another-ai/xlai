//! transformers.js-backed chat and agent sessions (wasm only).

use std::sync::Arc;

use wasm_bindgen::JsValue;
use xlai_backend_transformersjs::{TransformersJsBundle, TransformersJsConfig};
use xlai_runtime::{FileSystem, RuntimeBuilder};

use crate::agent_session::WasmAgentSession;
use crate::chat_session::WasmChatSession;
use crate::types::WasmTransformersSessionOptions;
use crate::wasm_helpers::js_error;

#[cfg(feature = "qts")]
use crate::factory::qts_runtime::apply_qts_config_to_builder;

pub(crate) fn create_transformers_chat_session_with_dyn_file_system(
    options: WasmTransformersSessionOptions,
    file_system: Option<Arc<dyn FileSystem>>,
) -> Result<WasmChatSession, JsValue> {
    let WasmTransformersSessionOptions {
        model_id,
        adapter,
        system_prompt,
        temperature,
        max_output_tokens,
        agent_loop: _,
        retry_policy,
        #[cfg(feature = "qts")]
        qts,
    } = options;

    let mut config = TransformersJsConfig::new(model_id);
    if let Some(temperature) = temperature {
        config = config.with_temperature(temperature);
    }
    if let Some(max_output_tokens) = max_output_tokens {
        config = config.with_max_output_tokens(max_output_tokens);
    }

    let bundle = TransformersJsBundle::new(config, adapter);
    let mut runtime_builder = RuntimeBuilder::new().with_chat_backend(bundle);

    if let Some(file_system) = file_system {
        runtime_builder = runtime_builder.with_file_system(file_system);
    }

    #[cfg(feature = "qts")]
    {
        runtime_builder = apply_qts_config_to_builder(runtime_builder, &qts)?;
    }

    let runtime = runtime_builder.build().map_err(js_error)?;

    let mut chat = runtime.chat_session();

    if let Some(system_prompt) = system_prompt {
        chat = chat.with_system_prompt(system_prompt);
    }

    if let Some(temperature) = temperature {
        chat = chat.with_temperature(temperature);
    }

    if let Some(max_output_tokens) = max_output_tokens {
        chat = chat.with_max_output_tokens(max_output_tokens);
    }

    if let Some(ref rp) = retry_policy {
        chat = chat.with_retry_policy(Some(rp.clone().into()));
    }

    Ok(WasmChatSession { inner: chat })
}

pub(crate) fn create_transformers_agent_session_with_dyn_file_system(
    options: WasmTransformersSessionOptions,
    file_system: Option<Arc<dyn FileSystem>>,
) -> Result<WasmAgentSession, JsValue> {
    let WasmTransformersSessionOptions {
        model_id,
        adapter,
        system_prompt,
        temperature,
        max_output_tokens,
        agent_loop,
        retry_policy,
        #[cfg(feature = "qts")]
        qts,
    } = options;

    let mut config = TransformersJsConfig::new(model_id);
    if let Some(temperature) = temperature {
        config = config.with_temperature(temperature);
    }
    if let Some(max_output_tokens) = max_output_tokens {
        config = config.with_max_output_tokens(max_output_tokens);
    }

    let bundle = TransformersJsBundle::new(config, adapter);
    let mut runtime_builder = RuntimeBuilder::new().with_chat_backend(bundle);

    if let Some(file_system) = file_system {
        runtime_builder = runtime_builder.with_file_system(file_system);
    }

    #[cfg(feature = "qts")]
    {
        runtime_builder = apply_qts_config_to_builder(runtime_builder, &qts)?;
    }

    let runtime = runtime_builder.build().map_err(js_error)?;

    let mut agent = runtime.agent_session().map_err(js_error)?;

    if let Some(system_prompt) = system_prompt {
        agent = agent.with_system_prompt(system_prompt);
    }

    if let Some(temperature) = temperature {
        agent = agent.with_temperature(temperature);
    }

    if let Some(max_output_tokens) = max_output_tokens {
        agent = agent.with_max_output_tokens(max_output_tokens);
    }

    if agent_loop == Some(false) {
        agent = agent.with_agent_loop_enabled(false);
    }

    if let Some(ref rp) = retry_policy {
        agent = agent.with_retry_policy(Some(rp.clone().into()));
    }

    Ok(WasmAgentSession { inner: agent })
}
