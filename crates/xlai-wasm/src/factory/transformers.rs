//! transformers.js-backed chat and agent sessions (wasm only).

use std::sync::Arc;

use wasm_bindgen::JsValue;
use xlai_backend_transformersjs::{TransformersJsBundle, TransformersJsConfig};
use xlai_runtime::{FileSystem, RuntimeBuilder};

use crate::agent_session::WasmAgentSession;
use crate::chat_session::WasmChatSession;
use crate::factory::openai::apply_runtime_execution_overrides;
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
        reasoning_effort: _,
        retry_policy,
        chat_execution,
        runtime_chat_execution_defaults,
        runtime_tts_execution_defaults,
        default_max_tool_round_trips,
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
    runtime_builder = apply_runtime_execution_overrides(
        runtime_builder,
        runtime_chat_execution_defaults.as_ref(),
        runtime_tts_execution_defaults.as_ref(),
        default_max_tool_round_trips,
    );

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

    if let Some(ref o) = chat_execution {
        chat = chat.with_chat_execution_overrides(o.clone().into());
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
        reasoning_effort: _,
        retry_policy,
        chat_execution,
        runtime_chat_execution_defaults,
        runtime_tts_execution_defaults,
        default_max_tool_round_trips,
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
    runtime_builder = apply_runtime_execution_overrides(
        runtime_builder,
        runtime_chat_execution_defaults.as_ref(),
        runtime_tts_execution_defaults.as_ref(),
        default_max_tool_round_trips,
    );

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

    if let Some(ref rp) = retry_policy {
        agent = agent.with_retry_policy(Some(rp.clone().into()));
    }

    if let Some(ref o) = chat_execution {
        agent = agent.with_chat_execution_overrides(o.clone().into());
    }

    Ok(WasmAgentSession { inner: agent })
}
