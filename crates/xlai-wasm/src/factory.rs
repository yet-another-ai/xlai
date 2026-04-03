//! Build [`Chat`] / [`Agent`] runtimes for OpenAI and (wasm) transformers.js.

use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use js_sys::Reflect;
use wasm_bindgen::JsValue;
use xlai_backend_openai::OpenAiConfig;
#[cfg(target_arch = "wasm32")]
use xlai_backend_transformersjs::{TransformersJsBundle, TransformersJsConfig};
use xlai_runtime::{FileSystem, MemoryFileSystem, RuntimeBuilder};

use crate::agent_session::WasmAgentSession;
use crate::chat_session::WasmChatSession;
use crate::types::{
    DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_MODEL, WasmChatSessionOptions,
};
#[cfg(target_arch = "wasm32")]
use crate::types::WasmTransformersSessionOptions;
use crate::wasm_helpers::js_error;

#[cfg(target_arch = "wasm32")]
use crate::js_file_system::JsFileSystem;

#[cfg(target_arch = "wasm32")]
pub(crate) fn parse_transformers_session_options(
    options: JsValue,
) -> Result<WasmTransformersSessionOptions, JsValue> {
    if options.is_null() || options.is_undefined() {
        return Err(js_error(
            "createTransformers*Session: options object is required",
        ));
    }

    let model_id = require_js_string_field(&options, "modelId")?;
    let adapter = Reflect::get(&options, &JsValue::from_str("adapter"))
        .map_err(|e| js_error(format!("failed to read adapter: {e:?}")))?;
    if adapter.is_null() || adapter.is_undefined() {
        return Err(js_error(
            "createTransformers*Session: adapter must be a non-null object with generate()",
        ));
    }

    Ok(WasmTransformersSessionOptions {
        model_id,
        adapter,
        system_prompt: optional_js_string_field(&options, "systemPrompt")?,
        temperature: optional_js_f32_field(&options, "temperature")?,
        max_output_tokens: optional_js_u32_field(&options, "maxOutputTokens")?,
    })
}

#[cfg(target_arch = "wasm32")]
fn require_js_string_field(target: &JsValue, field: &str) -> Result<String, JsValue> {
    let value = Reflect::get(target, &JsValue::from_str(field))
        .map_err(|e| js_error(format!("failed to read `{field}`: {e:?}")))?;
    value
        .as_string()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| js_error(format!("`{field}` must be a non-empty string")))
}

#[cfg(target_arch = "wasm32")]
fn optional_js_string_field(target: &JsValue, field: &str) -> Result<Option<String>, JsValue> {
    let value = Reflect::get(target, &JsValue::from_str(field))
        .map_err(|e| js_error(format!("failed to read `{field}`: {e:?}")))?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    let s = value
        .as_string()
        .ok_or_else(|| js_error(format!("`{field}` must be a string when set")))?;
    Ok(Some(s))
}

#[cfg(target_arch = "wasm32")]
fn optional_js_f32_field(target: &JsValue, field: &str) -> Result<Option<f32>, JsValue> {
    let value = Reflect::get(target, &JsValue::from_str(field))
        .map_err(|e| js_error(format!("failed to read `{field}`: {e:?}")))?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    value
        .as_f64()
        .map(|n| n as f32)
        .ok_or_else(|| js_error(format!("`{field}` must be a number when set")))
        .map(Some)
}

#[cfg(target_arch = "wasm32")]
fn optional_js_u32_field(target: &JsValue, field: &str) -> Result<Option<u32>, JsValue> {
    let value = Reflect::get(target, &JsValue::from_str(field))
        .map_err(|e| js_error(format!("failed to read `{field}`: {e:?}")))?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    value
        .as_f64()
        .and_then(|n| {
            let n = n.round();
            if n >= 0.0 && n <= u32::MAX as f64 {
                Some(n as u32)
            } else {
                None
            }
        })
        .ok_or_else(|| js_error(format!("`{field}` must be a non-negative integer when set")))
        .map(Some)
}

#[cfg(target_arch = "wasm32")]
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

    Ok(WasmChatSession { inner: chat })
}

#[cfg(target_arch = "wasm32")]
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

    Ok(WasmAgentSession { inner: agent })
}

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
            .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_owned()),
        options.api_key,
        options
            .model
            .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_owned()),
    ));

    if let Some(file_system) = file_system {
        runtime_builder = runtime_builder.with_file_system(file_system);
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

    Ok(WasmChatSession { inner: chat })
}

pub(crate) fn create_agent_session_with_dyn_file_system(
    options: WasmChatSessionOptions,
    file_system: Option<Arc<dyn FileSystem>>,
) -> Result<WasmAgentSession, JsValue> {
    let mut runtime_builder = RuntimeBuilder::new().with_chat_backend(OpenAiConfig::new(
        options
            .base_url
            .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_owned()),
        options.api_key,
        options
            .model
            .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_owned()),
    ));

    if let Some(file_system) = file_system {
        runtime_builder = runtime_builder.with_file_system(file_system);
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

    Ok(WasmAgentSession { inner: agent })
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn js_file_system_arc(callbacks: JsValue) -> Arc<dyn FileSystem> {
    Arc::new(JsFileSystem::new(callbacks))
}
