//! `wasm_bindgen` exports (one-shot `chat` / `agent` and session constructors).

use futures_util::StreamExt;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::wasm_bindgen;
use xlai_backend_openai::{OpenAiConfig, OpenAiTtsModel};
use xlai_core::{ChatContent, TtsChunk, TtsDeliveryMode, TtsModel, TtsRequest};

use crate::agent_session::WasmAgentSession;
use crate::chat_session::WasmChatSession;
use crate::factory::{create_agent_session_inner, create_chat_session_inner};
#[cfg(target_arch = "wasm32")]
use crate::factory::{
    create_agent_session_with_dyn_file_system, create_chat_session_with_dyn_file_system,
    create_transformers_agent_session_with_dyn_file_system,
    create_transformers_chat_session_with_dyn_file_system, js_file_system_arc,
    parse_transformers_session_options,
};
use crate::memory_fs::WasmMemoryFileSystem;
use crate::types::{
    DEFAULT_OPENAI_BASE_URL, DEFAULT_OPENAI_MODEL, WasmAgentRequest, WasmChatRequest,
    WasmChatSessionOptions, WasmTtsCallOptions,
};
use crate::wasm_helpers::js_error;

#[wasm_bindgen]
#[must_use]
pub fn package_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

#[wasm_bindgen]
pub async fn chat(options: JsValue) -> Result<JsValue, JsValue> {
    let WasmChatRequest {
        prompt,
        content,
        api_key,
        base_url,
        model,
        system_prompt,
        temperature,
        max_output_tokens,
    } = serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    let user_content = content.unwrap_or_else(|| ChatContent::text(prompt.clone()));
    let session_options = WasmChatSessionOptions {
        api_key,
        base_url,
        model,
        system_prompt,
        temperature,
        max_output_tokens,
    };
    let chat = create_chat_session_inner(session_options, None)?;
    chat.prompt_with_content(serde_wasm_bindgen::to_value(&user_content).map_err(js_error)?)
        .await
}

#[wasm_bindgen]
pub async fn agent(options: JsValue) -> Result<JsValue, JsValue> {
    let WasmAgentRequest {
        prompt,
        content,
        api_key,
        base_url,
        model,
        system_prompt,
        temperature,
        max_output_tokens,
    } = serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    let user_content = content.unwrap_or_else(|| ChatContent::text(prompt.clone()));
    let session_options = WasmChatSessionOptions {
        api_key,
        base_url,
        model,
        system_prompt,
        temperature,
        max_output_tokens,
    };
    let agent = create_agent_session_inner(session_options, None)?;
    agent
        .prompt_with_content(serde_wasm_bindgen::to_value(&user_content).map_err(js_error)?)
        .await
}

#[wasm_bindgen(js_name = createChatSession)]
pub fn create_chat_session(options: JsValue) -> Result<WasmChatSession, JsValue> {
    let options: WasmChatSessionOptions =
        serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    create_chat_session_inner(options, None)
}

#[wasm_bindgen(js_name = createChatSessionWithMemoryFileSystem)]
pub fn create_chat_session_with_memory_file_system(
    options: JsValue,
    file_system: &WasmMemoryFileSystem,
) -> Result<WasmChatSession, JsValue> {
    let options: WasmChatSessionOptions =
        serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    create_chat_session_inner(options, Some(file_system.inner.clone()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = createChatSessionWithFileSystem)]
pub fn create_chat_session_with_file_system(
    options: JsValue,
    file_system: JsValue,
) -> Result<WasmChatSession, JsValue> {
    let options: WasmChatSessionOptions =
        serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    create_chat_session_with_dyn_file_system(options, Some(js_file_system_arc(file_system)))
}

#[wasm_bindgen(js_name = createAgentSession)]
pub fn create_agent_session(options: JsValue) -> Result<WasmAgentSession, JsValue> {
    let options: WasmChatSessionOptions =
        serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    create_agent_session_inner(options, None)
}

#[wasm_bindgen(js_name = createAgentSessionWithMemoryFileSystem)]
pub fn create_agent_session_with_memory_file_system(
    options: JsValue,
    file_system: &WasmMemoryFileSystem,
) -> Result<WasmAgentSession, JsValue> {
    let options: WasmChatSessionOptions =
        serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    create_agent_session_inner(options, Some(file_system.inner.clone()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = createAgentSessionWithFileSystem)]
pub fn create_agent_session_with_file_system(
    options: JsValue,
    file_system: JsValue,
) -> Result<WasmAgentSession, JsValue> {
    let options: WasmChatSessionOptions =
        serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    create_agent_session_with_dyn_file_system(options, Some(js_file_system_arc(file_system)))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = createTransformersChatSession)]
pub fn create_transformers_chat_session(options: JsValue) -> Result<WasmChatSession, JsValue> {
    let options = parse_transformers_session_options(options)?;
    create_transformers_chat_session_with_dyn_file_system(options, None)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = createTransformersChatSessionWithMemoryFileSystem)]
pub fn create_transformers_chat_session_with_memory_file_system(
    options: JsValue,
    file_system: &WasmMemoryFileSystem,
) -> Result<WasmChatSession, JsValue> {
    let options = parse_transformers_session_options(options)?;
    create_transformers_chat_session_with_dyn_file_system(options, Some(file_system.inner.clone()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = createTransformersChatSessionWithFileSystem)]
pub fn create_transformers_chat_session_with_file_system(
    options: JsValue,
    file_system: JsValue,
) -> Result<WasmChatSession, JsValue> {
    let options = parse_transformers_session_options(options)?;
    create_transformers_chat_session_with_dyn_file_system(
        options,
        Some(js_file_system_arc(file_system)),
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = createTransformersAgentSession)]
pub fn create_transformers_agent_session(options: JsValue) -> Result<WasmAgentSession, JsValue> {
    let options = parse_transformers_session_options(options)?;
    create_transformers_agent_session_with_dyn_file_system(options, None)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = createTransformersAgentSessionWithMemoryFileSystem)]
pub fn create_transformers_agent_session_with_memory_file_system(
    options: JsValue,
    file_system: &WasmMemoryFileSystem,
) -> Result<WasmAgentSession, JsValue> {
    let options = parse_transformers_session_options(options)?;
    create_transformers_agent_session_with_dyn_file_system(options, Some(file_system.inner.clone()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = createTransformersAgentSessionWithFileSystem)]
pub fn create_transformers_agent_session_with_file_system(
    options: JsValue,
    file_system: JsValue,
) -> Result<WasmAgentSession, JsValue> {
    let options = parse_transformers_session_options(options)?;
    create_transformers_agent_session_with_dyn_file_system(
        options,
        Some(js_file_system_arc(file_system)),
    )
}

fn openai_tts_model(opts: &WasmTtsCallOptions) -> OpenAiTtsModel {
    let mut config = OpenAiConfig::new(
        opts.base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_owned()),
        opts.api_key.clone(),
        opts.model
            .clone()
            .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_owned()),
    );
    if let Some(ref m) = opts.tts_model {
        config = config.with_tts_model(m.clone());
    }
    OpenAiTtsModel::new(config)
}

fn tts_request_from_opts(opts: &WasmTtsCallOptions, delivery: TtsDeliveryMode) -> TtsRequest {
    TtsRequest {
        model: None,
        input: opts.input.clone(),
        voice: opts.voice.clone(),
        response_format: opts.response_format,
        speed: opts.speed,
        instructions: opts.instructions.clone(),
        delivery,
        metadata: Default::default(),
    }
}

#[wasm_bindgen]
pub async fn tts(options: JsValue) -> Result<JsValue, JsValue> {
    let opts: WasmTtsCallOptions = serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    let delivery = opts.delivery.unwrap_or(TtsDeliveryMode::Unary);
    let model = openai_tts_model(&opts);
    let request = tts_request_from_opts(&opts, delivery);
    let response = model.synthesize(request).await.map_err(js_error)?;
    serde_wasm_bindgen::to_value(&response).map_err(js_error)
}

#[wasm_bindgen(js_name = ttsStream)]
pub async fn tts_stream(options: JsValue) -> Result<JsValue, JsValue> {
    let opts: WasmTtsCallOptions = serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    let delivery = opts.delivery.unwrap_or(TtsDeliveryMode::Stream);
    let model = openai_tts_model(&opts);
    let request = tts_request_from_opts(&opts, delivery);
    let mut chunks = Vec::<TtsChunk>::new();
    let mut stream = model.synthesize_stream(request);
    while let Some(item) = stream.next().await {
        chunks.push(item.map_err(js_error)?);
    }
    serde_wasm_bindgen::to_value(&chunks).map_err(js_error)
}
