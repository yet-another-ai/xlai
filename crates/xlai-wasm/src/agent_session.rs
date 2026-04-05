//! Agent session handle.

use futures_util::StreamExt;
#[cfg(target_arch = "wasm32")]
use js_sys::Function;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
use xlai_core::ChatContent;
#[cfg(target_arch = "wasm32")]
use xlai_core::{ChatMessage, ToolDefinition, ToolResult};
use xlai_runtime::Agent;

#[cfg(target_arch = "wasm32")]
use js_sys::Promise;

use crate::wasm_helpers::{js_error, serialize_chat_response};
#[cfg(target_arch = "wasm32")]
use crate::wasm_helpers::{provider_js_value_error, tool_js_error, tool_js_value_error};

#[wasm_bindgen(js_name = AgentSession)]
pub struct WasmAgentSession {
    pub(crate) inner: Agent,
}

#[wasm_bindgen(js_class = AgentSession)]
impl WasmAgentSession {
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen(js_name = registerTool)]
    pub fn register_tool(
        &mut self,
        definition: JsValue,
        callback: Function,
    ) -> Result<(), JsValue> {
        let definition: ToolDefinition =
            serde_wasm_bindgen::from_value(definition).map_err(js_error)?;

        self.inner.register_tool(definition, move |arguments| {
            let callback = callback.clone();
            async move {
                let arguments = serde_wasm_bindgen::to_value(&arguments).map_err(tool_js_error)?;
                let result = callback
                    .call1(&JsValue::NULL, &arguments)
                    .map_err(tool_js_value_error)?;
                let result = JsFuture::from(Promise::resolve(&result))
                    .await
                    .map_err(tool_js_value_error)?;
                serde_wasm_bindgen::from_value::<ToolResult>(result).map_err(tool_js_error)
            }
        });

        Ok(())
    }

    /// Registers a JS async callback invoked before each **streamed** agent-loop model call
    /// (see `streamPrompt` / `streamPromptWithContent`). Signature:
    /// `(messages: ChatMessage[], estimatedInputTokens: number | null) => Promise<ChatMessage[]>`.
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen(js_name = registerContextCompressor)]
    pub fn register_context_compressor(&mut self, callback: Function) -> Result<(), JsValue> {
        self.inner
            .register_context_compressor(move |messages, estimated_input_tokens| {
                let callback = callback.clone();
                async move {
                    let messages_js =
                        serde_wasm_bindgen::to_value(&messages).map_err(tool_js_error)?;
                    let est_js = match estimated_input_tokens {
                        Some(n) => JsValue::from_f64(f64::from(n)),
                        None => JsValue::NULL,
                    };
                    let result = callback
                        .call2(&JsValue::NULL, &messages_js, &est_js)
                        .map_err(provider_js_value_error)?;
                    let result = JsFuture::from(Promise::resolve(&result))
                        .await
                        .map_err(provider_js_value_error)?;
                    serde_wasm_bindgen::from_value::<Vec<ChatMessage>>(result)
                        .map_err(tool_js_error)
                }
            });

        Ok(())
    }

    /// Runs the agent streaming loop (tool round-trips when enabled) and returns all execution
    /// events as a JSON array (`kind` + `data` per item).
    #[wasm_bindgen(js_name = streamPrompt)]
    pub async fn stream_prompt(&self, content: String) -> Result<JsValue, JsValue> {
        let mut stream = self.inner.stream_prompt(content);
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            let item = item.map_err(|e| js_error(e.to_string()))?;
            events.push(item);
        }
        serde_wasm_bindgen::to_value(&events).map_err(js_error)
    }

    #[wasm_bindgen(js_name = streamPromptWithContent)]
    pub async fn stream_prompt_with_content(&self, content: JsValue) -> Result<JsValue, JsValue> {
        let content: ChatContent = serde_wasm_bindgen::from_value(content).map_err(js_error)?;
        let mut stream = self.inner.stream_prompt_content(content);
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            let item = item.map_err(|e| js_error(e.to_string()))?;
            events.push(item);
        }
        serde_wasm_bindgen::to_value(&events).map_err(js_error)
    }

    pub async fn prompt(&self, content: String) -> Result<JsValue, JsValue> {
        serialize_chat_response(self.inner.prompt(content).await.map_err(js_error)?)
    }

    #[wasm_bindgen(js_name = promptWithContent)]
    pub async fn prompt_with_content(&self, content: JsValue) -> Result<JsValue, JsValue> {
        let content: ChatContent = serde_wasm_bindgen::from_value(content).map_err(js_error)?;
        serialize_chat_response(self.inner.prompt_content(content).await.map_err(js_error)?)
    }
}
