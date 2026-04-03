//! Agent session handle.

#[cfg(target_arch = "wasm32")]
use js_sys::Function;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
use xlai_core::ChatContent;
#[cfg(target_arch = "wasm32")]
use xlai_core::{ToolDefinition, ToolResult};
use xlai_runtime::Agent;

#[cfg(target_arch = "wasm32")]
use js_sys::Promise;

use crate::wasm_helpers::{js_error, serialize_chat_response};
#[cfg(target_arch = "wasm32")]
use crate::wasm_helpers::{tool_js_error, tool_js_value_error};

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

    pub async fn prompt(&self, content: String) -> Result<JsValue, JsValue> {
        serialize_chat_response(self.inner.prompt(content).await.map_err(js_error)?)
    }

    #[wasm_bindgen(js_name = promptWithContent)]
    pub async fn prompt_with_content(&self, content: JsValue) -> Result<JsValue, JsValue> {
        let content: ChatContent = serde_wasm_bindgen::from_value(content).map_err(js_error)?;
        serialize_chat_response(self.inner.prompt_content(content).await.map_err(js_error)?)
    }
}
