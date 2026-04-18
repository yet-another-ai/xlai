//! JS error conversion and small serialization helpers.

use wasm_bindgen::JsValue;
use xlai_core::ChatResponse;

#[cfg(target_arch = "wasm32")]
use js_sys::{Function, Reflect};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use xlai_core::{ErrorKind, XlaiError};

use crate::types::WasmChatResponse;

pub(crate) fn js_error(error: impl ToString) -> JsValue {
    JsValue::from_str(&error.to_string())
}

pub(crate) fn serialize_chat_response(response: ChatResponse) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&WasmChatResponse::from(response)).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn tool_js_error(error: impl ToString) -> XlaiError {
    XlaiError::new(ErrorKind::Tool, error.to_string())
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn tool_js_value_error(error: JsValue) -> XlaiError {
    XlaiError::new(
        ErrorKind::Tool,
        error
            .as_string()
            .unwrap_or_else(|| format!("javascript callback failed: {error:?}")),
    )
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn provider_js_value_error(error: JsValue) -> XlaiError {
    XlaiError::new(
        ErrorKind::Provider,
        error
            .as_string()
            .unwrap_or_else(|| format!("javascript callback failed: {error:?}")),
    )
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn file_system_js_error(error: impl ToString) -> XlaiError {
    XlaiError::new(xlai_core::ErrorKind::FileSystem, error.to_string())
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn file_system_js_value_error(error: JsValue) -> XlaiError {
    XlaiError::new(
        xlai_core::ErrorKind::FileSystem,
        error
            .as_string()
            .unwrap_or_else(|| format!("filesystem callback failed: {error:?}")),
    )
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn js_callback(target: &JsValue, name: &str) -> Result<Function, XlaiError> {
    let value =
        Reflect::get(target, &JsValue::from_str(name)).map_err(file_system_js_value_error)?;
    value.dyn_into::<Function>().map_err(|_| {
        XlaiError::new(
            xlai_core::ErrorKind::FileSystem,
            format!("filesystem callback `{name}` must be a function"),
        )
    })
}
