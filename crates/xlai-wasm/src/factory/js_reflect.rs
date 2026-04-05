//! Parse JS option objects for transformers.js sessions (wasm only).

use std::sync::Arc;

use js_sys::Reflect;
use wasm_bindgen::JsValue;
use xlai_runtime::FileSystem;

use crate::js_file_system::JsFileSystem;
use crate::types::WasmTransformersSessionOptions;
use crate::wasm_helpers::js_error;

#[cfg(feature = "qts")]
use crate::types::WasmQtsSessionConfig;

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

    #[cfg(feature = "qts")]
    let qts = {
        let q = Reflect::get(&options, &JsValue::from_str("qts"))
            .map_err(|e| js_error(format!("failed to read qts: {e:?}")))?;
        if q.is_null() || q.is_undefined() {
            None
        } else {
            Some(serde_wasm_bindgen::from_value::<WasmQtsSessionConfig>(q).map_err(js_error)?)
        }
    };

    Ok(WasmTransformersSessionOptions {
        model_id,
        adapter,
        system_prompt: optional_js_string_field(&options, "systemPrompt")?,
        temperature: optional_js_f32_field(&options, "temperature")?,
        max_output_tokens: optional_js_u32_field(&options, "maxOutputTokens")?,
        #[cfg(feature = "qts")]
        qts,
    })
}

fn require_js_string_field(target: &JsValue, field: &str) -> Result<String, JsValue> {
    let value = Reflect::get(target, &JsValue::from_str(field))
        .map_err(|e| js_error(format!("failed to read `{field}`: {e:?}")))?;
    value
        .as_string()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| js_error(format!("`{field}` must be a non-empty string")))
}

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

pub(crate) fn js_file_system_arc(callbacks: JsValue) -> Arc<dyn FileSystem> {
    Arc::new(JsFileSystem::new(callbacks))
}
