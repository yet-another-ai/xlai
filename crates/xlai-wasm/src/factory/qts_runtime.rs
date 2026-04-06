//! Shared wiring of browser QTS into [`RuntimeBuilder`] (feature `qts`).

#[cfg(feature = "qts")]
use std::sync::Arc;

#[cfg(feature = "qts")]
use crate::backend_qts_browser::{QtsBrowserTtsConfig, QtsBrowserTtsModel};
#[cfg(feature = "qts")]
use wasm_bindgen::JsValue;
#[cfg(feature = "qts")]
use xlai_runtime::RuntimeBuilder;

#[cfg(feature = "qts")]
use crate::types::{WasmChatSessionOptions, WasmQtsSessionConfig};
#[cfg(feature = "qts")]
use crate::wasm_helpers::js_error;

/// If `qts` is set, attach [`QtsBrowserTtsModel`] to the runtime.
#[cfg(feature = "qts")]
pub(crate) fn apply_qts_config_to_builder(
    mut builder: RuntimeBuilder,
    qts: &Option<WasmQtsSessionConfig>,
) -> Result<RuntimeBuilder, JsValue> {
    let Some(qts) = qts else {
        return Ok(builder);
    };
    let model = QtsBrowserTtsModel::new(QtsBrowserTtsConfig {
        manifest: qts.manifest.clone(),
    });
    builder = builder.with_tts_model(Arc::new(model));
    Ok(builder)
}

/// If session options include a `qts` block, attach [`QtsBrowserTtsModel`] to the runtime.
#[cfg(feature = "qts")]
pub(crate) fn apply_qts_session_to_builder(
    builder: RuntimeBuilder,
    options: &WasmChatSessionOptions,
) -> Result<RuntimeBuilder, JsValue> {
    apply_qts_config_to_builder(builder, &options.qts)
}

/// Build an [`xlai_runtime::XlaiRuntime`] with only local QTS as TTS (for `createLocalTtsRuntime`).
#[cfg(feature = "qts")]
pub(crate) fn build_runtime_tts_only(
    manifest: Option<crate::qts_browser::QtsModelManifest>,
) -> Result<xlai_runtime::XlaiRuntime, JsValue> {
    let model = QtsBrowserTtsModel::new(QtsBrowserTtsConfig { manifest });
    RuntimeBuilder::new()
        .with_tts_model(Arc::new(model))
        .build()
        .map_err(js_error)
}
