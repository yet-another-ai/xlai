//! Narrow WASM handle for local QTS via [`XlaiRuntime`] (feature `qts`).

#[cfg(feature = "qts")]
use std::sync::Arc;

#[cfg(feature = "qts")]
use futures_util::StreamExt;
#[cfg(feature = "qts")]
use wasm_bindgen::JsValue;
#[cfg(feature = "qts")]
use wasm_bindgen::prelude::wasm_bindgen;
#[cfg(feature = "qts")]
use wasm_bindgen_futures::future_to_promise;
#[cfg(feature = "qts")]
use xlai_core::{TtsChunk, TtsDeliveryMode};

#[cfg(feature = "qts")]
use crate::api::{tts_request_from_qts_opts, xlai_error_to_js};
#[cfg(feature = "qts")]
use crate::factory::qts_runtime::build_runtime_tts_only;
#[cfg(feature = "qts")]
use crate::types::{WasmQtsSessionConfig, WasmQtsTtsCallOptions};
#[cfg(feature = "qts")]
use crate::wasm_helpers::js_error;

/// Runtime containing only local QTS TTS (for `synthesize` / `stream_synthesize`).
#[cfg(feature = "qts")]
#[wasm_bindgen]
pub struct WasmLocalTtsRuntime {
    inner: Arc<xlai_runtime::XlaiRuntime>,
}

/// Build from optional session-style options (`{ manifest?: ... }`).
#[cfg(feature = "qts")]
#[wasm_bindgen(js_name = createLocalTtsRuntime)]
pub fn create_local_tts_runtime(options: JsValue) -> Result<WasmLocalTtsRuntime, JsValue> {
    let cfg: WasmQtsSessionConfig = if options.is_null() || options.is_undefined() {
        WasmQtsSessionConfig::default()
    } else {
        serde_wasm_bindgen::from_value(options).map_err(js_error)?
    };
    let runtime = build_runtime_tts_only(cfg.manifest)?;
    Ok(WasmLocalTtsRuntime {
        inner: Arc::new(runtime),
    })
}

#[cfg(feature = "qts")]
#[wasm_bindgen]
impl WasmLocalTtsRuntime {
    #[wasm_bindgen(js_name = localTtsSynthesize)]
    pub fn local_tts_synthesize(&self, options: JsValue) -> js_sys::Promise {
        let opts: WasmQtsTtsCallOptions = match serde_wasm_bindgen::from_value(options) {
            Ok(v) => v,
            Err(e) => return future_to_promise(async move { Err(js_error(e)) }),
        };
        let delivery = opts.delivery.unwrap_or(TtsDeliveryMode::Unary);
        let request = tts_request_from_qts_opts(&opts, delivery);
        let runtime = self.inner.clone();
        future_to_promise(async move {
            match runtime.synthesize(request).await {
                Ok(response) => serde_wasm_bindgen::to_value(&response).map_err(js_error),
                Err(e) => Err(xlai_error_to_js(e)),
            }
        })
    }

    #[wasm_bindgen(js_name = localTtsStream)]
    pub fn local_tts_stream(&self, options: JsValue) -> js_sys::Promise {
        let opts: WasmQtsTtsCallOptions = match serde_wasm_bindgen::from_value(options) {
            Ok(v) => v,
            Err(e) => return future_to_promise(async move { Err(js_error(e)) }),
        };
        let delivery = opts.delivery.unwrap_or(TtsDeliveryMode::Stream);
        let request = tts_request_from_qts_opts(&opts, delivery);
        let runtime = self.inner.clone();
        future_to_promise(async move {
            let mut stream = runtime
                .stream_synthesize(request)
                .map_err(xlai_error_to_js)?;
            let mut chunks = Vec::<TtsChunk>::new();
            while let Some(item) = stream.next().await {
                match item {
                    Ok(c) => chunks.push(c),
                    Err(e) => return Err(xlai_error_to_js(e)),
                }
            }
            serde_wasm_bindgen::to_value(&chunks).map_err(js_error)
        })
    }
}
