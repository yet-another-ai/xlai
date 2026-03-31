pub use xlai_core as core;
pub use xlai_runtime::{
    Chat, ChatExecutionEvent, RuntimeBuilder, ToolCallExecutionMode, XlaiRuntime,
};
pub use xlai_runtime::{OpenAiChatModel, OpenAiConfig};

use wasm_bindgen::prelude::wasm_bindgen;

#[must_use]
pub fn builder() -> RuntimeBuilder {
    RuntimeBuilder::new()
}

#[wasm_bindgen]
#[must_use]
pub fn package_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}
