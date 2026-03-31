pub use xlai_core as core;
pub use xlai_runtime::{Chat, ChatExecutionEvent, RuntimeBuilder, XlaiRuntime};
pub use xlai_runtime::{OpenAiChatModel, OpenAiConfig};

#[must_use]
pub fn builder() -> RuntimeBuilder {
    RuntimeBuilder::new()
}
