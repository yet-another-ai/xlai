//! Internal integration layer for **`xlai-native`** (native aggregate features).
//!
//! `xlai-wasm` does **not** depend on this crate; it re-exports directly from `xlai-core`,
//! `xlai-runtime`, and the browser backends.
//!
//! The semver-stable Rust contract is [`xlai_core`]; this crate is **not** published to crates.io.
//! It centralizes optional native backends (`llama`, `qts`) and shared re-exports used by
//! `xlai-native`.

pub use xlai_core as core;

pub use xlai_backend_gemini::{GeminiChatModel, GeminiConfig, GeminiImageGenerationModel};
pub use xlai_backend_openai::{
    OpenAiChatModel, OpenAiConfig, OpenAiImageGenerationModel, OpenAiTranscriptionModel,
    OpenAiTtsModel,
};
pub use xlai_backend_openrouter::{OpenRouterChatModel, OpenRouterConfig};
pub use xlai_backend_transformersjs::{
    TransformersJsBundle, TransformersJsChatModel, TransformersJsConfig,
};

#[cfg(feature = "llama")]
pub use xlai_backend_llama_cpp::{LlamaCppChatModel, LlamaCppConfig};

#[cfg(feature = "qts")]
pub use xlai_qts_core::{QtsTtsConfig, QtsTtsModel};

pub use xlai_runtime::{
    Agent, Chat, ChatExecutionEvent, ChatExecutionHandle, DirectoryFileSystem, FileSystem, FsEntry,
    FsEntryKind, FsPath, GeneratedImage, ImageGenerationBackground, ImageGenerationOutputFormat,
    ImageGenerationQuality, ImageGenerationRequest, ImageGenerationResponse, McpRegistry,
    MemoryFileSystem, ReadableFileSystem, RuntimeBuilder, ToolCallExecutionMode,
    WritableFileSystem, XlaiRuntime,
};

pub use xlai_core::{
    CancellationSignal, ChatExecutionConfig, ChatExecutionOverrides, ExecutionLatencyMode,
    TtsExecutionConfig, TtsExecutionOverrides,
};

#[cfg(not(target_arch = "wasm32"))]
pub use xlai_runtime::LocalFileSystem;

/// Starts a new [`RuntimeBuilder`] (same entrypoint for native and wasm facades).
#[must_use]
pub fn builder() -> RuntimeBuilder {
    RuntimeBuilder::new()
}
