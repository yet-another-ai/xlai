//! Internal integration layer: re-exports for `xlai-native` and `xlai-wasm`.
//!
//! The semver-stable Rust contract is [`xlai_core`]; this crate is **not** published to crates.io
//! and exists only to keep platform binding crates thin. Keeps re-exports and
//! [`RuntimeBuilder::new`] wiring in one place.

pub use xlai_core as core;

pub use xlai_backend_gemini::{GeminiChatModel, GeminiConfig, GeminiImageGenerationModel};
pub use xlai_backend_openai::{
    OpenAiChatModel, OpenAiConfig, OpenAiImageGenerationModel, OpenAiTranscriptionModel,
    OpenAiTtsModel,
};
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
