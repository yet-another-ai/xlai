//! Native platform entrypoint for desktop and server Rust applications.
//!
//! This crate lists **explicit** public re-exports (no blanket `pub use` of an internal aggregate).
//! Semver-stable domain contracts live in `xlai-core`. Session and runtime types come from
//! `xlai-runtime`, wired for native backends through the workspace-only `xlai-facade` crate.
//!
//! # Gemini
//!
//! The Google Gemini backend (`xlai-backend-gemini`) is **not** published to crates.io.
//! Prefer importing Gemini types from [`gemini`] for clarity; root-level `Gemini*` names are kept
//! for compatibility with earlier `xlai_native::*` paths.

pub use xlai_facade::core;

pub use xlai_facade::{
    Agent, CancellationSignal, Chat, ChatExecutionConfig, ChatExecutionEvent, ChatExecutionHandle,
    ChatExecutionOverrides, DirectoryFileSystem, EmbeddingRequest, EmbeddingResponse,
    ExecutionLatencyMode, FileSystem, FsEntry, FsEntryKind, FsPath, GeminiChatModel, GeminiConfig,
    GeminiEmbeddingModel, GeminiImageGenerationModel, GeneratedImage, ImageGenerationBackground,
    ImageGenerationOutputFormat, ImageGenerationQuality, ImageGenerationRequest,
    ImageGenerationResponse, LlamaCppChatModel, LlamaCppConfig, LlamaCppEmbeddingModel,
    McpRegistry, MemoryFileSystem, OpenAiChatModel, OpenAiConfig, OpenAiEmbeddingModel,
    OpenAiImageGenerationModel, OpenAiTranscriptionModel, OpenAiTtsModel, OpenRouterChatModel,
    OpenRouterConfig, ReadableFileSystem, RuntimeBuilder, ToolCallExecutionMode,
    TransformersJsBundle, TransformersJsChatModel, TransformersJsConfig,
    TransformersJsEmbeddingModel, TtsExecutionConfig, TtsExecutionOverrides, WritableFileSystem,
    XlaiRuntime,
};

#[cfg(feature = "qts")]
pub use xlai_facade::{QtsTtsConfig, QtsTtsModel};

#[cfg(not(target_arch = "wasm32"))]
pub use xlai_facade::LocalFileSystem;

/// Google Gemini backend types (`xlai-backend-gemini` is workspace-only, not on crates.io).
pub mod gemini {
    pub use xlai_facade::{GeminiChatModel, GeminiConfig, GeminiImageGenerationModel};
}

/// Starts a new [`RuntimeBuilder`].
#[must_use]
pub fn builder() -> xlai_facade::RuntimeBuilder {
    xlai_facade::builder()
}
