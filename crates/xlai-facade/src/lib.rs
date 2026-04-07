//! Shared public surface for `xlai-native` and `xlai-wasm`.
//!
//! Keeps re-exports and [`RuntimeBuilder::new`] wiring in one place so facades stay thin.

pub use xlai_core as core;

pub use xlai_backend_openai::{
    OpenAiChatModel, OpenAiConfig, OpenAiTranscriptionModel, OpenAiTtsModel,
};
pub use xlai_backend_transformersjs::{
    TransformersJsBundle, TransformersJsChatModel, TransformersJsConfig,
};

#[cfg(feature = "llama")]
pub use xlai_backend_llama_cpp::{LlamaCppChatModel, LlamaCppConfig};

#[cfg(feature = "qts")]
pub use xlai_qts_core::{QtsTtsConfig, QtsTtsModel};

pub use xlai_runtime::{
    Agent, Chat, ChatExecutionEvent, DirectoryFileSystem, FileSystem, FsEntry, FsEntryKind, FsPath,
    McpRegistry, MemoryFileSystem, ReadableFileSystem, RuntimeBuilder, ToolCallExecutionMode,
    WritableFileSystem, XlaiRuntime,
};

#[cfg(not(target_arch = "wasm32"))]
pub use xlai_runtime::LocalFileSystem;

/// Starts a new [`RuntimeBuilder`] (same entrypoint for native and wasm facades).
#[must_use]
pub fn builder() -> RuntimeBuilder {
    RuntimeBuilder::new()
}
