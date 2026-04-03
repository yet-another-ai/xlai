pub use xlai_backend_llama_cpp::{LlamaCppChatModel, LlamaCppConfig};
pub use xlai_backend_openai::{
    OpenAiChatModel, OpenAiConfig, OpenAiTranscriptionModel, OpenAiTtsModel,
};
pub use xlai_backend_transformersjs::{
    TransformersJsBundle, TransformersJsChatModel, TransformersJsConfig,
};
#[cfg(feature = "qts")]
pub use xlai_backend_qts::{QtsTtsConfig, QtsTtsModel};
pub use xlai_core as core;
pub use xlai_runtime::{
    Agent, Chat, ChatExecutionEvent, DirectoryFileSystem, FileSystem, FsEntry, FsEntryKind, FsPath,
    LocalFileSystem, McpRegistry, MemoryFileSystem, ReadableFileSystem, RuntimeBuilder,
    ToolCallExecutionMode, WritableFileSystem, XlaiRuntime,
};

#[must_use]
pub fn builder() -> RuntimeBuilder {
    RuntimeBuilder::new()
}
