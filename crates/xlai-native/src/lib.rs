pub use xlai_backend_openai::{OpenAiChatModel, OpenAiConfig};
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
