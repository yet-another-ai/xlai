use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::future::Future;
use std::pin::Pin;

use futures_core::Stream;
use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(not(target_arch = "wasm32"))]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
#[cfg(target_arch = "wasm32")]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[cfg(not(target_arch = "wasm32"))]
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;
#[cfg(target_arch = "wasm32")]
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + 'a>>;

pub type Metadata = BTreeMap<String, String>;
pub type SkillId = String;
pub type DocumentId = String;
pub type ChunkId = String;

#[cfg(not(target_arch = "wasm32"))]
pub trait RuntimeBound: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T> RuntimeBound for T where T: Send + Sync + ?Sized {}

#[cfg(target_arch = "wasm32")]
pub trait RuntimeBound {}
#[cfg(target_arch = "wasm32")]
impl<T> RuntimeBound for T where T: ?Sized {}

#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_arch = "wasm32"))]
impl<T> MaybeSend for T where T: Send + ?Sized {}

#[cfg(target_arch = "wasm32")]
pub trait MaybeSend {}
#[cfg(target_arch = "wasm32")]
impl<T> MaybeSend for T where T: ?Sized {}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RuntimeCapability {
    Chat,
    Embeddings,
    ToolCalling,
    SkillResolution,
    KnowledgeSearch,
    VectorSearch,
    FileSystem,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ErrorKind {
    Configuration,
    Validation,
    Provider,
    Tool,
    Skill,
    Knowledge,
    Vector,
    FileSystem,
    Unsupported,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct XlaiError {
    pub kind: ErrorKind,
    pub message: String,
}

impl XlaiError {
    #[must_use]
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl Display for XlaiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl Error for XlaiError {}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    pub metadata: Metadata,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolParameterType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolParameter {
    pub name: String,
    pub description: String,
    pub kind: ToolParameterType,
    pub required: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub tool_name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub tool_name: String,
    pub content: String,
    pub is_error: bool,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FinishReason {
    Completed,
    ToolCalls,
    Length,
    Stopped,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub available_tools: Vec<ToolDefinition>,
    pub skill_ids: Vec<SkillId>,
    pub metadata: Metadata,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatResponse {
    pub message: ChatMessage,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<TokenUsage>,
    pub finish_reason: FinishReason,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallChunk {
    pub index: usize,
    pub id: Option<String>,
    pub tool_name: Option<String>,
    pub arguments_delta: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ChatChunk {
    MessageStart { role: MessageRole },
    ContentDelta(String),
    ToolCallDelta(ToolCallChunk),
    Finished(ChatResponse),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub inputs: Vec<String>,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingResponse {
    pub vectors: Vec<Vec<f32>>,
    pub usage: Option<TokenUsage>,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Skill {
    pub id: SkillId,
    pub name: String,
    pub description: String,
    pub prompt_fragment: String,
    pub tags: Vec<String>,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KnowledgeDocument {
    pub id: DocumentId,
    pub title: String,
    pub content: String,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeChunk {
    pub id: ChunkId,
    pub document_id: DocumentId,
    pub content: String,
    pub metadata: Metadata,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KnowledgeQuery {
    pub text: String,
    pub limit: usize,
    pub skill_ids: Vec<SkillId>,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeHit {
    pub chunk: KnowledgeChunk,
    pub score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingRecord {
    pub id: ChunkId,
    pub vector: Vec<f32>,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VectorSearchQuery {
    pub namespace: String,
    pub vector: Vec<f32>,
    pub limit: usize,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VectorSearchHit {
    pub id: ChunkId,
    pub score: f32,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FsPath(String);

impl FsPath {
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl Display for FsPath {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<String> for FsPath {
    fn from(path: String) -> Self {
        Self(path)
    }
}

impl From<&str> for FsPath {
    fn from(path: &str) -> Self {
        Self(path.to_owned())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FsEntryKind {
    File,
    Directory,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FsEntry {
    pub path: FsPath,
    pub kind: FsEntryKind,
}

pub trait ChatModel: RuntimeBound {
    fn provider_name(&self) -> &'static str;

    fn generate(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, XlaiError>>;

    fn generate_stream(&self, request: ChatRequest) -> BoxStream<'_, Result<ChatChunk, XlaiError>> {
        Box::pin(stream::once(async move {
            self.generate(request).await.map(ChatChunk::Finished)
        }))
    }
}

pub trait EmbeddingModel: RuntimeBound {
    fn provider_name(&self) -> &'static str;

    fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> BoxFuture<'_, Result<EmbeddingResponse, XlaiError>>;
}

pub trait ToolExecutor: RuntimeBound {
    fn call_tool(&self, call: ToolCall) -> BoxFuture<'_, Result<ToolResult, XlaiError>>;
}

pub trait SkillStore: SkillFileSystem {
    fn resolve_skills<'a>(
        &'a self,
        ids: &'a [SkillId],
    ) -> BoxFuture<'a, Result<Vec<Skill>, XlaiError>>;
}

pub trait KnowledgeStore: RuntimeBound {
    fn upsert_documents(
        &self,
        documents: Vec<KnowledgeDocument>,
    ) -> BoxFuture<'_, Result<(), XlaiError>>;

    fn search(&self, query: KnowledgeQuery) -> BoxFuture<'_, Result<Vec<KnowledgeHit>, XlaiError>>;
}

pub trait VectorStore: RuntimeBound {
    fn upsert<'a>(
        &'a self,
        namespace: &'a str,
        records: Vec<EmbeddingRecord>,
    ) -> BoxFuture<'a, Result<(), XlaiError>>;

    fn search(
        &self,
        query: VectorSearchQuery,
    ) -> BoxFuture<'_, Result<Vec<VectorSearchHit>, XlaiError>>;
}

pub trait ReadableFileSystem: RuntimeBound {
    fn read<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<Vec<u8>, XlaiError>>;

    fn exists<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<bool, XlaiError>>;
}

pub trait WritableFileSystem: RuntimeBound {
    fn write<'a>(&'a self, path: &'a FsPath, data: Vec<u8>)
    -> BoxFuture<'a, Result<(), XlaiError>>;

    fn delete<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<(), XlaiError>>;

    fn create_dir_all<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<(), XlaiError>>;
}

pub trait DirectoryFileSystem: RuntimeBound {
    fn list<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<Vec<FsEntry>, XlaiError>>;
}

pub trait SkillFileSystem: ReadableFileSystem + DirectoryFileSystem {}

impl<T> SkillFileSystem for T where T: ReadableFileSystem + DirectoryFileSystem + ?Sized {}

pub trait FileSystem: ReadableFileSystem + WritableFileSystem + DirectoryFileSystem {}

impl<T> FileSystem for T where
    T: ReadableFileSystem + WritableFileSystem + DirectoryFileSystem + ?Sized
{
}
