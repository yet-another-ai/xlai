use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::future::Future;
use std::pin::Pin;

use async_stream::try_stream;
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

/// Structured metadata attached to runtime entities.
///
/// Message metadata is especially useful for local-only annotations in persisted
/// chat histories, such as marking entries as editable reminders or tracking
/// bookkeeping needed when replaying a session later.
pub type Metadata = BTreeMap<String, Value>;
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
    Transcription,
    Tts,
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

/// Vision / image detail hint for providers that support it (e.g. OpenAI).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    #[default]
    Auto,
    Low,
    High,
}

/// Binary or remote reference for multimodal parts.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum MediaSource {
    Url {
        url: String,
    },
    InlineData {
        mime_type: String,
        /// Base64-encoded bytes (no `data:` prefix).
        data_base64: String,
    },
}

/// One segment of a multimodal user/assistant/system message.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    Audio {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
    File {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

/// Message body as an ordered list of multimodal parts.
///
/// For JSON, a plain string still deserializes as a single [`ContentPart::Text`].
#[derive(Clone, Debug, PartialEq)]
pub struct ChatContent {
    pub parts: Vec<ContentPart>,
}

impl ChatContent {
    #[must_use]
    pub fn empty() -> Self {
        Self { parts: Vec::new() }
    }

    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            parts: vec![ContentPart::Text { text: text.into() }],
        }
    }

    #[must_use]
    pub fn from_parts(parts: Vec<ContentPart>) -> Self {
        Self { parts }
    }

    /// Concatenates all [`ContentPart::Text`] segments in order.
    #[must_use]
    pub fn text_parts_concatenated(&self) -> String {
        self.parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns the single text part if this is exactly one text block.
    #[must_use]
    pub fn as_single_text(&self) -> Option<&str> {
        match self.parts.as_slice() {
            [ContentPart::Text { text }] => Some(text.as_str()),
            _ => None,
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamTextDelta {
    /// Index into the assembled message [`ChatContent::parts`] for this stream.
    pub part_index: usize,
    pub delta: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
enum ChatContentSerde {
    Plain(String),
    WithParts {
        parts: Vec<ContentPart>,
    },
    /// JSON array of content parts, e.g. `[{"type":"text","text":"hi"}]`.
    PartsOnly(Vec<ContentPart>),
}

impl Serialize for ChatContent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if let Some(text) = self.as_single_text() {
            serializer.serialize_str(text)
        } else {
            ChatContentSerde::WithParts {
                parts: self.parts.clone(),
            }
            .serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for ChatContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match ChatContentSerde::deserialize(deserializer)? {
            ChatContentSerde::Plain(text) => Ok(Self::text(text)),
            ChatContentSerde::WithParts { parts } | ChatContentSerde::PartsOnly(parts) => {
                Ok(Self { parts })
            }
        }
    }
}

impl From<String> for ChatContent {
    fn from(value: String) -> Self {
        Self::text(value)
    }
}

impl From<&str> for ChatContent {
    fn from(value: &str) -> Self {
        Self::text(value)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: ChatContent,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    #[serde(default)]
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

/// How tool calls from a single model turn are executed relative to each other.
///
/// When any invoked tool in a turn is marked [`Sequential`](Self::Sequential), the
/// runtime runs **all** tool calls in that turn sequentially in model order (no overlap
/// with any other tool call in the same turn).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolCallExecutionMode {
    #[default]
    Concurrent,
    Sequential,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
    #[serde(default)]
    pub execution_mode: ToolCallExecutionMode,
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
    #[serde(default)]
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

/// Structured output request for providers that can honor it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StructuredOutputFormat {
    JsonSchema { schema: Value },
    LarkGrammar { grammar: String },
}

/// Structured output request for providers that can honor it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StructuredOutput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(flatten)]
    pub format: StructuredOutputFormat,
}

impl StructuredOutput {
    #[must_use]
    pub fn json_schema(schema: Value) -> Self {
        Self {
            name: None,
            description: None,
            format: StructuredOutputFormat::JsonSchema { schema },
        }
    }

    #[must_use]
    pub fn lark_grammar(grammar: impl Into<String>) -> Self {
        Self {
            name: None,
            description: None,
            format: StructuredOutputFormat::LarkGrammar {
                grammar: grammar.into(),
            },
        }
    }

    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub available_tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_output: Option<StructuredOutput>,
    #[serde(default)]
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
    #[serde(default)]
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
    MessageStart {
        role: MessageRole,
    },
    /// Incremental text for multimodal part at `part_index` (usually `0` for plain assistant streams).
    ContentDelta(StreamTextDelta),
    ToolCallDelta(ToolCallChunk),
    Finished(ChatResponse),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub inputs: Vec<String>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingResponse {
    pub vectors: Vec<Vec<f32>>,
    pub usage: Option<TokenUsage>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionRequest {
    pub model: Option<String>,
    pub audio: MediaSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionResponse {
    pub text: String,
    #[serde(default)]
    pub metadata: Metadata,
}

/// Output container format for text-to-speech (OpenAI-compatible `response_format` values).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TtsAudioFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

/// Whether the caller wants a single response body or a stream of audio chunks.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TtsDeliveryMode {
    /// One complete audio payload (e.g. HTTP body).
    #[default]
    Unary,
    /// Incremental audio (e.g. SSE deltas); semantics depend on the provider.
    Stream,
}

/// One reference sample for voice cloning / conditioning (provider-specific).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VoiceReferenceSample {
    pub audio: MediaSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    #[serde(default)]
    pub metadata: Metadata,
}

/// Provider-neutral voice selection.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum VoiceSpec {
    /// Built-in voice name (e.g. OpenAI `alloy`).
    Preset { name: String },
    /// Provider-stored or custom voice identifier.
    ProviderRef {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider: Option<String>,
    },
    /// Reference-audio-based cloning / conditioning (not all backends support this).
    Clone {
        references: Vec<VoiceReferenceSample>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TtsRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub input: String,
    pub voice: VoiceSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<TtsAudioFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default)]
    pub delivery: TtsDeliveryMode,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TtsResponse {
    pub audio: MediaSource,
    pub mime_type: String,
    #[serde(default)]
    pub metadata: Metadata,
}

/// One item from a streaming TTS response.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum TtsChunk {
    Started {
        mime_type: String,
        #[serde(default)]
        metadata: Metadata,
    },
    AudioDelta {
        data_base64: String,
    },
    Finished {
        response: TtsResponse,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Skill {
    pub id: SkillId,
    pub name: String,
    pub description: String,
    pub prompt_fragment: String,
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KnowledgeDocument {
    pub id: DocumentId,
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeChunk {
    pub id: ChunkId,
    pub document_id: DocumentId,
    pub content: String,
    #[serde(default)]
    pub metadata: Metadata,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KnowledgeQuery {
    pub text: String,
    pub limit: usize,
    #[serde(default)]
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
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VectorSearchQuery {
    pub namespace: String,
    pub vector: Vec<f32>,
    pub limit: usize,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VectorSearchHit {
    pub id: ChunkId,
    pub score: f32,
    #[serde(default)]
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

pub trait ChatBackend {
    type Model: ChatModel + 'static;

    fn into_chat_model(self) -> Self::Model;
}

impl<T> ChatBackend for T
where
    T: ChatModel + 'static,
{
    type Model = T;

    fn into_chat_model(self) -> Self::Model {
        self
    }
}

pub trait TranscriptionModel: RuntimeBound {
    fn provider_name(&self) -> &'static str;

    fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> BoxFuture<'_, Result<TranscriptionResponse, XlaiError>>;
}

pub trait TranscriptionBackend {
    type Model: TranscriptionModel + 'static;

    fn into_transcription_model(self) -> Self::Model;
}

impl<T> TranscriptionBackend for T
where
    T: TranscriptionModel + 'static,
{
    type Model = T;

    fn into_transcription_model(self) -> Self::Model {
        self
    }
}

pub trait TtsModel: RuntimeBound {
    fn provider_name(&self) -> &'static str;

    fn synthesize(&self, request: TtsRequest) -> BoxFuture<'_, Result<TtsResponse, XlaiError>>;

    /// Streaming synthesis. Default implementation calls [`Self::synthesize`] once and emits
    /// [`TtsChunk::Started`], one [`TtsChunk::AudioDelta`], then [`TtsChunk::Finished`].
    fn synthesize_stream(&self, request: TtsRequest) -> BoxStream<'_, Result<TtsChunk, XlaiError>> {
        Box::pin(try_stream! {
            let response = self.synthesize(request).await?;
            yield TtsChunk::Started {
                mime_type: response.mime_type.clone(),
                metadata: BTreeMap::new(),
            };
            let data_base64 = match &response.audio {
                MediaSource::InlineData { data_base64, .. } => data_base64.clone(),
                MediaSource::Url { .. } => Err(XlaiError::new(
                    ErrorKind::Unsupported,
                    "default TTS stream fallback requires inline audio bytes",
                ))?,
            };
            yield TtsChunk::AudioDelta { data_base64 };
            yield TtsChunk::Finished { response };
        })
    }
}

pub trait TtsBackend {
    type Model: TtsModel + 'static;

    fn into_tts_model(self) -> Self::Model;
}

impl<T> TtsBackend for T
where
    T: TtsModel + 'static,
{
    type Model = T;

    fn into_tts_model(self) -> Self::Model {
        self
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        ChatContent, ChatMessage, ContentPart, MediaSource, MessageRole, StructuredOutput,
        StructuredOutputFormat,
    };

    #[test]
    fn chat_message_deserializes_missing_metadata_as_empty_map() {
        let result: Result<ChatMessage, _> = serde_json::from_value(json!({
            "role": "System",
            "content": "Preserve this reminder.",
            "tool_name": null,
            "tool_call_id": null
        }));
        assert!(
            result.is_ok(),
            "chat message without metadata should deserialize"
        );
        let Ok(message) = result else {
            return;
        };

        assert_eq!(message.role, MessageRole::System);
        assert!(message.metadata.is_empty());
    }

    #[test]
    fn chat_message_supports_structured_metadata_values() {
        let message = ChatMessage {
            role: MessageRole::System,
            content: ChatContent::text("Preserve this reminder."),
            tool_name: None,
            tool_call_id: None,
            metadata: [(
                "reminder".to_owned(),
                json!({
                    "kind": "system_reminder",
                    "editable": true,
                    "tags": ["session", "mutable"]
                }),
            )]
            .into_iter()
            .collect(),
        };

        assert_eq!(
            message.metadata.get("reminder"),
            Some(&json!({
                "kind": "system_reminder",
                "editable": true,
                "tags": ["session", "mutable"]
            }))
        );
    }

    #[test]
    fn chat_content_serializes_single_text_as_plain_string() {
        let c = ChatContent::text("hello");
        let result = serde_json::to_value(&c);
        assert!(result.is_ok(), "serialize");
        let Ok(v) = result else {
            return;
        };
        assert_eq!(v, json!("hello"));
    }

    #[test]
    fn chat_content_round_trips_multimodal_parts() {
        let c = ChatContent::from_parts(vec![
            ContentPart::Text {
                text: "Describe:".to_owned(),
            },
            ContentPart::Image {
                source: MediaSource::Url {
                    url: "https://example.com/a.png".to_owned(),
                },
                mime_type: Some("image/png".to_owned()),
                detail: None,
            },
        ]);
        let serialized = serde_json::to_value(&c);
        assert!(serialized.is_ok(), "serialize");
        let Ok(v) = serialized else {
            return;
        };
        let deserialized: Result<ChatContent, _> = serde_json::from_value(v);
        assert!(deserialized.is_ok(), "deserialize");
        let Ok(back) = deserialized else {
            return;
        };
        assert_eq!(back, c);
    }

    #[test]
    fn chat_content_round_trips_audio_parts() {
        let c = ChatContent::from_parts(vec![ContentPart::Audio {
            source: MediaSource::InlineData {
                mime_type: "audio/wav".to_owned(),
                data_base64: "UklGRg==".to_owned(),
            },
            mime_type: Some("audio/wav".to_owned()),
        }]);
        let serialized = serde_json::to_value(&c);
        assert!(serialized.is_ok(), "serialize");
        let Ok(v) = serialized else {
            return;
        };
        let deserialized: Result<ChatContent, _> = serde_json::from_value(v);
        assert!(deserialized.is_ok(), "deserialize");
        let Ok(back) = deserialized else {
            return;
        };
        assert_eq!(back, c);
    }

    #[test]
    fn structured_output_round_trips_json_schema_format() {
        let output = StructuredOutput::json_schema(json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"],
            "additionalProperties": false
        }))
        .with_name("person")
        .with_description("A person object");

        let serialized = serde_json::to_value(&output);
        assert!(serialized.is_ok(), "serialize");
        let Ok(v) = serialized else {
            return;
        };
        assert_eq!(v["type"], json!("json_schema"));
        assert_eq!(v["schema"]["type"], json!("object"));

        let deserialized: Result<StructuredOutput, _> = serde_json::from_value(v);
        assert!(deserialized.is_ok(), "deserialize");
        let Ok(back) = deserialized else {
            return;
        };
        assert_eq!(back, output);
    }

    #[test]
    fn structured_output_round_trips_lark_grammar_format() {
        let output = StructuredOutput::lark_grammar("start: NAME\nNAME: /[a-z]+/")
            .with_name("record")
            .with_description("A simple lark grammar");

        let serialized = serde_json::to_value(&output);
        assert!(serialized.is_ok(), "serialize");
        let Ok(v) = serialized else {
            return;
        };
        assert_eq!(v["type"], json!("lark_grammar"));
        assert_eq!(v["grammar"], json!("start: NAME\nNAME: /[a-z]+/"));

        let deserialized: Result<StructuredOutput, _> = serde_json::from_value(v);
        assert!(deserialized.is_ok(), "deserialize");
        let Ok(back) = deserialized else {
            return;
        };
        assert_eq!(back, output);
        assert!(matches!(
            back.format,
            StructuredOutputFormat::LarkGrammar { .. }
        ));
    }
}
