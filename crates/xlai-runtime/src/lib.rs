mod agent;
mod chat;
mod fs;
pub mod local_common;
mod prompt;
mod skill_store;

use std::sync::Arc;

use async_stream::try_stream;
use futures_util::StreamExt;
use xlai_core::{
    BoxStream, ChatChunk, ChatModel, ChatRequest, ChatResponse, EmbeddingModel, EmbeddingRequest,
    EmbeddingResponse, ErrorKind, KnowledgeHit, KnowledgeQuery, KnowledgeStore, RuntimeCapability,
    Skill, SkillId, SkillStore, ToolCall, ToolExecutor, ToolResult, TranscriptionModel, TtsModel,
    VectorSearchHit, VectorSearchQuery, VectorStore, XlaiError,
};

pub use agent::{Agent, McpRegistry};
pub use chat::{Chat, ChatExecutionEvent};
#[cfg(not(target_arch = "wasm32"))]
pub use fs::LocalFileSystem;
pub use fs::{MemoryFileSystem, boxed_file_system};
pub use prompt::EmbeddedPromptStore;
pub use skill_store::MarkdownSkillStore;
pub use tera::Context as PromptContext;
pub use xlai_core::{
    ChatBackend, ChatContent, ChatRetryPolicy, ContentPart, DirectoryFileSystem, FileSystem,
    FsEntry, FsEntryKind, FsPath, ImageDetail, MediaSource, ReadableFileSystem, StreamTextDelta,
    ToolCallExecutionMode, TranscriptionBackend, TranscriptionRequest, TranscriptionResponse,
    TtsAudioFormat, TtsBackend, TtsChunk, TtsDeliveryMode, TtsRequest, TtsResponse,
    VoiceReferenceSample, VoiceSpec, WritableFileSystem,
};

#[derive(Clone, Default)]
pub struct RuntimeBuilder {
    chat_model: Option<Arc<dyn ChatModel>>,
    embedding_model: Option<Arc<dyn EmbeddingModel>>,
    transcription_model: Option<Arc<dyn TranscriptionModel>>,
    tts_model: Option<Arc<dyn TtsModel>>,
    tool_executor: Option<Arc<dyn ToolExecutor>>,
    skill_store: Option<Arc<dyn SkillStore>>,
    knowledge_store: Option<Arc<dyn KnowledgeStore>>,
    vector_store: Option<Arc<dyn VectorStore>>,
    file_system: Option<Arc<dyn FileSystem>>,
    capabilities: Vec<RuntimeCapability>,
}

impl RuntimeBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_chat_model(mut self, chat_model: Arc<dyn ChatModel>) -> Self {
        self.chat_model = Some(chat_model);
        self.capabilities.push(RuntimeCapability::Chat);
        self
    }

    #[must_use]
    pub fn with_chat_backend<B>(self, backend: B) -> Self
    where
        B: xlai_core::ChatBackend,
    {
        self.with_chat_model(Arc::new(backend.into_chat_model()))
    }

    #[must_use]
    pub fn with_embedding_model(mut self, embedding_model: Arc<dyn EmbeddingModel>) -> Self {
        self.embedding_model = Some(embedding_model);
        self.capabilities.push(RuntimeCapability::Embeddings);
        self
    }

    #[must_use]
    pub fn with_transcription_model(
        mut self,
        transcription_model: Arc<dyn TranscriptionModel>,
    ) -> Self {
        self.transcription_model = Some(transcription_model);
        self.capabilities.push(RuntimeCapability::Transcription);
        self
    }

    #[must_use]
    pub fn with_transcription_backend<B>(self, backend: B) -> Self
    where
        B: xlai_core::TranscriptionBackend,
    {
        self.with_transcription_model(Arc::new(backend.into_transcription_model()))
    }

    #[must_use]
    pub fn with_tts_model(mut self, tts_model: Arc<dyn TtsModel>) -> Self {
        self.tts_model = Some(tts_model);
        self.capabilities.push(RuntimeCapability::Tts);
        self
    }

    #[must_use]
    pub fn with_tts_backend<B>(self, backend: B) -> Self
    where
        B: xlai_core::TtsBackend,
    {
        self.with_tts_model(Arc::new(backend.into_tts_model()))
    }

    #[must_use]
    pub fn with_tool_executor(mut self, tool_executor: Arc<dyn ToolExecutor>) -> Self {
        self.tool_executor = Some(tool_executor);
        self.capabilities.push(RuntimeCapability::ToolCalling);
        self
    }

    #[must_use]
    pub fn with_skill_store(mut self, skill_store: Arc<dyn SkillStore>) -> Self {
        self.skill_store = Some(skill_store);
        self.capabilities.push(RuntimeCapability::SkillResolution);
        self
    }

    #[must_use]
    pub fn with_knowledge_store(mut self, knowledge_store: Arc<dyn KnowledgeStore>) -> Self {
        self.knowledge_store = Some(knowledge_store);
        self.capabilities.push(RuntimeCapability::KnowledgeSearch);
        self
    }

    #[must_use]
    pub fn with_vector_store(mut self, vector_store: Arc<dyn VectorStore>) -> Self {
        self.vector_store = Some(vector_store);
        self.capabilities.push(RuntimeCapability::VectorSearch);
        self
    }

    #[must_use]
    pub fn with_file_system(mut self, file_system: Arc<dyn FileSystem>) -> Self {
        self.file_system = Some(file_system);
        self.capabilities.push(RuntimeCapability::FileSystem);
        self
    }

    /// Builds an `XlaiRuntime` from the configured capabilities.
    ///
    /// # Errors
    ///
    /// Returns an error if no runtime capabilities were configured.
    pub fn build(self) -> Result<XlaiRuntime, XlaiError> {
        let runtime = XlaiRuntime {
            chat_model: self.chat_model,
            embedding_model: self.embedding_model,
            transcription_model: self.transcription_model,
            tts_model: self.tts_model,
            tool_executor: self.tool_executor,
            skill_store: self.skill_store,
            knowledge_store: self.knowledge_store,
            vector_store: self.vector_store,
            file_system: self.file_system,
            capabilities: dedup_capabilities(self.capabilities),
        };

        if runtime.capabilities.is_empty() {
            return Err(XlaiError::new(
                ErrorKind::Configuration,
                "runtime requires at least one configured capability",
            ));
        }

        Ok(runtime)
    }
}

#[derive(Clone)]
pub struct XlaiRuntime {
    chat_model: Option<Arc<dyn ChatModel>>,
    embedding_model: Option<Arc<dyn EmbeddingModel>>,
    transcription_model: Option<Arc<dyn TranscriptionModel>>,
    tts_model: Option<Arc<dyn TtsModel>>,
    tool_executor: Option<Arc<dyn ToolExecutor>>,
    skill_store: Option<Arc<dyn SkillStore>>,
    knowledge_store: Option<Arc<dyn KnowledgeStore>>,
    vector_store: Option<Arc<dyn VectorStore>>,
    file_system: Option<Arc<dyn FileSystem>>,
    capabilities: Vec<RuntimeCapability>,
}

impl XlaiRuntime {
    #[must_use]
    pub fn chat_session(&self) -> Chat {
        Chat::new(Arc::new(self.clone()))
    }

    /// Creates a high-level agent session with built-in tools enabled for the
    /// runtime's configured capabilities.
    ///
    /// # Errors
    ///
    /// Returns an error if the agent cannot initialize one of its built-in
    /// tools.
    pub fn agent_session(&self) -> Result<Agent, XlaiError> {
        Agent::new(Arc::new(self.clone()))
    }

    #[must_use]
    pub fn capabilities(&self) -> &[RuntimeCapability] {
        &self.capabilities
    }

    #[must_use]
    pub fn has_capability(&self, capability: RuntimeCapability) -> bool {
        self.capabilities.contains(&capability)
    }

    /// Executes a single chat request with the configured chat model.
    ///
    /// # Errors
    ///
    /// Returns an error if no chat model is configured or if the underlying
    /// model request fails.
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, XlaiError> {
        let chat_model = self
            .chat_model
            .as_ref()
            .ok_or_else(|| missing_dependency("chat model"))?;

        chat_model.generate(request).await
    }

    /// Starts a streaming chat request with the configured chat model.
    ///
    /// # Errors
    ///
    /// Returns an error if no chat model is configured.
    pub fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, Result<ChatChunk, XlaiError>>, XlaiError> {
        let chat_model = self
            .chat_model
            .clone()
            .ok_or_else(|| missing_dependency("chat model"))?;

        Ok(Box::pin(try_stream! {
            let mut stream = chat_model.generate_stream(request);
            while let Some(chunk) = stream.next().await {
                yield chunk?;
            }
        }))
    }

    /// Executes an embedding request with the configured embedding model.
    ///
    /// # Errors
    ///
    /// Returns an error if no embedding model is configured or if the
    /// underlying provider request fails.
    pub async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, XlaiError> {
        let embedding_model = self
            .embedding_model
            .as_ref()
            .ok_or_else(|| missing_dependency("embedding model"))?;

        embedding_model.embed(request).await
    }

    /// Executes an audio transcription request with the configured transcription model.
    ///
    /// # Errors
    ///
    /// Returns an error if no transcription model is configured or if the
    /// underlying provider request fails.
    pub async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse, XlaiError> {
        let transcription_model = self
            .transcription_model
            .as_ref()
            .ok_or_else(|| missing_dependency("transcription model"))?;

        transcription_model.transcribe(request).await
    }

    /// Synthesizes speech from text using the configured TTS model.
    ///
    /// # Errors
    ///
    /// Returns an error if no TTS model is configured or if the provider request fails.
    pub async fn synthesize(&self, request: TtsRequest) -> Result<TtsResponse, XlaiError> {
        let tts_model = self
            .tts_model
            .as_ref()
            .ok_or_else(|| missing_dependency("tts model"))?;

        tts_model.synthesize(request).await
    }

    /// Streams synthesized speech using the configured TTS model.
    ///
    /// # Errors
    ///
    /// Returns an error if no TTS model is configured.
    pub fn stream_synthesize(
        &self,
        request: TtsRequest,
    ) -> Result<BoxStream<'static, Result<TtsChunk, XlaiError>>, XlaiError> {
        let tts_model = self
            .tts_model
            .clone()
            .ok_or_else(|| missing_dependency("tts model"))?;

        Ok(Box::pin(try_stream! {
            let mut stream = tts_model.synthesize_stream(request);
            while let Some(chunk) = stream.next().await {
                yield chunk?;
            }
        }))
    }

    /// Executes a tool call through the configured runtime tool executor.
    ///
    /// # Errors
    ///
    /// Returns an error if no tool executor is configured or if the tool call
    /// fails.
    pub async fn call_tool(&self, call: ToolCall) -> Result<ToolResult, XlaiError> {
        let tool_executor = self
            .tool_executor
            .as_ref()
            .ok_or_else(|| missing_dependency("tool executor"))?;

        tool_executor.call_tool(call).await
    }

    /// Resolves skills through the configured skill store.
    ///
    /// # Errors
    ///
    /// Returns an error if no skill store is configured or if skill resolution
    /// fails.
    pub async fn resolve_skills(&self, ids: &[SkillId]) -> Result<Vec<Skill>, XlaiError> {
        let skill_store = self
            .skill_store
            .as_ref()
            .ok_or_else(|| missing_dependency("skill store"))?;

        skill_store.resolve_skills(ids).await
    }

    /// Searches the configured knowledge store.
    ///
    /// # Errors
    ///
    /// Returns an error if no knowledge store is configured or if the search
    /// fails.
    pub async fn search_knowledge(
        &self,
        query: KnowledgeQuery,
    ) -> Result<Vec<KnowledgeHit>, XlaiError> {
        let knowledge_store = self
            .knowledge_store
            .as_ref()
            .ok_or_else(|| missing_dependency("knowledge store"))?;

        knowledge_store.search(query).await
    }

    /// Searches the configured vector store.
    ///
    /// # Errors
    ///
    /// Returns an error if no vector store is configured or if the search
    /// fails.
    pub async fn search_vectors(
        &self,
        query: VectorSearchQuery,
    ) -> Result<Vec<VectorSearchHit>, XlaiError> {
        let vector_store = self
            .vector_store
            .as_ref()
            .ok_or_else(|| missing_dependency("vector store"))?;

        vector_store.search(query).await
    }

    /// Reads raw bytes from the configured filesystem.
    ///
    /// # Errors
    ///
    /// Returns an error if no filesystem is configured or if the file read
    /// fails.
    pub async fn read_file(&self, path: &FsPath) -> Result<Vec<u8>, XlaiError> {
        let file_system = self
            .file_system
            .as_ref()
            .ok_or_else(|| missing_dependency("file system"))?;

        file_system.read(path).await
    }

    /// Checks whether the configured filesystem contains the provided path.
    ///
    /// # Errors
    ///
    /// Returns an error if no filesystem is configured or if the existence
    /// check fails.
    pub async fn path_exists(&self, path: &FsPath) -> Result<bool, XlaiError> {
        let file_system = self
            .file_system
            .as_ref()
            .ok_or_else(|| missing_dependency("file system"))?;

        file_system.exists(path).await
    }

    /// Writes raw bytes to the configured filesystem.
    ///
    /// # Errors
    ///
    /// Returns an error if no filesystem is configured or if the file write
    /// fails.
    pub async fn write_file(&self, path: &FsPath, data: Vec<u8>) -> Result<(), XlaiError> {
        let file_system = self
            .file_system
            .as_ref()
            .ok_or_else(|| missing_dependency("file system"))?;

        file_system.write(path, data).await
    }

    /// Creates a directory and its missing parents in the configured filesystem.
    ///
    /// # Errors
    ///
    /// Returns an error if no filesystem is configured or if the directory
    /// creation fails.
    pub async fn create_dir_all(&self, path: &FsPath) -> Result<(), XlaiError> {
        let file_system = self
            .file_system
            .as_ref()
            .ok_or_else(|| missing_dependency("file system"))?;

        file_system.create_dir_all(path).await
    }

    /// Lists direct children under a directory in the configured filesystem.
    ///
    /// # Errors
    ///
    /// Returns an error if no filesystem is configured or if the directory list
    /// operation fails.
    pub async fn list_directory(&self, path: &FsPath) -> Result<Vec<FsEntry>, XlaiError> {
        let file_system = self
            .file_system
            .as_ref()
            .ok_or_else(|| missing_dependency("file system"))?;

        file_system.list(path).await
    }

    /// Deletes a file or directory tree from the configured filesystem.
    ///
    /// # Errors
    ///
    /// Returns an error if no filesystem is configured or if deletion fails.
    pub async fn delete_path(&self, path: &FsPath) -> Result<(), XlaiError> {
        let file_system = self
            .file_system
            .as_ref()
            .ok_or_else(|| missing_dependency("file system"))?;

        file_system.delete(path).await
    }
}

fn dedup_capabilities(capabilities: Vec<RuntimeCapability>) -> Vec<RuntimeCapability> {
    let mut deduped = Vec::new();

    for capability in capabilities {
        if !deduped.contains(&capability) {
            deduped.push(capability);
        }
    }

    deduped
}

fn missing_dependency(name: &str) -> XlaiError {
    XlaiError::new(
        ErrorKind::Unsupported,
        format!("runtime was built without a configured {name}"),
    )
}

#[cfg(test)]
mod tests;
