mod agent;
mod chat;
mod fs;
mod prompt;
mod skill_store;

use std::sync::Arc;

use async_stream::try_stream;
use futures_util::StreamExt;
use xlai_core::{
    BoxStream, ChatChunk, ChatModel, ChatRequest, ChatResponse, EmbeddingModel, EmbeddingRequest,
    EmbeddingResponse, ErrorKind, KnowledgeHit, KnowledgeQuery, KnowledgeStore, RuntimeCapability,
    Skill, SkillId, SkillStore, ToolCall, ToolExecutor, ToolResult, VectorSearchHit,
    VectorSearchQuery, VectorStore, XlaiError,
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
    ChatBackend, DirectoryFileSystem, FileSystem, FsEntry, FsEntryKind, FsPath, ReadableFileSystem,
    ToolCallExecutionMode, WritableFileSystem,
};

#[derive(Clone, Default)]
pub struct RuntimeBuilder {
    chat_model: Option<Arc<dyn ChatModel>>,
    embedding_model: Option<Arc<dyn EmbeddingModel>>,
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
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, MutexGuard};

    use futures_util::{StreamExt, stream};
    use serde_json::json;
    use tokio::time::{Duration, sleep};
    use xlai_core::{
        BoxFuture, BoxStream, ChatChunk, ChatMessage, ChatModel, ChatRequest, ChatResponse,
        FinishReason, FsEntryKind, FsPath, MessageRole, RuntimeCapability, Skill, SkillStore,
        ToolCall, ToolCallExecutionMode, ToolDefinition, ToolParameter, ToolParameterType,
        WritableFileSystem, XlaiError,
    };

    use super::{
        ChatExecutionEvent, EmbeddedPromptStore, MarkdownSkillStore, MemoryFileSystem,
        PromptContext, RuntimeBuilder,
    };

    fn empty_metadata() -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::new()
    }

    fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        match mutex.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn chat_executes_registered_tools_across_round_trips() -> Result<(), XlaiError> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(RecordingChatModel::new(
            requests.clone(),
            vec![
                ChatResponse {
                    message: assistant_message(""),
                    tool_calls: vec![ToolCall {
                        id: "tool_1".to_owned(),
                        tool_name: "lookup_weather".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    }],
                    usage: None,
                    finish_reason: FinishReason::ToolCalls,
                    metadata: empty_metadata(),
                },
                ChatResponse {
                    message: assistant_message("Paris is sunny."),
                    tool_calls: Vec::new(),
                    usage: None,
                    finish_reason: FinishReason::Completed,
                    metadata: empty_metadata(),
                },
            ],
        ));

        let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;

        let mut chat = runtime.chat_session();
        chat.register_tool(weather_tool_definition(), |arguments| async move {
            Ok(xlai_core::ToolResult {
                tool_name: "ignored_by_runtime".to_owned(),
                content: format!(
                    "weather for {}: sunny",
                    arguments["city"].as_str().unwrap_or("unknown")
                ),
                is_error: false,
                metadata: empty_metadata(),
            })
        });

        let response = chat.prompt("What's the weather in Paris?").await?;

        assert_eq!(response.message.content, "Paris is sunny.");

        let requests = lock_unpoisoned(&requests);
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].available_tools.len(), 1);
        assert_eq!(requests[1].messages.len(), 3);
        assert_eq!(requests[1].messages[2].role, MessageRole::Tool);
        assert_eq!(
            requests[1].messages[2].tool_name.as_deref(),
            Some("lookup_weather")
        );
        assert_eq!(
            requests[1].messages[2].tool_call_id.as_deref(),
            Some("tool_1")
        );
        assert_eq!(requests[1].messages[2].content, "weather for Paris: sunny");

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn agent_session_injects_skill_tool_when_skill_store_is_configured()
    -> Result<(), XlaiError> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(RecordingChatModel::new(
            requests.clone(),
            vec![ChatResponse {
                message: assistant_message("agent reply"),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            }],
        ));

        let runtime = RuntimeBuilder::new()
            .with_chat_model(model)
            .with_skill_store(seed_markdown_skill_store().await?)
            .build()?;

        let agent = runtime.agent_session()?;
        let response = agent.prompt("Hello").await?;
        assert_eq!(response.message.content, "agent reply");

        let requests = lock_unpoisoned(&requests);
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0]
                .available_tools
                .iter()
                .any(|tool| tool.name == "skill"),
            "expected agent sessions to expose the built-in skill tool"
        );

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn agent_skill_tool_uses_configured_skill_store() -> Result<(), XlaiError> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(RecordingChatModel::new(
            requests.clone(),
            vec![
                ChatResponse {
                    message: assistant_message(""),
                    tool_calls: vec![ToolCall {
                        id: "skill_1".to_owned(),
                        tool_name: "skill".to_owned(),
                        arguments: json!({
                            "skill": "review.code",
                            "args": "Focus on correctness."
                        }),
                    }],
                    usage: None,
                    finish_reason: FinishReason::ToolCalls,
                    metadata: empty_metadata(),
                },
                ChatResponse {
                    message: assistant_message("skill loaded"),
                    tool_calls: Vec::new(),
                    usage: None,
                    finish_reason: FinishReason::Completed,
                    metadata: empty_metadata(),
                },
            ],
        ));

        let runtime = RuntimeBuilder::new()
            .with_chat_model(model)
            .with_skill_store(seed_markdown_skill_store().await?)
            .build()?;

        let agent = runtime.agent_session()?;
        let response = agent.prompt("Review this patch").await?;
        assert_eq!(response.message.content, "skill loaded");

        let requests = lock_unpoisoned(&requests);
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].messages.len(), 3);
        assert_eq!(requests[1].messages[2].tool_name.as_deref(), Some("skill"));
        assert!(
            requests[1].messages[2]
                .content
                .contains("Prioritize bugs, regressions, and missing tests."),
            "expected the skill tool result to include the resolved prompt fragment"
        );
        assert!(
            requests[1].messages[2]
                .content
                .contains("Focus on correctness."),
            "expected the skill tool result to include the supplied skill arguments"
        );

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn agent_mcp_registry_exposes_registered_tools_alongside_builtins()
    -> Result<(), XlaiError> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(RecordingChatModel::new(
            requests.clone(),
            vec![ChatResponse {
                message: assistant_message("agent reply"),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            }],
        ));

        let runtime = RuntimeBuilder::new()
            .with_chat_model(model)
            .with_skill_store(seed_markdown_skill_store().await?)
            .build()?;

        let mut agent = runtime.agent_session()?;
        agent
            .mcp_registry()
            .register_tool(weather_tool_definition(), |_| async {
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_weather".to_owned(),
                    content: "weather for Paris: sunny".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            });

        let mcp_tools = agent.mcp_registry().tool_definitions();
        assert_eq!(mcp_tools.len(), 1);
        assert_eq!(mcp_tools[0].name, "lookup_weather");

        let response = agent.prompt("Hello").await?;
        assert_eq!(response.message.content, "agent reply");

        let requests = lock_unpoisoned(&requests);
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0]
                .available_tools
                .iter()
                .any(|tool| tool.name == "skill"),
            "expected agent sessions to keep exposing built-in tools"
        );
        assert!(
            requests[0]
                .available_tools
                .iter()
                .any(|tool| tool.name == "lookup_weather"),
            "expected agent sessions to expose MCP-registered tools"
        );

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn agent_mcp_registry_executes_registered_tool_calls() -> Result<(), XlaiError> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(RecordingChatModel::new(
            requests.clone(),
            vec![
                ChatResponse {
                    message: assistant_message(""),
                    tool_calls: vec![ToolCall {
                        id: "tool_1".to_owned(),
                        tool_name: "lookup_weather".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    }],
                    usage: None,
                    finish_reason: FinishReason::ToolCalls,
                    metadata: empty_metadata(),
                },
                ChatResponse {
                    message: assistant_message("Paris is sunny."),
                    tool_calls: Vec::new(),
                    usage: None,
                    finish_reason: FinishReason::Completed,
                    metadata: empty_metadata(),
                },
            ],
        ));

        let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
        let mut agent = runtime.agent_session()?;
        agent
            .mcp_registry()
            .register_tool(weather_tool_definition(), |arguments| async move {
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_weather".to_owned(),
                    content: format!(
                        "weather for {}: sunny",
                        arguments["city"].as_str().unwrap_or("unknown")
                    ),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            });

        let response = agent.prompt("What's the weather in Paris?").await?;
        assert_eq!(response.message.content, "Paris is sunny.");

        let requests = lock_unpoisoned(&requests);
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].messages.len(), 3);
        assert_eq!(
            requests[1].messages[2].tool_name.as_deref(),
            Some("lookup_weather")
        );
        assert_eq!(requests[1].messages[2].content, "weather for Paris: sunny");

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn agent_register_tool_shorthand_routes_through_mcp_registry() -> Result<(), XlaiError> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(RecordingChatModel::new(
            requests.clone(),
            vec![
                ChatResponse {
                    message: assistant_message(""),
                    tool_calls: vec![ToolCall {
                        id: "tool_1".to_owned(),
                        tool_name: "lookup_weather".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    }],
                    usage: None,
                    finish_reason: FinishReason::ToolCalls,
                    metadata: empty_metadata(),
                },
                ChatResponse {
                    message: assistant_message("Paris is sunny."),
                    tool_calls: Vec::new(),
                    usage: None,
                    finish_reason: FinishReason::Completed,
                    metadata: empty_metadata(),
                },
            ],
        ));

        let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
        let mut agent = runtime.agent_session()?;
        agent.register_tool(weather_tool_definition(), |arguments| async move {
            Ok(xlai_core::ToolResult {
                tool_name: "lookup_weather".to_owned(),
                content: format!(
                    "weather for {}: sunny",
                    arguments["city"].as_str().unwrap_or("unknown")
                ),
                is_error: false,
                metadata: empty_metadata(),
            })
        });

        let mcp_tools = agent.mcp_registry().tool_definitions();
        assert_eq!(mcp_tools.len(), 1);
        assert_eq!(mcp_tools[0].name, "lookup_weather");

        let response = agent.prompt("What's the weather in Paris?").await?;
        assert_eq!(response.message.content, "Paris is sunny.");

        let requests = lock_unpoisoned(&requests);
        assert_eq!(requests.len(), 2);
        assert!(
            requests[0]
                .available_tools
                .iter()
                .any(|tool| tool.name == "lookup_weather"),
            "expected shorthand registration to expose an MCP tool to the model"
        );
        assert_eq!(
            requests[1].messages[2].tool_name.as_deref(),
            Some("lookup_weather")
        );
        assert_eq!(requests[1].messages[2].content, "weather for Paris: sunny");

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn chat_can_load_embedded_system_prompt_assets() -> Result<(), XlaiError> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(RecordingChatModel::new(
            requests.clone(),
            vec![ChatResponse {
                message: assistant_message("Prompt loaded."),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            }],
        ));

        let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
        let chat = runtime
            .chat_session()
            .with_system_prompt_asset("system/tool-description-skill.md")?;

        let response = chat.prompt("Say something brief.").await?;
        assert_eq!(response.message.content, "Prompt loaded.");

        let requests = lock_unpoisoned(&requests);
        assert_eq!(requests.len(), 1);
        let system_prompt = requests[0].system_prompt.as_deref();
        assert_eq!(
            system_prompt,
            Some(EmbeddedPromptStore::load("system/tool-description-skill.md")?.as_str())
        );

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn chat_can_render_embedded_system_prompt_templates() -> Result<(), XlaiError> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(RecordingChatModel::new(
            requests.clone(),
            vec![ChatResponse {
                message: assistant_message("Rendered prompt loaded."),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            }],
        ));

        let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
        let mut context = PromptContext::new();
        context.insert("skill_tag_name", "tool_skill");
        let chat = runtime
            .chat_session()
            .with_system_prompt_template("system/tool-description-skill.md", &context)?;

        let response = chat.prompt("Say something brief.").await?;
        assert_eq!(response.message.content, "Rendered prompt loaded.");

        let requests = lock_unpoisoned(&requests);
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0]
                .system_prompt
                .as_deref()
                .is_some_and(|prompt| prompt.contains("tool_skill"))
        );

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn chat_stream_emits_model_and_tool_events() -> Result<(), XlaiError> {
        let model = Arc::new(StreamingChatModel::new(vec![
            vec![
                ChatChunk::MessageStart {
                    role: MessageRole::Assistant,
                },
                ChatChunk::ContentDelta("Looking up weather".to_owned()),
                ChatChunk::Finished(ChatResponse {
                    message: assistant_message("Looking up weather"),
                    tool_calls: vec![ToolCall {
                        id: "tool_stream_1".to_owned(),
                        tool_name: "lookup_weather".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    }],
                    usage: None,
                    finish_reason: FinishReason::ToolCalls,
                    metadata: empty_metadata(),
                }),
            ],
            vec![
                ChatChunk::MessageStart {
                    role: MessageRole::Assistant,
                },
                ChatChunk::ContentDelta("Paris is sunny.".to_owned()),
                ChatChunk::Finished(ChatResponse {
                    message: assistant_message("Paris is sunny."),
                    tool_calls: Vec::new(),
                    usage: None,
                    finish_reason: FinishReason::Completed,
                    metadata: empty_metadata(),
                }),
            ],
        ]));

        let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;

        let mut chat = runtime.chat_session();
        chat.register_tool(weather_tool_definition(), |arguments| async move {
            Ok(xlai_core::ToolResult {
                tool_name: "lookup_weather".to_owned(),
                content: format!(
                    "weather for {}: sunny",
                    arguments["city"].as_str().unwrap_or("unknown")
                ),
                is_error: false,
                metadata: empty_metadata(),
            })
        });

        let mut stream = chat.stream_prompt("Stream the weather.");
        let mut content_deltas = Vec::new();
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut finished_messages = Vec::new();

        while let Some(event) = stream.next().await {
            match event? {
                ChatExecutionEvent::Model(ChatChunk::ContentDelta(delta)) => {
                    content_deltas.push(delta);
                }
                ChatExecutionEvent::Model(ChatChunk::Finished(response)) => {
                    finished_messages.push(response.message.content);
                }
                ChatExecutionEvent::ToolCall(call) => {
                    saw_tool_call = true;
                    assert_eq!(call.tool_name, "lookup_weather");
                }
                ChatExecutionEvent::ToolResult(result) => {
                    saw_tool_result = true;
                    assert_eq!(result.content, "weather for Paris: sunny");
                }
                ChatExecutionEvent::Model(
                    ChatChunk::MessageStart { .. } | ChatChunk::ToolCallDelta(_),
                ) => {}
            }
        }

        assert_eq!(
            content_deltas,
            vec!["Looking up weather", "Paris is sunny."]
        );
        assert!(saw_tool_call);
        assert!(saw_tool_result);
        assert_eq!(
            finished_messages,
            vec!["Looking up weather", "Paris is sunny."]
        );

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn chat_executes_multiple_tool_calls_concurrently_by_default() -> Result<(), XlaiError> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(RecordingChatModel::new(
            requests.clone(),
            vec![
                ChatResponse {
                    message: assistant_message(""),
                    tool_calls: vec![
                        ToolCall {
                            id: "tool_1".to_owned(),
                            tool_name: "lookup_weather".to_owned(),
                            arguments: json!({ "city": "Paris" }),
                        },
                        ToolCall {
                            id: "tool_2".to_owned(),
                            tool_name: "lookup_time".to_owned(),
                            arguments: json!({ "city": "Paris" }),
                        },
                    ],
                    usage: None,
                    finish_reason: FinishReason::ToolCalls,
                    metadata: empty_metadata(),
                },
                ChatResponse {
                    message: assistant_message("Paris is sunny and 9am."),
                    tool_calls: Vec::new(),
                    usage: None,
                    finish_reason: FinishReason::Completed,
                    metadata: empty_metadata(),
                },
            ],
        ));

        let active_calls = Arc::new(AtomicUsize::new(0));
        let max_active_calls = Arc::new(AtomicUsize::new(0));
        let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;

        let mut chat = runtime.chat_session();
        {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            chat.register_tool(weather_tool_definition(), move |_| {
                let active_calls = active_calls.clone();
                let max_active_calls = max_active_calls.clone();
                async move {
                    let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                    max_active_calls.fetch_max(current, Ordering::SeqCst);
                    sleep(Duration::from_millis(25)).await;
                    active_calls.fetch_sub(1, Ordering::SeqCst);
                    Ok(xlai_core::ToolResult {
                        tool_name: "lookup_weather".to_owned(),
                        content: "weather for Paris: sunny".to_owned(),
                        is_error: false,
                        metadata: empty_metadata(),
                    })
                }
            });
        }
        {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            chat.register_tool(time_tool_definition(), move |_| {
                let active_calls = active_calls.clone();
                let max_active_calls = max_active_calls.clone();
                async move {
                    let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                    max_active_calls.fetch_max(current, Ordering::SeqCst);
                    sleep(Duration::from_millis(25)).await;
                    active_calls.fetch_sub(1, Ordering::SeqCst);
                    Ok(xlai_core::ToolResult {
                        tool_name: "lookup_time".to_owned(),
                        content: "time for Paris: 9am".to_owned(),
                        is_error: false,
                        metadata: empty_metadata(),
                    })
                }
            });
        }

        let response = chat.prompt("What's the weather and time in Paris?").await?;
        assert_eq!(response.message.content, "Paris is sunny and 9am.");

        assert!(
            max_active_calls.load(Ordering::SeqCst) >= 2,
            "default mode should overlap tool executions",
        );

        let requests = lock_unpoisoned(&requests);
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].messages.len(), 4);
        assert_eq!(
            requests[1].messages[2].tool_name.as_deref(),
            Some("lookup_weather")
        );
        assert_eq!(
            requests[1].messages[3].tool_name.as_deref(),
            Some("lookup_time")
        );

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn chat_can_execute_multiple_tool_calls_concurrently() -> Result<(), XlaiError> {
        let model = Arc::new(RecordingChatModel::new(
            Arc::new(Mutex::new(Vec::new())),
            vec![
                ChatResponse {
                    message: assistant_message(""),
                    tool_calls: vec![
                        ToolCall {
                            id: "tool_1".to_owned(),
                            tool_name: "lookup_weather".to_owned(),
                            arguments: json!({ "city": "Paris" }),
                        },
                        ToolCall {
                            id: "tool_2".to_owned(),
                            tool_name: "lookup_time".to_owned(),
                            arguments: json!({ "city": "Paris" }),
                        },
                    ],
                    usage: None,
                    finish_reason: FinishReason::ToolCalls,
                    metadata: empty_metadata(),
                },
                ChatResponse {
                    message: assistant_message("Paris is sunny and 9am."),
                    tool_calls: Vec::new(),
                    usage: None,
                    finish_reason: FinishReason::Completed,
                    metadata: empty_metadata(),
                },
            ],
        ));

        let active_calls = Arc::new(AtomicUsize::new(0));
        let max_active_calls = Arc::new(AtomicUsize::new(0));

        let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;

        let mut chat = runtime.chat_session();

        {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            chat.register_tool(weather_tool_definition(), move |_| {
                let active_calls = active_calls.clone();
                let max_active_calls = max_active_calls.clone();
                async move {
                    let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                    max_active_calls.fetch_max(current, Ordering::SeqCst);
                    sleep(Duration::from_millis(25)).await;
                    active_calls.fetch_sub(1, Ordering::SeqCst);
                    Ok(xlai_core::ToolResult {
                        tool_name: "lookup_weather".to_owned(),
                        content: "weather for Paris: sunny".to_owned(),
                        is_error: false,
                        metadata: empty_metadata(),
                    })
                }
            });
        }
        {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            chat.register_tool(time_tool_definition(), move |_| {
                let active_calls = active_calls.clone();
                let max_active_calls = max_active_calls.clone();
                async move {
                    let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                    max_active_calls.fetch_max(current, Ordering::SeqCst);
                    sleep(Duration::from_millis(25)).await;
                    active_calls.fetch_sub(1, Ordering::SeqCst);
                    Ok(xlai_core::ToolResult {
                        tool_name: "lookup_time".to_owned(),
                        content: "time for Paris: 9am".to_owned(),
                        is_error: false,
                        metadata: empty_metadata(),
                    })
                }
            });
        }

        let response = chat.prompt("What's the weather and time in Paris?").await?;
        assert_eq!(response.message.content, "Paris is sunny and 9am.");
        assert!(
            max_active_calls.load(Ordering::SeqCst) >= 2,
            "concurrent mode should overlap tool executions",
        );

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn chat_runs_tool_batch_sequentially_when_any_tool_is_sequential() -> Result<(), XlaiError>
    {
        let execution_order = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(RecordingChatModel::new(
            Arc::new(Mutex::new(Vec::new())),
            vec![
                ChatResponse {
                    message: assistant_message(""),
                    tool_calls: vec![
                        ToolCall {
                            id: "tool_1".to_owned(),
                            tool_name: "lookup_weather".to_owned(),
                            arguments: json!({ "city": "Paris" }),
                        },
                        ToolCall {
                            id: "tool_2".to_owned(),
                            tool_name: "lookup_time".to_owned(),
                            arguments: json!({ "city": "Paris" }),
                        },
                    ],
                    usage: None,
                    finish_reason: FinishReason::ToolCalls,
                    metadata: empty_metadata(),
                },
                ChatResponse {
                    message: assistant_message("Paris is sunny and 9am."),
                    tool_calls: Vec::new(),
                    usage: None,
                    finish_reason: FinishReason::Completed,
                    metadata: empty_metadata(),
                },
            ],
        ));

        let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;

        let mut chat = runtime.chat_session();
        {
            let execution_order = execution_order.clone();
            chat.register_tool(weather_tool_definition_sequential(), move |arguments| {
                let execution_order = execution_order.clone();
                async move {
                    lock_unpoisoned(&execution_order).push(format!(
                        "weather:{}",
                        arguments["city"].as_str().unwrap_or("unknown")
                    ));
                    Ok(xlai_core::ToolResult {
                        tool_name: "lookup_weather".to_owned(),
                        content: "weather for Paris: sunny".to_owned(),
                        is_error: false,
                        metadata: empty_metadata(),
                    })
                }
            });
        }
        {
            let execution_order = execution_order.clone();
            chat.register_tool(time_tool_definition(), move |arguments| {
                let execution_order = execution_order.clone();
                async move {
                    lock_unpoisoned(&execution_order).push(format!(
                        "time:{}",
                        arguments["city"].as_str().unwrap_or("unknown")
                    ));
                    Ok(xlai_core::ToolResult {
                        tool_name: "lookup_time".to_owned(),
                        content: "time for Paris: 9am".to_owned(),
                        is_error: false,
                        metadata: empty_metadata(),
                    })
                }
            });
        }

        let response = chat.prompt("What's the weather and time in Paris?").await?;
        assert_eq!(response.message.content, "Paris is sunny and 9am.");

        let execution_order = lock_unpoisoned(&execution_order);
        assert_eq!(
            *execution_order,
            vec!["weather:Paris".to_owned(), "time:Paris".to_owned()]
        );

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn runtime_file_system_helpers_use_configured_backend() -> Result<(), XlaiError> {
        let runtime = RuntimeBuilder::new()
            .with_file_system(Arc::new(MemoryFileSystem::new()))
            .build()?;

        assert!(runtime.has_capability(RuntimeCapability::FileSystem));

        runtime.create_dir_all(&FsPath::from("/docs")).await?;
        runtime
            .write_file(&FsPath::from("/docs/readme.md"), b"runtime".to_vec())
            .await?;

        let bytes = runtime.read_file(&FsPath::from("/docs/readme.md")).await?;
        assert_eq!(bytes, b"runtime".to_vec());

        let exists = runtime
            .path_exists(&FsPath::from("/docs/readme.md"))
            .await?;
        assert!(exists);

        let entries = runtime.list_directory(&FsPath::from("/docs")).await?;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, FsPath::from("/docs/readme.md"));
        assert_eq!(entries[0].kind, FsEntryKind::File);

        runtime
            .delete_path(&FsPath::from("/docs/readme.md"))
            .await?;
        let exists = runtime
            .path_exists(&FsPath::from("/docs/readme.md"))
            .await?;
        assert!(!exists);

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn runtime_file_system_helpers_error_without_backend() -> Result<(), XlaiError> {
        let runtime = RuntimeBuilder::new()
            .with_chat_model(Arc::new(RecordingChatModel::new(
                Arc::new(Mutex::new(Vec::new())),
                vec![ChatResponse {
                    message: assistant_message("unused"),
                    tool_calls: Vec::new(),
                    usage: None,
                    finish_reason: FinishReason::Completed,
                    metadata: empty_metadata(),
                }],
            )))
            .build()?;

        let error = match runtime.read_file(&FsPath::from("/docs/readme.md")).await {
            Ok(_) => {
                return Err(XlaiError::new(
                    xlai_core::ErrorKind::Validation,
                    "expected missing filesystem dependency error",
                ));
            }
            Err(error) => error,
        };
        assert_eq!(error.kind, xlai_core::ErrorKind::Unsupported);
        assert!(
            error.message.contains("file system"),
            "expected error message to mention file system"
        );

        Ok(())
    }

    fn weather_tool_definition() -> ToolDefinition {
        ToolDefinition {
            name: "lookup_weather".to_owned(),
            description: "Lookup current weather".to_owned(),
            parameters: vec![ToolParameter {
                name: "city".to_owned(),
                description: "The city name".to_owned(),
                kind: ToolParameterType::String,
                required: true,
            }],
            execution_mode: ToolCallExecutionMode::Concurrent,
        }
    }

    fn weather_tool_definition_sequential() -> ToolDefinition {
        ToolDefinition {
            execution_mode: ToolCallExecutionMode::Sequential,
            ..weather_tool_definition()
        }
    }

    fn time_tool_definition() -> ToolDefinition {
        ToolDefinition {
            name: "lookup_time".to_owned(),
            description: "Lookup current time".to_owned(),
            parameters: vec![ToolParameter {
                name: "city".to_owned(),
                description: "The city name".to_owned(),
                kind: ToolParameterType::String,
                required: true,
            }],
            execution_mode: ToolCallExecutionMode::Concurrent,
        }
    }

    fn assistant_message(content: &str) -> ChatMessage {
        ChatMessage {
            role: MessageRole::Assistant,
            content: content.to_owned(),
            tool_name: None,
            tool_call_id: None,
            metadata: empty_metadata(),
        }
    }

    fn sample_skill() -> Skill {
        Skill {
            id: "review.code".to_owned(),
            name: "review.code".to_owned(),
            description: "Reviews code with a bug-finding mindset.".to_owned(),
            prompt_fragment: "Prioritize bugs, regressions, and missing tests.".to_owned(),
            tags: vec!["review".to_owned(), "quality".to_owned()],
            metadata: empty_metadata(),
        }
    }

    fn sample_skill_markdown() -> String {
        format!(
            "---\nname: {}\ndescription: {}\ntags:\n  - review\n  - quality\n---\n{}\n",
            sample_skill().name,
            sample_skill().description,
            sample_skill().prompt_fragment
        )
    }

    async fn seed_markdown_skill_store() -> Result<Arc<dyn SkillStore>, XlaiError> {
        let file_system = Arc::new(MemoryFileSystem::new());
        let file_system_trait: Arc<dyn xlai_core::SkillFileSystem> = file_system.clone();
        let skill_store = Arc::new(MarkdownSkillStore::new(file_system_trait));

        file_system
            .create_dir_all(&FsPath::from("/skills/review"))
            .await?;
        file_system
            .write(
                &FsPath::from("/skills/review/SKILL.md"),
                sample_skill_markdown().into_bytes(),
            )
            .await?;
        file_system
            .write(
                &FsPath::from("/skills/review/README.md"),
                b"Extra skill file".to_vec(),
            )
            .await?;
        Ok(skill_store)
    }

    struct RecordingChatModel {
        requests: Arc<Mutex<Vec<ChatRequest>>>,
        responses: Mutex<VecDeque<ChatResponse>>,
    }

    impl RecordingChatModel {
        fn new(requests: Arc<Mutex<Vec<ChatRequest>>>, responses: Vec<ChatResponse>) -> Self {
            Self {
                requests,
                responses: Mutex::new(VecDeque::from(responses)),
            }
        }
    }

    impl ChatModel for RecordingChatModel {
        fn provider_name(&self) -> &'static str {
            "recording-test"
        }

        fn generate(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, XlaiError>> {
            Box::pin(async move {
                lock_unpoisoned(&self.requests).push(request);
                lock_unpoisoned(&self.responses).pop_front().ok_or_else(|| {
                    XlaiError::new(xlai_core::ErrorKind::Provider, "missing response")
                })
            })
        }
    }

    struct StreamingChatModel {
        streams: Mutex<VecDeque<Vec<ChatChunk>>>,
    }

    impl StreamingChatModel {
        fn new(streams: Vec<Vec<ChatChunk>>) -> Self {
            Self {
                streams: Mutex::new(VecDeque::from(streams)),
            }
        }
    }

    impl ChatModel for StreamingChatModel {
        fn provider_name(&self) -> &'static str {
            "streaming-test"
        }

        fn generate(
            &self,
            _request: ChatRequest,
        ) -> BoxFuture<'_, Result<ChatResponse, XlaiError>> {
            Box::pin(async {
                Err(XlaiError::new(
                    xlai_core::ErrorKind::Unsupported,
                    "generate is not used in this test model",
                ))
            })
        }

        fn generate_stream(
            &self,
            _request: ChatRequest,
        ) -> BoxStream<'_, Result<ChatChunk, XlaiError>> {
            let chunks = lock_unpoisoned(&self.streams)
                .pop_front()
                .unwrap_or_default();

            Box::pin(stream::iter(chunks.into_iter().map(Ok)))
        }
    }
}
