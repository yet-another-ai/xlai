use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use futures_util::{StreamExt, stream};
use xlai_core::{
    BoxFuture, BoxStream, ChatChunk, ChatContent, ChatMessage, ChatModel, ChatRequest,
    ChatResponse, EmbeddingModel, EmbeddingRequest, EmbeddingResponse, FsPath, GeneratedImage,
    ImageGenerationModel, ImageGenerationRequest, ImageGenerationResponse, MediaSource,
    MessageRole, Metadata, Skill, SkillLoadPolicy, SkillResource, SkillStore,
    ToolCallExecutionMode, ToolDefinition, ToolSchema, TranscriptionModel, TranscriptionRequest,
    TranscriptionResponse, TtsModel, TtsRequest, TtsResponse, XlaiError,
};

use crate::{Agent, ChatExecutionEvent, MarkdownSkillStore, MemoryFileSystem};
use xlai_core::WritableFileSystem;

pub(super) fn empty_metadata() -> Metadata {
    Metadata::new()
}

pub(super) fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

pub(super) fn weather_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "lookup_weather",
        "Lookup current weather",
        ToolSchema::object(
            BTreeMap::from([(
                "city".to_owned(),
                ToolSchema::string().with_description("The city name"),
            )]),
            vec!["city".to_owned()],
        ),
    )
    .with_execution_mode(ToolCallExecutionMode::Concurrent)
}

pub(super) fn weather_tool_definition_sequential() -> ToolDefinition {
    ToolDefinition {
        execution_mode: ToolCallExecutionMode::Sequential,
        ..weather_tool_definition()
    }
}

pub(super) fn time_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "lookup_time",
        "Lookup current time",
        ToolSchema::object(
            BTreeMap::from([(
                "city".to_owned(),
                ToolSchema::string().with_description("The city name"),
            )]),
            vec!["city".to_owned()],
        ),
    )
    .with_execution_mode(ToolCallExecutionMode::Concurrent)
}

pub(super) fn time_tool_definition_sequential() -> ToolDefinition {
    ToolDefinition {
        execution_mode: ToolCallExecutionMode::Sequential,
        ..time_tool_definition()
    }
}

pub(super) fn calendar_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "lookup_calendar",
        "Lookup current calendar",
        ToolSchema::object(
            BTreeMap::from([(
                "city".to_owned(),
                ToolSchema::string().with_description("The city name"),
            )]),
            vec!["city".to_owned()],
        ),
    )
    .with_execution_mode(ToolCallExecutionMode::Concurrent)
}

pub(super) fn calendar_tool_definition_sequential() -> ToolDefinition {
    ToolDefinition {
        execution_mode: ToolCallExecutionMode::Sequential,
        ..calendar_tool_definition()
    }
}

pub(super) fn assistant_message(content: &str) -> ChatMessage {
    ChatMessage {
        role: MessageRole::Assistant,
        content: ChatContent::text(content),
        tool_name: None,
        tool_call_id: None,
        metadata: empty_metadata(),
    }
}

pub(super) fn sample_skill() -> Skill {
    Skill {
        id: "review.code".to_owned(),
        name: "review.code".to_owned(),
        description: "Reviews code with a bug-finding mindset.".to_owned(),
        prompt_fragment: "Prioritize bugs, regressions, and missing tests.".to_owned(),
        resources: vec![SkillResource {
            path: "README.md".to_owned(),
            kind: None,
            purpose: None,
            required: false,
        }],
        entrypoints: vec!["SKILL.md".to_owned()],
        load_policy: Some(SkillLoadPolicy::default()),
        tags: vec!["review".to_owned(), "quality".to_owned()],
        metadata: empty_metadata(),
    }
}

pub(super) fn sample_skill_markdown() -> String {
    format!(
        "---\nname: {}\ndescription: {}\ntags:\n  - review\n  - quality\n---\n{}\n",
        sample_skill().name,
        sample_skill().description,
        sample_skill().prompt_fragment
    )
}

pub(super) async fn seed_markdown_skill_store() -> Result<Arc<dyn SkillStore>, XlaiError> {
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

pub(super) async fn seed_markdown_skill_store_with_resources()
-> Result<Arc<dyn SkillStore>, XlaiError> {
    let file_system = Arc::new(MemoryFileSystem::new());
    let file_system_trait: Arc<dyn xlai_core::SkillFileSystem> = file_system.clone();
    let skill_store = Arc::new(MarkdownSkillStore::new(file_system_trait));

    file_system
        .create_dir_all(&FsPath::from("/skills/review/references"))
        .await?;
    file_system
        .create_dir_all(&FsPath::from("/skills/review/templates"))
        .await?;
    file_system
        .write(
            &FsPath::from("/skills/review/SKILL.md"),
            br#"---
name: review.code
description: Reviews code with a bug-finding mindset.
tags:
  - review
  - quality
resources:
  - path: references/checklist.md
    kind: markdown
    purpose: review checklist
    required: true
  - path: templates/final.md
    kind: markdown
    purpose: response template
---
Prioritize bugs, regressions, and missing tests.
"#
            .to_vec(),
        )
        .await?;
    file_system
        .write(
            &FsPath::from("/skills/review/references/checklist.md"),
            b"Checklist item 1".to_vec(),
        )
        .await?;
    file_system
        .write(
            &FsPath::from("/skills/review/templates/final.md"),
            b"Template body".to_vec(),
        )
        .await?;

    Ok(skill_store)
}

pub(super) struct RecordingChatModel {
    requests: Arc<Mutex<Vec<ChatRequest>>>,
    responses: Mutex<VecDeque<ChatResponse>>,
}

impl RecordingChatModel {
    pub(super) fn new(
        requests: Arc<Mutex<Vec<ChatRequest>>>,
        responses: Vec<ChatResponse>,
    ) -> Self {
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
            lock_unpoisoned(&self.responses)
                .pop_front()
                .ok_or_else(|| XlaiError::new(xlai_core::ErrorKind::Provider, "missing response"))
        })
    }
}

pub(super) struct StreamingChatModel {
    streams: Mutex<VecDeque<Vec<ChatChunk>>>,
}

impl StreamingChatModel {
    pub(super) fn new(streams: Vec<Vec<ChatChunk>>) -> Self {
        Self {
            streams: Mutex::new(VecDeque::from(streams)),
        }
    }
}

impl ChatModel for StreamingChatModel {
    fn provider_name(&self) -> &'static str {
        "streaming-test"
    }

    fn generate(&self, _request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, XlaiError>> {
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

pub(super) struct RecordingTtsModel {
    requests: Arc<Mutex<Vec<TtsRequest>>>,
    responses: Mutex<VecDeque<TtsResponse>>,
}

impl RecordingTtsModel {
    pub(super) fn new(requests: Arc<Mutex<Vec<TtsRequest>>>, responses: Vec<TtsResponse>) -> Self {
        Self {
            requests,
            responses: Mutex::new(VecDeque::from(responses)),
        }
    }
}

impl TtsModel for RecordingTtsModel {
    fn provider_name(&self) -> &'static str {
        "recording-tts-test"
    }

    fn synthesize(&self, request: TtsRequest) -> BoxFuture<'_, Result<TtsResponse, XlaiError>> {
        Box::pin(async move {
            lock_unpoisoned(&self.requests).push(request);
            lock_unpoisoned(&self.responses).pop_front().ok_or_else(|| {
                XlaiError::new(xlai_core::ErrorKind::Provider, "missing tts response")
            })
        })
    }
}

pub(super) struct RecordingTranscriptionModel {
    requests: Arc<Mutex<Vec<TranscriptionRequest>>>,
    responses: Mutex<VecDeque<TranscriptionResponse>>,
}

impl RecordingTranscriptionModel {
    pub(super) fn new(
        requests: Arc<Mutex<Vec<TranscriptionRequest>>>,
        responses: Vec<TranscriptionResponse>,
    ) -> Self {
        Self {
            requests,
            responses: Mutex::new(VecDeque::from(responses)),
        }
    }
}

impl TranscriptionModel for RecordingTranscriptionModel {
    fn provider_name(&self) -> &'static str {
        "recording-transcription-test"
    }

    fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> BoxFuture<'_, Result<TranscriptionResponse, XlaiError>> {
        Box::pin(async move {
            lock_unpoisoned(&self.requests).push(request);
            lock_unpoisoned(&self.responses).pop_front().ok_or_else(|| {
                XlaiError::new(
                    xlai_core::ErrorKind::Provider,
                    "missing transcription response",
                )
            })
        })
    }
}

pub(super) struct RecordingImageGenerationModel {
    requests: Arc<Mutex<Vec<ImageGenerationRequest>>>,
    responses: Mutex<VecDeque<ImageGenerationResponse>>,
}

impl RecordingImageGenerationModel {
    pub(super) fn new(
        requests: Arc<Mutex<Vec<ImageGenerationRequest>>>,
        responses: Vec<ImageGenerationResponse>,
    ) -> Self {
        Self {
            requests,
            responses: Mutex::new(VecDeque::from(responses)),
        }
    }
}

impl ImageGenerationModel for RecordingImageGenerationModel {
    fn provider_name(&self) -> &'static str {
        "recording-image-generation-test"
    }

    fn generate_image(
        &self,
        request: ImageGenerationRequest,
    ) -> BoxFuture<'_, Result<ImageGenerationResponse, XlaiError>> {
        Box::pin(async move {
            lock_unpoisoned(&self.requests).push(request);
            lock_unpoisoned(&self.responses).pop_front().ok_or_else(|| {
                XlaiError::new(
                    xlai_core::ErrorKind::Provider,
                    "missing image generation response",
                )
            })
        })
    }
}

pub(super) struct RecordingEmbeddingModel {
    requests: Arc<Mutex<Vec<EmbeddingRequest>>>,
    responses: Mutex<VecDeque<EmbeddingResponse>>,
}

impl RecordingEmbeddingModel {
    pub(super) fn new(
        requests: Arc<Mutex<Vec<EmbeddingRequest>>>,
        responses: Vec<EmbeddingResponse>,
    ) -> Self {
        Self {
            requests,
            responses: Mutex::new(VecDeque::from(responses)),
        }
    }
}

impl EmbeddingModel for RecordingEmbeddingModel {
    fn provider_name(&self) -> &'static str {
        "recording-embedding-test"
    }

    fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> BoxFuture<'_, Result<EmbeddingResponse, XlaiError>> {
        Box::pin(async move {
            lock_unpoisoned(&self.requests).push(request);
            lock_unpoisoned(&self.responses).pop_front().ok_or_else(|| {
                XlaiError::new(xlai_core::ErrorKind::Provider, "missing embedding response")
            })
        })
    }
}

pub(super) fn sample_generated_image_response() -> ImageGenerationResponse {
    ImageGenerationResponse {
        images: vec![GeneratedImage {
            image: MediaSource::InlineData {
                mime_type: "image/png".to_owned(),
                data: vec![137, 80, 78, 71],
            },
            mime_type: Some("image/png".to_owned()),
            revised_prompt: Some("A bright orange cat sitting on a windowsill".to_owned()),
            metadata: empty_metadata(),
        }],
        metadata: empty_metadata(),
    }
}

pub(super) async fn agent_stream_prompt_final_response(
    agent: &Agent,
    prompt: &str,
) -> Result<ChatResponse, XlaiError> {
    let mut stream = agent.stream_prompt(prompt);
    let mut last = None;
    while let Some(item) = stream.next().await {
        let item = item?;
        if let ChatExecutionEvent::Model(ChatChunk::Finished(resp)) = item {
            last = Some(resp);
        }
    }
    last.ok_or_else(|| {
        XlaiError::new(
            xlai_core::ErrorKind::Provider,
            "stream ended without a finished model response",
        )
    })
}
