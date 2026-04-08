use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

use async_stream::try_stream;
use futures_util::StreamExt;
use futures_util::future::try_join_all;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tera::Context;
use xlai_core::{
    BoxFuture, BoxStream, ChatChunk, ChatContent, ChatMessage, ChatRequest, ChatResponse,
    ChatRetryPolicy, ContentPart, ErrorKind, MaybeSend, MessageRole, ReasoningEffort, RuntimeBound,
    StructuredOutput, ToolCall, ToolCallExecutionMode, ToolDefinition, ToolResult, XlaiError,
};

use crate::{EmbeddedPromptStore, XlaiRuntime};

#[cfg(not(target_arch = "wasm32"))]
type ToolCallback =
    dyn Fn(Value) -> BoxFuture<'static, Result<ToolResult, XlaiError>> + Send + Sync;
#[cfg(target_arch = "wasm32")]
type ToolCallback = dyn Fn(Value) -> BoxFuture<'static, Result<ToolResult, XlaiError>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolOrigin {
    Builtin,
    Local,
    Mcp,
}

#[derive(Clone)]
struct RegisteredTool {
    definition: ToolDefinition,
    callback: Arc<ToolCallback>,
    origin: ToolOrigin,
}

/// One item from a chat/agent stream (model chunks, tool calls, tool results).
///
/// Serialized for WASM/JS as `{"kind":"model"|"toolCall"|"toolResult","data":...}` (camelCase).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "camelCase")]
pub enum ChatExecutionEvent {
    Model(ChatChunk),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
}

#[derive(Clone)]
pub(crate) struct ToolExecutionOutcome {
    pub(crate) call: ToolCall,
    pub(crate) result: ToolResult,
}

#[derive(Clone)]
pub struct Chat {
    runtime: Arc<XlaiRuntime>,
    model: Option<String>,
    system_prompt: Option<String>,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    reasoning_effort: Option<ReasoningEffort>,
    structured_output: Option<StructuredOutput>,
    retry_policy: Option<ChatRetryPolicy>,
    tools: BTreeMap<String, RegisteredTool>,
}

impl Chat {
    #[must_use]
    pub fn new(runtime: Arc<XlaiRuntime>) -> Self {
        Self {
            runtime,
            model: None,
            system_prompt: None,
            temperature: None,
            max_output_tokens: None,
            reasoning_effort: None,
            structured_output: None,
            retry_policy: None,
            tools: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Loads a raw system prompt from the embedded prompt catalog.
    ///
    /// # Errors
    ///
    /// Returns an error if the named prompt asset does not exist or is not
    /// valid UTF-8.
    pub fn with_system_prompt_asset(
        mut self,
        prompt_name: impl AsRef<str>,
    ) -> Result<Self, XlaiError> {
        self.system_prompt = Some(EmbeddedPromptStore::load(prompt_name.as_ref())?);
        Ok(self)
    }

    /// Renders an embedded system prompt template with `tera`.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt asset does not exist, cannot be parsed as
    /// a template, or fails to render with the provided context.
    pub fn with_system_prompt_template(
        mut self,
        prompt_name: impl AsRef<str>,
        context: &Context,
    ) -> Result<Self, XlaiError> {
        self.system_prompt = Some(EmbeddedPromptStore::render(prompt_name.as_ref(), context)?);
        Ok(self)
    }

    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    #[must_use]
    pub fn with_max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    #[must_use]
    pub fn with_reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(reasoning_effort);
        self
    }

    #[must_use]
    pub fn with_structured_output(mut self, structured_output: StructuredOutput) -> Self {
        self.structured_output = Some(structured_output);
        self
    }

    /// Sets an optional auto-retry policy for outgoing chat requests.
    ///
    /// Backends that support it use this hint; others ignore it. When `None`, no policy is sent
    /// (backends typically do not auto-retry).
    #[must_use]
    pub fn with_retry_policy(mut self, retry_policy: Option<ChatRetryPolicy>) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    pub fn register_tool<F, Fut>(&mut self, definition: ToolDefinition, callback: F) -> &mut Self
    where
        F: Fn(Value) -> Fut + RuntimeBound + 'static,
        Fut: Future<Output = Result<ToolResult, XlaiError>> + MaybeSend + 'static,
    {
        self.register_tool_with_origin(definition, ToolOrigin::Local, callback)
    }

    pub(crate) fn register_tool_with_origin<F, Fut>(
        &mut self,
        definition: ToolDefinition,
        origin: ToolOrigin,
        callback: F,
    ) -> &mut Self
    where
        F: Fn(Value) -> Fut + RuntimeBound + 'static,
        Fut: Future<Output = Result<ToolResult, XlaiError>> + MaybeSend + 'static,
    {
        let tool_name = definition.name.clone();
        let callback = Arc::new(
            move |arguments| -> BoxFuture<'static, Result<ToolResult, XlaiError>> {
                Box::pin(callback(arguments))
            },
        );

        self.tools.insert(
            tool_name,
            RegisteredTool {
                definition,
                callback,
                origin,
            },
        );
        self
    }

    #[must_use]
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.definition.clone())
            .collect()
    }

    #[must_use]
    pub(crate) fn tool_definitions_with_origin(&self, origin: ToolOrigin) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .filter(|tool| tool.origin == origin)
            .map(|tool| tool.definition.clone())
            .collect()
    }

    /// Sends a single user prompt through this chat session (one model call).
    ///
    /// # Errors
    ///
    /// Returns an error if the configured model request fails. Tool calls returned by the model
    /// are not executed here; use [`crate::Agent`] for automatic tool round-trips.
    pub async fn prompt(&self, content: impl Into<String>) -> Result<ChatResponse, XlaiError> {
        self.prompt_content(ChatContent::text(content.into())).await
    }

    /// Sends a single user turn with structured multimodal [`ChatContent`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::prompt`].
    pub async fn prompt_content(&self, content: ChatContent) -> Result<ChatResponse, XlaiError> {
        self.execute(vec![ChatMessage {
            role: MessageRole::User,
            content,
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }])
        .await
    }

    /// Sends a single user turn built from ordered [`ContentPart`]s (text, image, file, …).
    ///
    /// # Errors
    ///
    /// Same as [`Self::prompt`].
    pub async fn prompt_parts(&self, parts: Vec<ContentPart>) -> Result<ChatResponse, XlaiError> {
        self.prompt_content(ChatContent::from_parts(parts)).await
    }

    /// Runs exactly one model call with the given message history.
    ///
    /// Does not execute tool callbacks. For multi-round tool execution, use [`crate::Agent`].
    ///
    /// # Errors
    ///
    /// Returns an error if the configured model request fails.
    pub async fn execute(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse, XlaiError> {
        self.runtime.chat(self.build_request(messages)).await
    }

    /// Runs registered tools and the runtime tool executor for this batch (used by [`crate::Agent`]).
    pub(crate) async fn dispatch_tool_calls(
        &self,
        calls: Vec<ToolCall>,
    ) -> Result<Vec<ToolExecutionOutcome>, XlaiError> {
        execute_tool_calls_from(self.runtime.as_ref(), &self.tools, calls).await
    }

    pub(crate) fn tool_result_as_message(call: &ToolCall, result: &ToolResult) -> ChatMessage {
        tool_result_message(call, result)
    }

    #[must_use]
    pub fn stream_prompt(
        &self,
        content: impl Into<String>,
    ) -> BoxStream<'static, Result<ChatExecutionEvent, XlaiError>> {
        self.stream_prompt_content(ChatContent::text(content.into()))
    }

    /// Like [`Self::stream_prompt`], but with structured [`ChatContent`].
    #[must_use]
    pub fn stream_prompt_content(
        &self,
        content: ChatContent,
    ) -> BoxStream<'static, Result<ChatExecutionEvent, XlaiError>> {
        self.stream(vec![ChatMessage {
            role: MessageRole::User,
            content,
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }])
    }

    /// Like [`Self::stream_prompt`], but with multimodal [`ContentPart`]s.
    #[must_use]
    pub fn stream_prompt_parts(
        &self,
        parts: Vec<ContentPart>,
    ) -> BoxStream<'static, Result<ChatExecutionEvent, XlaiError>> {
        self.stream_prompt_content(ChatContent::from_parts(parts))
    }

    /// Streams a single model turn for `messages` (no tool execution).
    ///
    /// Multi-round streaming with tools is implemented on [`crate::Agent`].
    #[must_use]
    pub fn stream(
        &self,
        messages: Vec<ChatMessage>,
    ) -> BoxStream<'static, Result<ChatExecutionEvent, XlaiError>> {
        let runtime = Arc::clone(&self.runtime);
        let chat = self.clone();

        Box::pin(try_stream! {
            let request = chat.build_request(messages);
            let mut response = None;
            let mut model_stream = runtime.stream_chat(request)?;

            while let Some(chunk) = model_stream.next().await {
                let chunk = chunk?;
                if let ChatChunk::Finished(final_response) = &chunk {
                    response = Some(final_response.clone());
                }
                yield ChatExecutionEvent::Model(chunk);
            }

            response.ok_or_else(|| {
                XlaiError::new(
                    ErrorKind::Provider,
                    "stream ended without a final chat response",
                )
            })?;
        })
    }

    fn build_request(&self, messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest {
            model: self.model.clone(),
            system_prompt: self.system_prompt.clone(),
            messages,
            available_tools: self.tool_definitions(),
            structured_output: self.structured_output.clone(),
            metadata: BTreeMap::new(),
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            reasoning_effort: self.reasoning_effort,
            retry_policy: self.retry_policy.clone(),
        }
    }

    /// Best-effort estimate of input-side token load for the outgoing request that would be built
    /// from `messages` (same shape as [`Self::execute`] / [`Self::stream`]).
    ///
    /// Uses JSON serialization size with a bytes÷4 heuristic; not provider-tokenizer-accurate.
    pub(crate) fn estimate_input_tokens_for_messages(
        &self,
        messages: &[ChatMessage],
    ) -> Option<u32> {
        estimate_chat_request_input_tokens(&self.build_request(messages.to_vec()))
    }
}

/// Best-effort input token estimate from the full [`ChatRequest`] wire payload.
///
/// Returns [`None`] if the request cannot be serialized. This is intentionally approximate.
pub(crate) fn estimate_chat_request_input_tokens(request: &ChatRequest) -> Option<u32> {
    let bytes = serde_json::to_vec(request).ok()?;
    // Common rough rule: ~4 UTF-8 bytes per token for English-ish text; JSON adds structure overhead.
    let est = (bytes.len() as u128).saturating_add(3) / 4;
    Some(est.min(u32::MAX as u128) as u32)
}

fn tool_result_message(call: &ToolCall, result: &ToolResult) -> ChatMessage {
    ChatMessage {
        role: MessageRole::Tool,
        content: ChatContent::text(result.content.clone()),
        tool_name: Some(call.tool_name.clone()),
        tool_call_id: Some(call.id.clone()),
        metadata: result.metadata.clone(),
    }
}

async fn execute_tool_call_from(
    runtime: &XlaiRuntime,
    tools: &BTreeMap<String, RegisteredTool>,
    call: ToolCall,
) -> Result<ToolResult, XlaiError> {
    if let Some(tool) = tools.get(&call.tool_name) {
        let result = (tool.callback)(call.arguments).await?;
        return Ok(ToolResult {
            tool_name: tool.definition.name.clone(),
            content: result.content,
            is_error: result.is_error,
            metadata: result.metadata,
        });
    }

    runtime.call_tool(call).await
}

fn resolve_tool_batch_execution_mode(
    calls: &[ToolCall],
    tools: &BTreeMap<String, RegisteredTool>,
) -> ToolCallExecutionMode {
    for call in calls {
        if let Some(tool) = tools.get(&call.tool_name)
            && tool.definition.execution_mode == ToolCallExecutionMode::Sequential
        {
            return ToolCallExecutionMode::Sequential;
        }
    }
    ToolCallExecutionMode::Concurrent
}

async fn execute_tool_calls_from(
    runtime: &XlaiRuntime,
    tools: &BTreeMap<String, RegisteredTool>,
    calls: Vec<ToolCall>,
) -> Result<Vec<ToolExecutionOutcome>, XlaiError> {
    let mode = resolve_tool_batch_execution_mode(&calls, tools);
    match mode {
        ToolCallExecutionMode::Sequential => {
            let mut outcomes = Vec::with_capacity(calls.len());
            for call in calls {
                let result = execute_tool_call_from(runtime, tools, call.clone()).await?;
                outcomes.push(ToolExecutionOutcome { call, result });
            }
            Ok(outcomes)
        }
        ToolCallExecutionMode::Concurrent => {
            try_join_all(calls.into_iter().map(|call| async {
                let result = execute_tool_call_from(runtime, tools, call.clone()).await?;
                Ok(ToolExecutionOutcome { call, result })
            }))
            .await
        }
    }
}
