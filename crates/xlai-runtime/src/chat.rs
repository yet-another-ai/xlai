use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

use async_stream::try_stream;
use futures_util::StreamExt;
use futures_util::future::try_join_all;
use serde_json::Value;
use tera::Context;
use xlai_core::{
    BoxFuture, BoxStream, ChatChunk, ChatMessage, ChatRequest, ChatResponse, ErrorKind, MaybeSend,
    MessageRole, RuntimeBound, ToolCall, ToolCallExecutionMode, ToolDefinition, ToolResult,
    XlaiError,
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

#[derive(Clone, Debug)]
pub enum ChatExecutionEvent {
    Model(ChatChunk),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
}

#[derive(Clone)]
struct ToolExecutionOutcome {
    call: ToolCall,
    result: ToolResult,
}

#[derive(Clone)]
pub struct Chat {
    runtime: Arc<XlaiRuntime>,
    model: Option<String>,
    system_prompt: Option<String>,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    max_tool_round_trips: usize,
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
            max_tool_round_trips: 8,
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
    pub fn with_max_tool_round_trips(mut self, max_tool_round_trips: usize) -> Self {
        self.max_tool_round_trips = max_tool_round_trips;
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

    /// Sends a single user prompt through this chat session.
    ///
    /// # Errors
    ///
    /// Returns an error if the configured model request fails, if a tool callback
    /// fails, or if the chat exceeds the maximum number of tool round trips.
    pub async fn prompt(&self, content: impl Into<String>) -> Result<ChatResponse, XlaiError> {
        self.execute(vec![ChatMessage {
            role: MessageRole::User,
            content: content.into(),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }])
        .await
    }

    /// Executes a chat turn with the provided message history.
    ///
    /// # Errors
    ///
    /// Returns an error if the configured model request fails, if a tool callback
    /// fails, or if the chat exceeds the maximum number of tool round trips.
    pub async fn execute(&self, mut messages: Vec<ChatMessage>) -> Result<ChatResponse, XlaiError> {
        for _ in 0..self.max_tool_round_trips {
            let response = self
                .runtime
                .chat(self.build_request(messages.clone()))
                .await?;
            messages.push(response.message.clone());

            if response.tool_calls.is_empty() {
                return Ok(response);
            }

            let outcomes = self.execute_tool_calls(response.tool_calls).await?;
            for outcome in outcomes {
                messages.push(tool_result_message(&outcome.call, &outcome.result));
            }
        }

        Err(XlaiError::new(
            ErrorKind::Tool,
            "chat exceeded the maximum number of tool round trips",
        ))
    }

    #[must_use]
    pub fn stream_prompt(
        &self,
        content: impl Into<String>,
    ) -> BoxStream<'static, Result<ChatExecutionEvent, XlaiError>> {
        self.stream(vec![ChatMessage {
            role: MessageRole::User,
            content: content.into(),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }])
    }

    #[must_use]
    pub fn stream(
        &self,
        mut messages: Vec<ChatMessage>,
    ) -> BoxStream<'static, Result<ChatExecutionEvent, XlaiError>> {
        let runtime = Arc::clone(&self.runtime);
        let model = self.model.clone();
        let system_prompt = self.system_prompt.clone();
        let temperature = self.temperature;
        let max_output_tokens = self.max_output_tokens;
        let max_tool_round_trips = self.max_tool_round_trips;
        let tools = self.tools.clone();

        Box::pin(try_stream! {
            for _ in 0..max_tool_round_trips {
                let request = ChatRequest {
                    model: model.clone(),
                    system_prompt: system_prompt.clone(),
                    messages: messages.clone(),
                    available_tools: tools
                        .values()
                        .map(|tool| tool.definition.clone())
                        .collect(),
                    metadata: BTreeMap::new(),
                    temperature,
                    max_output_tokens,
                };

                let mut response = None;
                let mut model_stream = runtime.stream_chat(request)?;

                while let Some(chunk) = model_stream.next().await {
                    let chunk = chunk?;
                    if let ChatChunk::Finished(final_response) = &chunk {
                        response = Some(final_response.clone());
                    }
                    yield ChatExecutionEvent::Model(chunk);
                }

                let response = response.ok_or_else(|| {
                    XlaiError::new(
                        ErrorKind::Provider,
                        "stream ended without a final chat response",
                    )
                })?;
                messages.push(response.message.clone());

                if response.tool_calls.is_empty() {
                    return;
                }

                for call in &response.tool_calls {
                    yield ChatExecutionEvent::ToolCall(call.clone());
                }

                let outcomes = execute_tool_calls_from(
                    runtime.as_ref(),
                    &tools,
                    response.tool_calls,
                )
                .await?;

                for outcome in outcomes {
                    messages.push(tool_result_message(&outcome.call, &outcome.result));
                    yield ChatExecutionEvent::ToolResult(outcome.result);
                }
            }

            Err(XlaiError::new(
                ErrorKind::Tool,
                "chat exceeded the maximum number of tool round trips",
            ))?;
        })
    }

    fn build_request(&self, messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest {
            model: self.model.clone(),
            system_prompt: self.system_prompt.clone(),
            messages,
            available_tools: self.tool_definitions(),
            metadata: BTreeMap::new(),
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
        }
    }

    async fn execute_tool_calls(
        &self,
        calls: Vec<ToolCall>,
    ) -> Result<Vec<ToolExecutionOutcome>, XlaiError> {
        execute_tool_calls_from(self.runtime.as_ref(), &self.tools, calls).await
    }
}

fn tool_result_message(call: &ToolCall, result: &ToolResult) -> ChatMessage {
    ChatMessage {
        role: MessageRole::Tool,
        content: result.content.clone(),
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
