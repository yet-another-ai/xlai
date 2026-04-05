use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

mod builtin;
mod mcp;

use async_stream::try_stream;
use futures_util::StreamExt;
use serde_json::Value;
use tera::Context;
use xlai_core::{
    BoxFuture, BoxStream, ChatChunk, ChatContent, ChatMessage, ChatResponse, ContentPart,
    ErrorKind, MaybeSend, MessageRole, RuntimeBound, StructuredOutput, ToolDefinition, ToolResult,
    XlaiError,
};

use crate::chat::Chat;
use crate::{ChatExecutionEvent, XlaiRuntime};

pub use mcp::McpRegistry;

#[cfg(not(target_arch = "wasm32"))]
type ContextCompressorFn = dyn Fn(Vec<ChatMessage>, Option<u32>) -> BoxFuture<'static, Result<Vec<ChatMessage>, XlaiError>>
    + Send
    + Sync;
#[cfg(target_arch = "wasm32")]
type ContextCompressorFn = dyn Fn(
    Vec<ChatMessage>,
    Option<u32>,
) -> BoxFuture<'static, Result<Vec<ChatMessage>, XlaiError>>;

/// Runs before each model call in the streaming agent loop (see [`Agent::with_context_compressor`]).
async fn prepare_messages_for_agent_round(
    chat: &Chat,
    messages: &[ChatMessage],
    compressor: &Option<Arc<ContextCompressorFn>>,
) -> Result<Vec<ChatMessage>, XlaiError> {
    let Some(compressor) = compressor.as_ref() else {
        return Ok(messages.to_vec());
    };

    let estimated_input_tokens = chat.estimate_input_tokens_for_messages(messages);
    let rewritten = (compressor)(messages.to_vec(), estimated_input_tokens).await?;

    if rewritten.is_empty() {
        return Err(XlaiError::new(
            ErrorKind::Provider,
            "context compressor returned an empty message list",
        ));
    }

    Ok(rewritten)
}

/// High-level agent session: multi-round tool execution runs only on **streaming** APIs when
/// the agent loop is enabled (see [`Self::with_agent_loop_enabled`]). Unary [`Self::execute`] / [`Self::prompt`] always issue
/// a single model call (no tool callbacks), so callers are not blocked for arbitrarily long runs.
/// [`Chat`] also performs only single model calls.
///
/// Optionally, [`Self::with_context_compressor`] can rewrite the message list before each looped
/// model call (streaming + agent loop only).
#[derive(Clone)]
pub struct Agent {
    chat: Chat,
    agent_loop_enabled: bool,
    max_tool_round_trips: usize,
    context_compressor: Option<Arc<ContextCompressorFn>>,
}

impl Agent {
    /// Creates an agent session and wires in built-in tools for the runtime.
    ///
    /// # Errors
    ///
    /// Returns an error if a built-in tool cannot be configured.
    pub fn new(runtime: Arc<XlaiRuntime>) -> Result<Self, XlaiError> {
        let mut chat = Chat::new(runtime.clone());
        builtin::register_builtin_tools(&mut chat, runtime)?;
        Ok(Self {
            chat,
            agent_loop_enabled: true,
            max_tool_round_trips: 8,
            context_compressor: None,
        })
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.chat = self.chat.with_model(model);
        self
    }

    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.chat = self.chat.with_system_prompt(system_prompt);
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
        self.chat = self.chat.with_system_prompt_asset(prompt_name)?;
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
        self.chat = self
            .chat
            .with_system_prompt_template(prompt_name, context)?;
        Ok(self)
    }

    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.chat = self.chat.with_temperature(temperature);
        self
    }

    #[must_use]
    pub fn with_max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.chat = self.chat.with_max_output_tokens(max_output_tokens);
        self
    }

    #[must_use]
    pub fn with_structured_output(mut self, structured_output: StructuredOutput) -> Self {
        self.chat = self.chat.with_structured_output(structured_output);
        self
    }

    #[must_use]
    pub fn with_max_tool_round_trips(mut self, max_tool_round_trips: usize) -> Self {
        self.max_tool_round_trips = max_tool_round_trips;
        self
    }

    /// Controls the multi-round agent loop on **streaming** APIs only ([`Self::stream`],
    /// [`Self::stream_prompt`], etc.).
    ///
    /// When `false`, those streams perform a single model turn and do not run tool callbacks.
    ///
    /// When `true` (default), streams repeat model → tools → model until there are no tool calls
    /// (bounded by [`Self::with_max_tool_round_trips`]).
    ///
    /// [`Self::execute`] and [`Self::prompt`] always perform exactly one model call and never run
    /// tool callbacks here (use streaming if you need the agent tool loop).
    #[must_use]
    pub fn with_agent_loop_enabled(mut self, agent_loop_enabled: bool) -> Self {
        self.agent_loop_enabled = agent_loop_enabled;
        self
    }

    /// Registers an async hook that runs **before each model call** in the streaming agent loop
    /// ([`Self::stream`], [`Self::stream_prompt`], etc.), when [`Self::with_agent_loop_enabled`] is
    /// `true`.
    ///
    /// The callback receives the full accumulated [`ChatMessage`] history for this stream and a
    /// best-effort [`Option<u32>`] input-token estimate derived from JSON-serialized request size
    /// (bytes÷4 heuristic; not tokenizer-accurate).
    /// It should return the message list to send for that round (often a compressed or summarized
    /// copy). The in-memory history used for the next round still grows with assistant and tool
    /// messages as usual; only the outgoing request for that step uses the returned list.
    ///
    /// If the hook returns an empty vector, the stream fails with [`ErrorKind::Provider`].
    ///
    /// Unary [`Self::prompt`] / [`Self::execute`] do not invoke this hook. When the agent loop is
    /// disabled on streaming, [`Self::stream`] delegates to [`Chat::stream`] and the hook is not
    /// used.
    pub fn register_context_compressor<F, Fut>(&mut self, f: F) -> &mut Self
    where
        F: Fn(Vec<ChatMessage>, Option<u32>) -> Fut + RuntimeBound + 'static,
        Fut: Future<Output = Result<Vec<ChatMessage>, XlaiError>> + MaybeSend + 'static,
    {
        self.context_compressor = Some(Arc::new(
            move |messages,
                  estimated_input_tokens|
                  -> BoxFuture<'static, Result<Vec<ChatMessage>, XlaiError>> {
                Box::pin(f(messages, estimated_input_tokens))
            },
        ));
        self
    }

    /// See [`Self::register_context_compressor`].
    #[must_use]
    pub fn with_context_compressor<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(Vec<ChatMessage>, Option<u32>) -> Fut + RuntimeBound + 'static,
        Fut: Future<Output = Result<Vec<ChatMessage>, XlaiError>> + MaybeSend + 'static,
    {
        self.register_context_compressor(f);
        self
    }

    #[must_use]
    pub fn mcp_registry(&mut self) -> McpRegistry<'_> {
        McpRegistry::new(&mut self.chat)
    }

    /// Registers an MCP tool directly on the agent.
    ///
    /// This is shorthand for `agent.mcp_registry().register_tool(...)`.
    pub fn register_tool<F, Fut>(&mut self, definition: ToolDefinition, callback: F) -> &mut Self
    where
        F: Fn(Value) -> Fut + xlai_core::RuntimeBound + 'static,
        Fut: std::future::Future<Output = Result<ToolResult, XlaiError>>
            + xlai_core::MaybeSend
            + 'static,
    {
        self.mcp_registry().register_tool(definition, callback);
        self
    }

    #[must_use]
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.chat.tool_definitions()
    }

    #[must_use]
    pub fn chat(&self) -> &Chat {
        &self.chat
    }

    #[must_use]
    pub fn chat_mut(&mut self) -> &mut Chat {
        &mut self.chat
    }

    #[must_use]
    pub fn into_chat(self) -> Chat {
        self.chat
    }

    /// Sends a single user prompt (one model call; no automatic tool execution).
    ///
    /// # Errors
    ///
    /// Returns an error if the configured model request fails.
    pub async fn prompt(&self, content: impl Into<String>) -> Result<ChatResponse, XlaiError> {
        self.execute(vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::text(content.into()),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }])
        .await
    }

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

    pub async fn prompt_parts(&self, parts: Vec<ContentPart>) -> Result<ChatResponse, XlaiError> {
        self.execute(vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::from_parts(parts),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }])
        .await
    }

    /// Runs exactly one model call with the given message history (no tool execution).
    ///
    /// For multi-round tool execution with incremental output, use [`Self::stream`] instead.
    ///
    /// # Errors
    ///
    /// Returns an error if the configured model request fails.
    pub async fn execute(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse, XlaiError> {
        self.chat.execute(messages).await
    }

    #[must_use]
    pub fn stream_prompt(
        &self,
        content: impl Into<String>,
    ) -> BoxStream<'static, Result<ChatExecutionEvent, XlaiError>> {
        self.stream(vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::text(content.into()),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }])
    }

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

    #[must_use]
    pub fn stream_prompt_parts(
        &self,
        parts: Vec<ContentPart>,
    ) -> BoxStream<'static, Result<ChatExecutionEvent, XlaiError>> {
        self.stream(vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::from_parts(parts),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
        }])
    }

    #[must_use]
    pub fn stream(
        &self,
        messages: Vec<ChatMessage>,
    ) -> BoxStream<'static, Result<ChatExecutionEvent, XlaiError>> {
        if !self.agent_loop_enabled {
            return self.chat.stream(messages);
        }

        let chat = self.chat.clone();
        let max = self.max_tool_round_trips;
        let compressor = self.context_compressor.clone();

        Box::pin(try_stream! {
            let mut messages = messages;
            for _ in 0..max {
                let to_send =
                    prepare_messages_for_agent_round(&chat, &messages, &compressor).await?;

                let mut final_response: Option<ChatResponse> = None;
                let mut inner = chat.stream(to_send);
                while let Some(item) = inner.next().await {
                    let item = item?;
                    if let ChatExecutionEvent::Model(ChatChunk::Finished(resp)) = &item {
                        final_response = Some(resp.clone());
                    }
                    yield item;
                }

                let response = final_response.ok_or_else(|| {
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

                let outcomes = chat.dispatch_tool_calls(response.tool_calls).await?;
                for outcome in outcomes {
                    messages.push(Chat::tool_result_as_message(&outcome.call, &outcome.result));
                    yield ChatExecutionEvent::ToolResult(outcome.result);
                }
            }

            Err(XlaiError::new(
                ErrorKind::Tool,
                "agent exceeded the maximum number of tool round trips",
            ))?;
        })
    }
}
