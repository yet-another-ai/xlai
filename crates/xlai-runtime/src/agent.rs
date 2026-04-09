use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

mod builtin;
mod mcp;
mod system_reminder;

use async_stream::try_stream;
use futures_util::StreamExt;
use serde_json::Value;
use tera::Context;
use xlai_core::{
    BoxFuture, BoxStream, ChatChunk, ChatContent, ChatMessage, ChatResponse, ContentPart,
    ErrorKind, MaybeSend, MessageRole, ReasoningEffort, RuntimeBound, StructuredOutput,
    ToolDefinition, ToolResult, XlaiError,
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

#[cfg(not(target_arch = "wasm32"))]
type SystemReminderFn =
    dyn Fn(Vec<ChatMessage>) -> BoxFuture<'static, Result<String, XlaiError>> + Send + Sync;
#[cfg(target_arch = "wasm32")]
type SystemReminderFn = dyn Fn(Vec<ChatMessage>) -> BoxFuture<'static, Result<String, XlaiError>>;

const AGENT_LOOP_CONTINUE_PROMPT: &str = "continue";

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

fn ensure_agent_loop_user_message(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    if messages
        .iter()
        .any(|message| message.role == MessageRole::User)
    {
        return messages;
    }

    let mut messages = messages;
    messages.push(ChatMessage {
        role: MessageRole::User,
        content: ChatContent::text(AGENT_LOOP_CONTINUE_PROMPT),
        tool_name: None,
        tool_call_id: None,
        metadata: BTreeMap::new(),
    });
    messages
}

/// High-level agent session: multi-round tool execution runs only on **streaming** APIs when
/// the agent loop is enabled (see [`Self::with_agent_loop_enabled`]). Unary [`Self::execute`] / [`Self::prompt`] always issue
/// a single model call (no tool callbacks), so callers are not blocked for arbitrarily long runs.
/// [`Chat`] also performs only single model calls.
///
/// Optionally, [`Self::with_context_compressor`] can rewrite the message list before each looped
/// model call (streaming + agent loop only).
///
/// [`Self::register_system_reminder`] injects a composed system reminder before every model call
/// (unary and streaming, including each agent-loop round), placed immediately before the last
/// message in the outgoing request. Reminder rows are **internal**: they are not stored in the
/// agent loop’s growing transcript, are omitted from [`Self::stream`] yields, and any prior
/// reminder-shaped messages are stripped from inbound history before each request so user-managed
/// chat history never accumulates them.
#[derive(Clone)]
pub struct Agent {
    runtime: Arc<XlaiRuntime>,
    chat: Chat,
    agent_loop_enabled: bool,
    max_tool_round_trips: usize,
    context_compressor: Option<Arc<ContextCompressorFn>>,
    system_reminder: Option<Arc<SystemReminderFn>>,
}

impl Agent {
    /// Creates an agent session and wires in built-in tools for the runtime.
    ///
    /// # Errors
    ///
    /// Returns an error if a built-in tool cannot be configured.
    pub fn new(runtime: Arc<XlaiRuntime>) -> Result<Self, XlaiError> {
        let mut chat = Chat::new(runtime.clone());
        builtin::register_builtin_tools(&mut chat, runtime.clone())?;
        Ok(Self {
            runtime,
            chat,
            agent_loop_enabled: true,
            max_tool_round_trips: 8,
            context_compressor: None,
            system_reminder: None,
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
    pub fn with_reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.chat = self.chat.with_reasoning_effort(reasoning_effort);
        self
    }

    #[must_use]
    pub fn with_structured_output(mut self, structured_output: StructuredOutput) -> Self {
        self.chat = self.chat.with_structured_output(structured_output);
        self
    }

    /// See [`Chat::with_retry_policy`].
    #[must_use]
    pub fn with_retry_policy(mut self, retry_policy: Option<xlai_core::ChatRetryPolicy>) -> Self {
        self.chat = self.chat.with_retry_policy(retry_policy);
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

    /// Registers an async hook invoked before **every** model call ([`Self::execute`], [`Self::prompt`],
    /// and [`Self::stream`] / agent-loop rounds).
    ///
    /// The callback receives the current **transcript** for that request (never including internal
    /// reminder rows) and returns additional reminder text. The runtime merges that string with
    /// built-in reminder sections (available and invoked skills when applicable), then inserts one
    /// ephemeral `system` message immediately before the last message in the outgoing list (or
    /// appends if the list is empty). That message is not part of persisted conversation history.
    ///
    /// If the hook returns an empty or whitespace-only string, no user fragment is added; built-in
    /// sections may still produce a reminder. If everything is empty after composition, no extra
    /// message is inserted.
    pub fn register_system_reminder<F, Fut>(&mut self, f: F) -> &mut Self
    where
        F: Fn(Vec<ChatMessage>) -> Fut + RuntimeBound + 'static,
        Fut: Future<Output = Result<String, XlaiError>> + MaybeSend + 'static,
    {
        self.system_reminder = Some(Arc::new(
            move |messages| -> BoxFuture<'static, Result<String, XlaiError>> {
                Box::pin(f(messages))
            },
        ));
        self
    }

    /// See [`Self::register_system_reminder`].
    #[must_use]
    pub fn with_system_reminder<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(Vec<ChatMessage>) -> Fut + RuntimeBound + 'static,
        Fut: Future<Output = Result<String, XlaiError>> + MaybeSend + 'static,
    {
        self.register_system_reminder(f);
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

    async fn prepare_outgoing_with_reminder(
        &self,
        messages: Vec<ChatMessage>,
        ensure_user_message: bool,
    ) -> Result<Vec<ChatMessage>, XlaiError> {
        let messages = system_reminder::strip_internal_reminders(messages);

        let user_fragment = match &self.system_reminder {
            Some(cb) => {
                let s = (cb)(messages.clone()).await?;
                let t = s.trim();
                if t.is_empty() {
                    None
                } else {
                    Some(t.to_owned())
                }
            }
            None => None,
        };

        let body = system_reminder::compose_system_reminder_body(
            self.runtime.as_ref(),
            &messages,
            user_fragment.as_deref(),
        )
        .await?;

        let messages = if ensure_user_message {
            ensure_agent_loop_user_message(messages)
        } else {
            messages
        };

        let Some(body) = body else {
            return Ok(messages);
        };

        Ok(system_reminder::insert_system_reminder_near_tail(
            messages, body,
        ))
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
        let messages = self.prepare_outgoing_with_reminder(messages, false).await?;
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
        let agent = self.clone();

        if !agent.agent_loop_enabled {
            return Box::pin(try_stream! {
                let to_send = agent.prepare_outgoing_with_reminder(messages, false).await?;
                let mut inner = agent.chat.stream(to_send);
                while let Some(item) = inner.next().await {
                    let item = item?;
                    yield item;
                }
            });
        }

        let chat = agent.chat.clone();
        let max = agent.max_tool_round_trips;
        let compressor = agent.context_compressor.clone();

        Box::pin(try_stream! {
            let mut messages = messages;
            for _ in 0..max {
                let mut to_send =
                    prepare_messages_for_agent_round(&chat, &messages, &compressor).await?;
                to_send = agent.prepare_outgoing_with_reminder(to_send, true).await?;

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

                let assistant_message = response
                    .message
                    .clone()
                    .with_assistant_tool_calls(&response.tool_calls);
                messages.push(assistant_message);

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
