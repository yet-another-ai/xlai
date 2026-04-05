use std::collections::BTreeMap;
use std::sync::Arc;

mod builtin;
mod mcp;

use async_stream::try_stream;
use futures_util::StreamExt;
use serde_json::Value;
use tera::Context;
use xlai_core::{
    BoxStream, ChatChunk, ChatContent, ChatMessage, ChatResponse, ContentPart, ErrorKind,
    MessageRole, StructuredOutput, ToolDefinition, ToolResult, XlaiError,
};

use crate::chat::Chat;
use crate::{ChatExecutionEvent, XlaiRuntime};

pub use mcp::McpRegistry;

/// High-level agent session: multi-round tool execution runs only on **streaming** APIs when
/// the agent loop is enabled (see [`Self::with_agent_loop_enabled`]). Unary [`Self::execute`] / [`Self::prompt`] always issue
/// a single model call (no tool callbacks), so callers are not blocked for arbitrarily long runs.
/// [`Chat`] also performs only single model calls.
#[derive(Clone)]
pub struct Agent {
    chat: Chat,
    agent_loop_enabled: bool,
    max_tool_round_trips: usize,
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

        Box::pin(try_stream! {
            let mut messages = messages;
            for _ in 0..max {
                let mut final_response: Option<ChatResponse> = None;
                let mut inner = chat.stream(messages.clone());
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
