use std::sync::Arc;

mod builtin;

use serde_json::Value;
use tera::Context;
use xlai_core::{BoxStream, ChatMessage, ChatResponse, ToolDefinition, ToolResult, XlaiError};

use crate::{Chat, ChatExecutionEvent, ToolCallExecutionMode, XlaiRuntime};

/// High-level agent session API built on top of `Chat`.
#[derive(Clone)]
pub struct Agent {
    chat: Chat,
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
        Ok(Self { chat })
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
    pub fn with_max_tool_round_trips(mut self, max_tool_round_trips: usize) -> Self {
        self.chat = self.chat.with_max_tool_round_trips(max_tool_round_trips);
        self
    }

    #[must_use]
    pub fn with_tool_call_execution_mode(
        mut self,
        tool_call_execution_mode: ToolCallExecutionMode,
    ) -> Self {
        self.chat = self
            .chat
            .with_tool_call_execution_mode(tool_call_execution_mode);
        self
    }

    pub fn register_tool<F, Fut>(&mut self, definition: ToolDefinition, callback: F) -> &mut Self
    where
        F: Fn(Value) -> Fut + xlai_core::RuntimeBound + 'static,
        Fut: std::future::Future<Output = Result<ToolResult, XlaiError>>
            + xlai_core::MaybeSend
            + 'static,
    {
        self.chat.register_tool(definition, callback);
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

    /// Sends a single user prompt through this agent session.
    ///
    /// # Errors
    ///
    /// Returns an error if the configured model request fails, if a tool callback
    /// fails, or if the chat exceeds the maximum number of tool round trips.
    pub async fn prompt(&self, content: impl Into<String>) -> Result<ChatResponse, XlaiError> {
        self.chat.prompt(content).await
    }

    /// Executes a chat turn with the provided message history.
    ///
    /// # Errors
    ///
    /// Returns an error if the configured model request fails, if a tool callback
    /// fails, or if the chat exceeds the maximum number of tool round trips.
    pub async fn execute(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse, XlaiError> {
        self.chat.execute(messages).await
    }

    #[must_use]
    pub fn stream_prompt(
        &self,
        content: impl Into<String>,
    ) -> BoxStream<'static, Result<ChatExecutionEvent, XlaiError>> {
        self.chat.stream_prompt(content)
    }

    #[must_use]
    pub fn stream(
        &self,
        messages: Vec<ChatMessage>,
    ) -> BoxStream<'static, Result<ChatExecutionEvent, XlaiError>> {
        self.chat.stream(messages)
    }
}
