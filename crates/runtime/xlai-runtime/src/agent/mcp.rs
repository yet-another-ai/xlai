use std::future::Future;

use serde_json::Value;
use xlai_core::{MaybeSend, RuntimeBound, ToolDefinition, ToolResult, XlaiError};

use crate::Chat;
use crate::chat::ToolOrigin;

/// Agent-local registry for MCP-provided tools.
pub struct McpRegistry<'a> {
    chat: &'a mut Chat,
}

impl<'a> McpRegistry<'a> {
    pub(super) fn new(chat: &'a mut Chat) -> Self {
        Self { chat }
    }

    pub fn register_tool<F, Fut>(&mut self, definition: ToolDefinition, callback: F) -> &mut Self
    where
        F: Fn(Value) -> Fut + RuntimeBound + 'static,
        Fut: Future<Output = Result<ToolResult, XlaiError>> + MaybeSend + 'static,
    {
        self.chat
            .register_tool_with_origin(definition, ToolOrigin::Mcp, callback);
        self
    }

    #[must_use]
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.chat.tool_definitions_with_origin(ToolOrigin::Mcp)
    }
}
