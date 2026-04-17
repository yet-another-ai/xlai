use xlai_core::{
    CancellationSignal, ChatExecutionConfig, ChatMessage, ChatRequest, ContentPart, ErrorKind,
    MessageRole, StructuredOutput, ToolDefinition, XlaiError,
};

use super::schema::validate_structured_output_schema;

/// Options for turning a [`ChatRequest`] into a text-only local prompt.
#[derive(Clone, Debug)]
pub struct LocalChatPrepareOptions {
    pub default_temperature: f32,
    pub default_max_output_tokens: u32,
    /// When set, a non-empty `ChatRequest::model` must match this string exactly.
    pub expected_model_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptMessage {
    pub role: PromptRole,
    pub content: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug)]
pub struct PreparedLocalChatRequest {
    pub messages: Vec<PromptMessage>,
    pub available_tools: Vec<ToolDefinition>,
    pub structured_output: Option<StructuredOutput>,
    pub temperature: f32,
    pub max_output_tokens: u32,
    /// Merged advisory execution hints from the original [`ChatRequest`].
    pub execution: Option<ChatExecutionConfig>,
    /// Cooperative cancellation for in-process local generation.
    pub cancellation: Option<CancellationSignal>,
}

impl PreparedLocalChatRequest {
    /// Builds a text-only local chat request from a core [`ChatRequest`].
    ///
    /// # Errors
    ///
    /// Returns an error on model mismatch (when configured), non-text content, or invalid state.
    pub fn from_chat_request(
        request: ChatRequest,
        options: &LocalChatPrepareOptions,
    ) -> Result<Self, XlaiError> {
        let mut messages = Vec::new();

        if let Some(system_prompt) = request.system_prompt {
            let system_prompt = system_prompt.trim();
            if !system_prompt.is_empty() {
                messages.push(PromptMessage {
                    role: PromptRole::System,
                    content: system_prompt.to_owned(),
                });
            }
        }

        for message in request.messages {
            messages.push(PromptMessage {
                role: PromptRole::from_message_role(message.role)?,
                content: extract_text_content(&message)?,
            });
        }

        if let Some(model_name) = request.model.as_deref()
            && let Some(expected) = options.expected_model_name.as_deref()
            && model_name != expected
        {
            return Err(XlaiError::new(
                ErrorKind::Validation,
                format!(
                    "chat request targeted model `{model_name}`, but this backend is configured for `{expected}`"
                ),
            ));
        }

        Ok(Self {
            messages,
            available_tools: request.available_tools,
            structured_output: request.structured_output,
            temperature: request.temperature.unwrap_or(options.default_temperature),
            max_output_tokens: request
                .max_output_tokens
                .unwrap_or(options.default_max_output_tokens),
            execution: request.execution,
            cancellation: request.cancellation,
        })
    }

    /// Validates request invariants shared by local backends.
    ///
    /// # Errors
    ///
    /// Returns an error when messages are empty, token limits are invalid, schemas are invalid,
    /// or structured output is combined with tools.
    pub fn validate_common(&self) -> Result<(), XlaiError> {
        if self.messages.is_empty() {
            return Err(XlaiError::new(
                ErrorKind::Validation,
                "chat requests must contain at least one message or system prompt",
            ));
        }

        if self.max_output_tokens == 0 {
            return Err(XlaiError::new(
                ErrorKind::Validation,
                "max_output_tokens must be greater than zero",
            ));
        }

        if let Some(structured_output) = &self.structured_output {
            validate_structured_output_schema(structured_output)?;
        }

        if self.structured_output.is_some() && !self.available_tools.is_empty() {
            return Err(XlaiError::new(
                ErrorKind::Unsupported,
                "structured output cannot be combined with tool calling in the same request",
            ));
        }

        Ok(())
    }
}

impl PromptRole {
    pub fn from_message_role(role: MessageRole) -> Result<Self, XlaiError> {
        match role {
            MessageRole::System => Ok(Self::System),
            MessageRole::User => Ok(Self::User),
            MessageRole::Assistant => Ok(Self::Assistant),
            MessageRole::Tool => Ok(Self::Tool),
        }
    }

    #[must_use]
    pub fn as_template_role(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }

    #[must_use]
    pub fn as_manual_label(self) -> &'static str {
        match self {
            Self::System => "System",
            Self::User => "User",
            Self::Assistant => "Assistant",
            Self::Tool => "Tool",
        }
    }
}

/// Extracts plain text from a [`ChatMessage`], rejecting non-text parts.
///
/// # Errors
///
/// Returns an error when a non-text [`ContentPart`] is present.
pub fn extract_text_content(message: &ChatMessage) -> Result<String, XlaiError> {
    let mut text = String::new();

    for part in &message.content.parts {
        match part {
            ContentPart::Text { text: part_text } => text.push_str(part_text),
            _ => {
                return Err(XlaiError::new(
                    ErrorKind::Unsupported,
                    format!(
                        "local chat backends support text-only content; message role {:?} contained a non-text part",
                        message.role
                    ),
                ));
            }
        }
    }

    if message.role == MessageRole::Tool {
        let mut content = String::new();
        if let Some(tool_name) = message.tool_name.as_deref() {
            content.push_str("Tool: ");
            content.push_str(tool_name);
        } else {
            content.push_str("Tool result");
        }
        if let Some(tool_call_id) = message.tool_call_id.as_deref() {
            content.push_str("\nCall ID: ");
            content.push_str(tool_call_id);
        }
        content.push_str("\nResult:\n");
        content.push_str(&text);
        return Ok(content);
    }

    Ok(text)
}
