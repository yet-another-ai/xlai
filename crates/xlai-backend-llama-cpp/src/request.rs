use xlai_core::{
    ChatMessage, ChatRequest, ContentPart, ErrorKind, MessageRole, StructuredOutput,
    ToolDefinition, XlaiError,
};

use crate::LlamaCppConfig;
use crate::prompt::validate_structured_output_schema;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PromptMessage {
    pub(crate) role: PromptRole,
    pub(crate) content: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PromptRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedRequest {
    pub(crate) messages: Vec<PromptMessage>,
    pub(crate) available_tools: Vec<ToolDefinition>,
    pub(crate) structured_output: Option<StructuredOutput>,
    pub(crate) temperature: f32,
    pub(crate) max_output_tokens: u32,
}

impl PreparedRequest {
    pub(crate) fn from_core_request(
        config: &LlamaCppConfig,
        request: ChatRequest,
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

        let requested_model = request.model;
        if let Some(model_name) = requested_model.as_deref() {
            let expected = config.resolved_model_name();
            if model_name != expected {
                return Err(XlaiError::new(
                    ErrorKind::Validation,
                    format!(
                        "chat request targeted model `{model_name}`, but this llama.cpp backend is configured for `{expected}`"
                    ),
                ));
            }
        }

        Ok(Self {
            messages,
            available_tools: request.available_tools,
            structured_output: request.structured_output,
            temperature: request.temperature.unwrap_or(config.temperature),
            max_output_tokens: request
                .max_output_tokens
                .unwrap_or(config.max_output_tokens),
        })
    }

    pub(crate) fn validate_against(&self, config: &LlamaCppConfig) -> Result<(), XlaiError> {
        if self.messages.is_empty() {
            return Err(XlaiError::new(
                ErrorKind::Validation,
                "llama.cpp chat requests must contain at least one message or system prompt",
            ));
        }

        if self.max_output_tokens == 0 {
            return Err(XlaiError::new(
                ErrorKind::Validation,
                "llama.cpp max_output_tokens must be greater than zero",
            ));
        }

        if let Some(structured_output) = &self.structured_output {
            validate_structured_output_schema(structured_output)?;
        }

        if self.structured_output.is_some() && !self.available_tools.is_empty() {
            return Err(XlaiError::new(
                ErrorKind::Unsupported,
                "llama.cpp does not currently support combining structured output with tool calling in the same request",
            ));
        }

        if config.n_gpu_layers > 0 && !xlai_llama_cpp_sys::supports_gpu_offload() {
            return Err(XlaiError::new(
                ErrorKind::Unsupported,
                "this xlai llama.cpp build was compiled without GPU offload support",
            ));
        }

        Ok(())
    }
}

impl PromptRole {
    pub(crate) fn from_message_role(role: MessageRole) -> Result<Self, XlaiError> {
        match role {
            MessageRole::System => Ok(Self::System),
            MessageRole::User => Ok(Self::User),
            MessageRole::Assistant => Ok(Self::Assistant),
            MessageRole::Tool => Ok(Self::Tool),
        }
    }

    #[must_use]
    pub(crate) fn as_template_role(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }

    #[must_use]
    pub(crate) fn as_manual_label(self) -> &'static str {
        match self {
            Self::System => "System",
            Self::User => "User",
            Self::Assistant => "Assistant",
            Self::Tool => "Tool",
        }
    }
}

pub(crate) fn extract_text_content(message: &ChatMessage) -> Result<String, XlaiError> {
    let mut text = String::new();

    for part in &message.content.parts {
        match part {
            ContentPart::Text { text: part_text } => text.push_str(part_text),
            _ => {
                return Err(XlaiError::new(
                    ErrorKind::Unsupported,
                    format!(
                        "llama.cpp currently supports text-only chat content; message role {:?} contained a non-text part",
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
