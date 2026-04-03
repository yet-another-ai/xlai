use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::content::{ChatContent, MessageRole, StreamTextDelta};
use crate::error::XlaiError;
use crate::metadata::Metadata;
use crate::runtime::{BoxFuture, BoxStream, RuntimeBound};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: ChatContent,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolParameterType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolParameter {
    pub name: String,
    pub description: String,
    pub kind: ToolParameterType,
    pub required: bool,
}

/// How tool calls from a single model turn are executed relative to each other.
///
/// When any invoked tool in a turn is marked [`Sequential`](Self::Sequential), the
/// runtime runs **all** tool calls in that turn sequentially in model order (no overlap
/// with any other tool call in the same turn).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolCallExecutionMode {
    #[default]
    Concurrent,
    Sequential,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
    #[serde(default)]
    pub execution_mode: ToolCallExecutionMode,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub tool_name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub tool_name: String,
    pub content: String,
    pub is_error: bool,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FinishReason {
    Completed,
    ToolCalls,
    Length,
    Stopped,
}

/// Structured output request for providers that can honor it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StructuredOutputFormat {
    JsonSchema { schema: Value },
    LarkGrammar { grammar: String },
}

/// Structured output request for providers that can honor it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StructuredOutput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(flatten)]
    pub format: StructuredOutputFormat,
}

impl StructuredOutput {
    #[must_use]
    pub fn json_schema(schema: Value) -> Self {
        Self {
            name: None,
            description: None,
            format: StructuredOutputFormat::JsonSchema { schema },
        }
    }

    #[must_use]
    pub fn lark_grammar(grammar: impl Into<String>) -> Self {
        Self {
            name: None,
            description: None,
            format: StructuredOutputFormat::LarkGrammar {
                grammar: grammar.into(),
            },
        }
    }

    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub available_tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_output: Option<StructuredOutput>,
    #[serde(default)]
    pub metadata: Metadata,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatResponse {
    pub message: ChatMessage,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<TokenUsage>,
    pub finish_reason: FinishReason,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallChunk {
    pub index: usize,
    pub id: Option<String>,
    pub tool_name: Option<String>,
    pub arguments_delta: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ChatChunk {
    MessageStart {
        role: MessageRole,
    },
    /// Incremental text for multimodal part at `part_index` (usually `0` for plain assistant streams).
    ContentDelta(StreamTextDelta),
    ToolCallDelta(ToolCallChunk),
    Finished(ChatResponse),
}

pub trait ChatModel: RuntimeBound {
    fn provider_name(&self) -> &'static str;

    fn generate(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, XlaiError>>;

    fn generate_stream(&self, request: ChatRequest) -> BoxStream<'_, Result<ChatChunk, XlaiError>> {
        Box::pin(stream::once(async move {
            self.generate(request).await.map(ChatChunk::Finished)
        }))
    }
}

pub trait ChatBackend {
    type Model: ChatModel + 'static;

    fn into_chat_model(self) -> Self::Model;
}

impl<T> ChatBackend for T
where
    T: ChatModel + 'static,
{
    type Model = T;

    fn into_chat_model(self) -> Self::Model {
        self
    }
}
