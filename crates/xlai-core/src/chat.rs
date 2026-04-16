use std::collections::BTreeMap;

use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::content::{ChatContent, MessageRole, StreamTextDelta};
use crate::error::XlaiError;
use crate::execution::{CancellationSignal, ChatExecutionConfig};
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

pub const XLAI_ASSISTANT_TOOL_CALLS_METADATA_KEY: &str = "xlai_assistant_tool_calls";

impl ChatMessage {
    #[must_use]
    pub fn with_assistant_tool_calls(mut self, tool_calls: &[ToolCall]) -> Self {
        if tool_calls.is_empty() {
            self.metadata.remove(XLAI_ASSISTANT_TOOL_CALLS_METADATA_KEY);
        } else {
            self.metadata.insert(
                XLAI_ASSISTANT_TOOL_CALLS_METADATA_KEY.to_owned(),
                json!(tool_calls),
            );
        }
        self
    }

    #[must_use]
    pub fn assistant_tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.metadata
            .get(XLAI_ASSISTANT_TOOL_CALLS_METADATA_KEY)
            .and_then(|value| serde_json::from_value(value.clone()).ok())
    }
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

/// Recursive tool input schema used by providers that accept JSON Schema-like tool definitions.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolSchema {
    #[serde(flatten)]
    pub kind: ToolSchemaKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolSchemaKind {
    String,
    Number,
    Integer,
    Boolean,
    Array {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        items: Option<Box<ToolSchema>>,
    },
    Object {
        #[serde(default)]
        properties: BTreeMap<String, ToolSchema>,
        #[serde(default)]
        required: Vec<String>,
        #[serde(default, rename = "additionalProperties")]
        additional_properties: bool,
    },
}

impl ToolSchema {
    #[must_use]
    pub fn string() -> Self {
        Self {
            kind: ToolSchemaKind::String,
            description: None,
        }
    }

    #[must_use]
    pub fn number() -> Self {
        Self {
            kind: ToolSchemaKind::Number,
            description: None,
        }
    }

    #[must_use]
    pub fn integer() -> Self {
        Self {
            kind: ToolSchemaKind::Integer,
            description: None,
        }
    }

    #[must_use]
    pub fn boolean() -> Self {
        Self {
            kind: ToolSchemaKind::Boolean,
            description: None,
        }
    }

    #[must_use]
    pub fn array(items: Option<ToolSchema>) -> Self {
        Self {
            kind: ToolSchemaKind::Array {
                items: items.map(Box::new),
            },
            description: None,
        }
    }

    #[must_use]
    pub fn object(properties: BTreeMap<String, ToolSchema>, required: Vec<String>) -> Self {
        Self {
            kind: ToolSchemaKind::Object {
                properties,
                required,
                additional_properties: false,
            },
            description: None,
        }
    }

    #[must_use]
    pub fn object_with_additional_properties(
        properties: BTreeMap<String, ToolSchema>,
        required: Vec<String>,
        additional_properties: bool,
    ) -> Self {
        Self {
            kind: ToolSchemaKind::Object {
                properties,
                required,
                additional_properties,
            },
            description: None,
        }
    }

    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    #[must_use]
    pub fn from_parameter_kind(kind: ToolParameterType) -> Self {
        match kind {
            ToolParameterType::String => Self::string(),
            ToolParameterType::Number => Self::number(),
            ToolParameterType::Integer => Self::integer(),
            ToolParameterType::Boolean => Self::boolean(),
            ToolParameterType::Array => Self::array(None),
            ToolParameterType::Object => Self::object(BTreeMap::new(), Vec::new()),
        }
    }

    #[must_use]
    pub fn from_parameters(parameters: &[ToolParameter]) -> Self {
        let mut properties = BTreeMap::new();
        let mut required = Vec::new();

        for parameter in parameters {
            let property = Self::from_parameter_kind(parameter.kind)
                .with_description(parameter.description.clone());
            properties.insert(parameter.name.clone(), property);
            if parameter.required {
                required.push(parameter.name.clone());
            }
        }

        Self::object(properties, required)
    }

    #[must_use]
    pub fn json_schema(&self) -> Value {
        match &self.kind {
            ToolSchemaKind::String => {
                schema_with_description("string", self.description.as_deref())
            }
            ToolSchemaKind::Number => {
                schema_with_description("number", self.description.as_deref())
            }
            ToolSchemaKind::Integer => {
                schema_with_description("integer", self.description.as_deref())
            }
            ToolSchemaKind::Boolean => {
                schema_with_description("boolean", self.description.as_deref())
            }
            ToolSchemaKind::Array { items } => {
                let mut schema = serde_json::json!({
                    "type": "array",
                });
                if let Some(description) = &self.description {
                    schema["description"] = Value::String(description.clone());
                }
                if let Some(items) = items {
                    schema["items"] = items.json_schema();
                }
                schema
            }
            ToolSchemaKind::Object {
                properties,
                required,
                additional_properties,
            } => {
                let mut properties_json = serde_json::Map::new();
                for (name, schema) in properties {
                    properties_json.insert(name.clone(), schema.json_schema());
                }
                let mut schema = serde_json::json!({
                    "type": "object",
                    "properties": properties_json,
                    "required": required,
                    "additionalProperties": additional_properties,
                });
                if let Some(description) = &self.description {
                    schema["description"] = Value::String(description.clone());
                }
                schema
            }
        }
    }
}

fn schema_with_description(kind: &'static str, description: Option<&str>) -> Value {
    let mut schema = serde_json::json!({ "type": kind });
    if let Some(description) = description {
        schema["description"] = Value::String(description.to_owned());
    }
    schema
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<ToolSchema>,
    /// Legacy flat parameter model retained during the nested-schema migration.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameters: Vec<ToolParameter>,
    #[serde(default)]
    pub execution_mode: ToolCallExecutionMode,
}

impl ToolDefinition {
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: ToolSchema,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: Some(input_schema),
            parameters: Vec::new(),
            execution_mode: ToolCallExecutionMode::Concurrent,
        }
    }

    #[must_use]
    pub fn from_parameters(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Vec<ToolParameter>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: Some(ToolSchema::from_parameters(&parameters)),
            parameters,
            execution_mode: ToolCallExecutionMode::Concurrent,
        }
    }

    #[must_use]
    pub fn with_execution_mode(mut self, execution_mode: ToolCallExecutionMode) -> Self {
        self.execution_mode = execution_mode;
        self
    }

    #[must_use]
    pub fn with_input_schema(mut self, input_schema: ToolSchema) -> Self {
        self.input_schema = Some(input_schema);
        self.parameters.clear();
        self
    }

    #[must_use]
    pub fn resolved_input_schema(&self) -> ToolSchema {
        self.input_schema
            .clone()
            .unwrap_or_else(|| ToolSchema::from_parameters(&self.parameters))
    }
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

/// Advisory reasoning budget hint for providers that support explicit effort control.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
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

/// Optional auto-retry hints for chat requests.
///
/// Backends interpret this struct; unsupported providers ignore it. Fields are advisory: each
/// backend may clamp attempts, backoff, or disable retry for streaming mid-flight.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatRetryPolicy {
    /// When `false`, backends should not perform automatic retries for this request.
    #[serde(default = "default_chat_retry_enabled")]
    pub enabled: bool,
    /// Number of **additional** attempts after the first failure (0 = at most one try).
    #[serde(default = "default_chat_retry_max_retries")]
    pub max_retries: u32,
    /// Initial delay before the first retry, in milliseconds.
    #[serde(default = "default_chat_retry_initial_backoff_ms")]
    pub initial_backoff_ms: u64,
    /// Maximum delay between retries, in milliseconds (exponential backoff caps here).
    #[serde(default = "default_chat_retry_max_backoff_ms")]
    pub max_backoff_ms: u64,
}

fn default_chat_retry_enabled() -> bool {
    true
}

fn default_chat_retry_max_retries() -> u32 {
    2
}

fn default_chat_retry_initial_backoff_ms() -> u64 {
    200
}

fn default_chat_retry_max_backoff_ms() -> u64 {
    10_000
}

impl Default for ChatRetryPolicy {
    fn default() -> Self {
        Self {
            enabled: default_chat_retry_enabled(),
            max_retries: default_chat_retry_max_retries(),
            initial_backoff_ms: default_chat_retry_initial_backoff_ms(),
            max_backoff_ms: default_chat_retry_max_backoff_ms(),
        }
    }
}

impl ChatRetryPolicy {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    #[must_use]
    pub fn with_initial_backoff_ms(mut self, ms: u64) -> Self {
        self.initial_backoff_ms = ms;
        self
    }

    #[must_use]
    pub fn with_max_backoff_ms(mut self, ms: u64) -> Self {
        self.max_backoff_ms = ms;
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_policy: Option<ChatRetryPolicy>,
    /// Advisory execution hints merged from runtime/session/request layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ChatExecutionConfig>,
    /// In-process cancellation; omitted from JSON and CBOR wire formats.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub cancellation: Option<CancellationSignal>,
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
        message_index: usize,
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

    /// Best-effort warmup for local providers (e.g. load weights). Default is a no-op.
    fn warmup(&self) -> BoxFuture<'_, Result<(), XlaiError>> {
        Box::pin(async { Ok(()) })
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
