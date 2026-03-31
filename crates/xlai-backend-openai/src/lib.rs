use async_stream::try_stream;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use xlai_core::{
    BoxFuture, BoxStream, ChatChunk, ChatMessage, ChatModel, ChatRequest, ChatResponse,
    ErrorKind, FinishReason, MessageRole, TokenUsage, ToolCall, ToolCallChunk, ToolDefinition,
    ToolParameterType, XlaiError,
};

#[derive(Clone, Debug)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl OpenAiConfig {
    #[must_use]
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }
}

#[derive(Clone, Debug)]
pub struct OpenAiChatModel {
    client: Client,
    config: OpenAiConfig,
}

impl OpenAiChatModel {
    #[must_use]
    pub fn new(config: OpenAiConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }
}

impl ChatModel for OpenAiChatModel {
    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }

    fn generate<'a>(
        &'a self,
        request: ChatRequest,
    ) -> BoxFuture<'a, Result<ChatResponse, XlaiError>> {
        Box::pin(async move {
            let endpoint = self.config.chat_completions_url();
            let payload = OpenAiChatRequest::from_core_request(&self.config, request, false);

            let response = self
                .client
                .post(endpoint)
                .bearer_auth(&self.config.api_key)
                .json(&payload)
                .send()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            let response = response
                .error_for_status()
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            let response: OpenAiChatResponse = response
                .json()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            response.into_core_response()
        })
    }

    fn generate_stream<'a>(
        &'a self,
        request: ChatRequest,
    ) -> BoxStream<'a, Result<ChatChunk, XlaiError>> {
        Box::pin(try_stream! {
            let endpoint = self.config.chat_completions_url();
            let payload = OpenAiChatRequest::from_core_request(&self.config, request, true);

            let response = self
                .client
                .post(endpoint)
                .bearer_auth(&self.config.api_key)
                .json(&payload)
                .send()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            let response = response
                .error_for_status()
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            let mut bytes_stream = response.bytes_stream();
            let mut parser = SseParser::default();
            let mut state = StreamState::default();
            let mut message_started = false;

            while let Some(chunk) = bytes_stream.next().await {
                let chunk = chunk
                    .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

                for event in parser.push(&chunk) {
                    if event == "[DONE]" {
                        let response = state.into_chat_response()?;
                        yield ChatChunk::Finished(response);
                        return;
                    }

                    let chunk: OpenAiStreamResponse = serde_json::from_str(&event).map_err(|error| {
                        XlaiError::new(
                            ErrorKind::Provider,
                            format!("failed to parse stream event: {error}"),
                        )
                    })?;

                    let choice = chunk.choices.into_iter().next().ok_or_else(|| {
                        XlaiError::new(
                            ErrorKind::Provider,
                            "openai-compatible stream response contained no choices",
                        )
                    })?;

                    if !message_started {
                        message_started = true;
                        yield ChatChunk::MessageStart {
                            role: MessageRole::Assistant,
                        };
                    }

                    if let Some(content) = choice.delta.content {
                        state.message_content.push_str(&content);
                        yield ChatChunk::ContentDelta(content);
                    }

                    if let Some(tool_calls) = choice.delta.tool_calls {
                        for tool_call in tool_calls {
                            let chunk = state.apply_tool_delta(tool_call);
                            yield ChatChunk::ToolCallDelta(chunk);
                        }
                    }

                    if let Some(reason) = choice.finish_reason {
                        state.finish_reason = finish_reason_from_api(Some(reason.as_str()));
                    }
                }
            }

            let response = state.into_chat_response()?;
            yield ChatChunk::Finished(response);
        })
    }
}

#[derive(Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiRequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

impl OpenAiChatRequest {
    fn from_core_request(config: &OpenAiConfig, request: ChatRequest, stream: bool) -> Self {
        let tools = (!request.available_tools.is_empty())
            .then(|| request.available_tools.iter().map(OpenAiTool::from).collect::<Vec<_>>());

        Self {
            model: request.model.unwrap_or_else(|| config.model.clone()),
            messages: request
                .messages
                .into_iter()
                .map(OpenAiRequestMessage::from)
                .collect(),
            temperature: request.temperature,
            max_tokens: request.max_output_tokens,
            tool_choice: tools.as_ref().map(|_| "auto"),
            tools,
            stream: stream.then_some(true),
        }
    }
}

#[derive(Serialize)]
struct OpenAiRequestMessage {
    role: &'static str,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<ChatMessage> for OpenAiRequestMessage {
    fn from(message: ChatMessage) -> Self {
        let role = match message.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };

        Self {
            role,
            content: message.content,
            name: message.tool_name,
            tool_call_id: message.tool_call_id,
        }
    }
}

#[derive(Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunctionDefinition,
}

impl From<&ToolDefinition> for OpenAiTool {
    fn from(tool: &ToolDefinition) -> Self {
        Self {
            kind: "function",
            function: OpenAiFunctionDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool_json_schema(tool),
            },
        }
    }
}

#[derive(Serialize)]
struct OpenAiFunctionDefinition {
    name: String,
    description: String,
    parameters: Value,
}

fn tool_json_schema(tool: &ToolDefinition) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();

    for parameter in &tool.parameters {
        properties.insert(
            parameter.name.clone(),
            serde_json::json!({
                "type": tool_parameter_kind(parameter.kind),
                "description": parameter.description,
            }),
        );

        if parameter.required {
            required.push(parameter.name.clone());
        }
    }

    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

const fn tool_parameter_kind(kind: ToolParameterType) -> &'static str {
    match kind {
        ToolParameterType::String => "string",
        ToolParameterType::Number => "number",
        ToolParameterType::Integer => "integer",
        ToolParameterType::Boolean => "boolean",
        ToolParameterType::Array => "array",
        ToolParameterType::Object => "object",
    }
}

#[derive(Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

impl OpenAiChatResponse {
    fn into_core_response(self) -> Result<ChatResponse, XlaiError> {
        let choice = self.choices.into_iter().next().ok_or_else(|| {
            XlaiError::new(
                ErrorKind::Provider,
                "openai-compatible response contained no choices",
            )
        })?;

        let message = ChatMessage {
            role: MessageRole::Assistant,
            content: choice.message.content.unwrap_or_default(),
            tool_name: None,
            tool_call_id: None,
            metadata: Default::default(),
        };

        let tool_calls = choice
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(ToolCall::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ChatResponse {
            message,
            tool_calls,
            usage: self.usage.map(Into::into),
            finish_reason: finish_reason_from_api(choice.finish_reason.as_deref()),
            metadata: Default::default(),
        })
    }
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Deserialize)]
struct OpenAiToolCall {
    id: String,
    function: OpenAiFunctionCall,
}

impl TryFrom<OpenAiToolCall> for ToolCall {
    type Error = XlaiError;

    fn try_from(value: OpenAiToolCall) -> Result<Self, Self::Error> {
        let arguments = serde_json::from_str(&value.function.arguments).map_err(|error| {
            XlaiError::new(
                ErrorKind::Provider,
                format!("failed to parse tool call arguments: {error}"),
            )
        })?;

        Ok(Self {
            id: value.id,
            tool_name: value.function.name,
            arguments,
        })
    }
}

#[derive(Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl From<OpenAiUsage> for TokenUsage {
    fn from(value: OpenAiUsage) -> Self {
        Self {
            input_tokens: value.prompt_tokens,
            output_tokens: value.completion_tokens,
            total_tokens: value.total_tokens,
        }
    }
}

struct StreamState {
    message_content: String,
    tool_calls: Vec<PartialToolCall>,
    finish_reason: FinishReason,
}

impl Default for StreamState {
    fn default() -> Self {
        Self {
            message_content: String::new(),
            tool_calls: Vec::new(),
            finish_reason: FinishReason::Completed,
        }
    }
}

impl StreamState {
    fn apply_tool_delta(&mut self, delta: OpenAiStreamToolCallDelta) -> ToolCallChunk {
        let index = delta.index.unwrap_or(0);

        while self.tool_calls.len() <= index {
            self.tool_calls.push(PartialToolCall::default());
        }

        let tool_call = &mut self.tool_calls[index];
        if let Some(id) = delta.id {
            tool_call.id = Some(id);
        }

        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                tool_call.tool_name = Some(name);
            }

            if let Some(arguments) = function.arguments {
                tool_call.arguments.push_str(&arguments);
                return ToolCallChunk {
                    index,
                    id: tool_call.id.clone(),
                    tool_name: tool_call.tool_name.clone(),
                    arguments_delta: arguments,
                };
            }
        }

        ToolCallChunk {
            index,
            id: tool_call.id.clone(),
            tool_name: tool_call.tool_name.clone(),
            arguments_delta: String::new(),
        }
    }

    fn into_chat_response(self) -> Result<ChatResponse, XlaiError> {
        let tool_calls = self
            .tool_calls
            .into_iter()
            .enumerate()
            .map(|(index, call)| call.into_tool_call(index))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ChatResponse {
            message: ChatMessage {
                role: MessageRole::Assistant,
                content: self.message_content,
                tool_name: None,
                tool_call_id: None,
                metadata: Default::default(),
            },
            tool_calls,
            usage: None,
            finish_reason: self.finish_reason,
            metadata: Default::default(),
        })
    }
}

#[derive(Default)]
struct PartialToolCall {
    id: Option<String>,
    tool_name: Option<String>,
    arguments: String,
}

impl PartialToolCall {
    fn into_tool_call(self, index: usize) -> Result<ToolCall, XlaiError> {
        let tool_name = self.tool_name.ok_or_else(|| {
            XlaiError::new(
                ErrorKind::Provider,
                format!("streamed tool call at index {index} was missing a name"),
            )
        })?;
        let arguments = serde_json::from_str(&self.arguments).map_err(|error| {
            XlaiError::new(
                ErrorKind::Provider,
                format!("failed to parse streamed tool arguments: {error}"),
            )
        })?;

        Ok(ToolCall {
            id: self.id.unwrap_or_else(|| format!("tool_call_{index}")),
            tool_name,
            arguments,
        })
    }
}

#[derive(Default)]
struct SseParser {
    buffer: String,
}

impl SseParser {
    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));

        let mut events = Vec::new();
        while let Some(index) = self.buffer.find("\n\n") {
            let raw_event = self.buffer[..index].to_owned();
            self.buffer.drain(..index + 2);

            let data = raw_event
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim)
                .collect::<Vec<_>>()
                .join("");

            if !data.is_empty() {
                events.push(data);
            }
        }

        events
    }
}

#[derive(Deserialize)]
struct OpenAiStreamResponse {
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiStreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiStreamToolCallDelta>>,
}

#[derive(Deserialize)]
struct OpenAiStreamToolCallDelta {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiStreamFunctionDelta>,
}

#[derive(Deserialize)]
struct OpenAiStreamFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

fn finish_reason_from_api(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        Some("stop") => FinishReason::Stopped,
        _ => FinishReason::Completed,
    }
}
