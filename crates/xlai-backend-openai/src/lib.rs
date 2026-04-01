use std::collections::BTreeMap;

use async_stream::try_stream;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures_util::StreamExt;
use reqwest::StatusCode;
use reqwest::{
    Client,
    multipart::{Form, Part},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use xlai_core::{
    BoxFuture, BoxStream, ChatBackend, ChatChunk, ChatContent, ChatMessage, ChatModel, ChatRequest,
    ChatResponse, ContentPart, ErrorKind, FinishReason, ImageDetail, MediaSource, MessageRole,
    StreamTextDelta, TokenUsage, ToolCall, ToolCallChunk, ToolDefinition, ToolParameterType,
    TranscriptionBackend, TranscriptionModel, TranscriptionRequest, TranscriptionResponse,
    XlaiError,
};

#[derive(Clone, Debug)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub transcription_model: Option<String>,
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
            transcription_model: None,
        }
    }

    #[must_use]
    pub fn with_transcription_model(mut self, model: impl Into<String>) -> Self {
        self.transcription_model = Some(model.into());
        self
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    fn audio_transcriptions_url(&self) -> String {
        format!(
            "{}/audio/transcriptions",
            self.base_url.trim_end_matches('/')
        )
    }
}

#[derive(Clone, Debug)]
pub struct OpenAiChatModel {
    client: Client,
    config: OpenAiConfig,
}

#[derive(Clone, Debug)]
pub struct OpenAiTranscriptionModel {
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

impl OpenAiTranscriptionModel {
    #[must_use]
    pub fn new(config: OpenAiConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }
}

impl ChatBackend for OpenAiConfig {
    type Model = OpenAiChatModel;

    fn into_chat_model(self) -> Self::Model {
        OpenAiChatModel::new(self)
    }
}

impl TranscriptionBackend for OpenAiConfig {
    type Model = OpenAiTranscriptionModel;

    fn into_transcription_model(self) -> Self::Model {
        OpenAiTranscriptionModel::new(self)
    }
}

impl ChatModel for OpenAiChatModel {
    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }

    fn generate(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, XlaiError>> {
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

            let response = require_success_response(response).await?;

            let response: OpenAiChatResponse = response
                .json()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            response.into_core_response()
        })
    }

    fn generate_stream(&self, request: ChatRequest) -> BoxStream<'_, Result<ChatChunk, XlaiError>> {
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

            let response = require_success_response(response).await?;

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
                        yield ChatChunk::ContentDelta(StreamTextDelta {
                            part_index: 0,
                            delta: content,
                        });
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

impl TranscriptionModel for OpenAiTranscriptionModel {
    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }

    fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> BoxFuture<'_, Result<TranscriptionResponse, XlaiError>> {
        Box::pin(async move {
            let endpoint = self.config.audio_transcriptions_url();
            let payload = OpenAiTranscriptionRequest::from_core_request(&self.config, request)?;

            let response = self
                .client
                .post(endpoint)
                .bearer_auth(&self.config.api_key)
                .multipart(payload.into_multipart_form()?)
                .send()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            let response = require_success_response(response).await?;

            let response: OpenAiTranscriptionResponse = response
                .json()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            Ok(response.into_core_response())
        })
    }
}

async fn require_success_response(
    response: reqwest::Response,
) -> Result<reqwest::Response, XlaiError> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let request_id = response
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("<failed to read provider error body: {error}>"));

    Err(XlaiError::new(
        ErrorKind::Provider,
        format_provider_error_message(status, request_id.as_deref(), &body),
    ))
}

fn format_provider_error_message(
    status: StatusCode,
    request_id: Option<&str>,
    body: &str,
) -> String {
    let body = body.trim();

    let mut message = format!("openai-compatible request failed with {status}");
    if let Some(request_id) = request_id {
        message.push_str(&format!(" (request_id={request_id})"));
    }

    if body.is_empty() {
        return message;
    }

    if let Ok(envelope) = serde_json::from_str::<OpenAiErrorEnvelope>(body) {
        message.push_str(": ");
        message.push_str(&envelope.error.message);

        let mut details = Vec::new();
        if let Some(kind) = envelope.error.kind.as_deref() {
            details.push(format!("type={kind}"));
        }
        if let Some(code) = envelope.error.code.as_deref() {
            details.push(format!("code={code}"));
        }
        if let Some(param) = envelope.error.param.as_deref() {
            details.push(format!("param={param}"));
        }

        if !details.is_empty() {
            message.push_str(" [");
            message.push_str(&details.join(", "));
            message.push(']');
        }

        return message;
    }

    message.push_str(": ");
    message.push_str(body);
    message
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
        let tools = (!request.available_tools.is_empty()).then(|| {
            request
                .available_tools
                .iter()
                .map(OpenAiTool::from)
                .collect::<Vec<_>>()
        });

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

struct OpenAiTranscriptionRequest {
    model: String,
    audio_bytes: Vec<u8>,
    mime_type: String,
    filename: String,
    language: Option<String>,
    prompt: Option<String>,
    temperature: Option<f32>,
}

impl OpenAiTranscriptionRequest {
    fn from_core_request(
        config: &OpenAiConfig,
        request: TranscriptionRequest,
    ) -> Result<Self, XlaiError> {
        let TranscriptionRequest {
            model,
            audio,
            mime_type,
            filename,
            language,
            prompt,
            temperature,
            metadata: _,
        } = request;

        let (source_mime_type, data_base64) = match audio {
            MediaSource::InlineData {
                mime_type,
                data_base64,
            } => (mime_type, data_base64),
            MediaSource::Url { .. } => {
                return Err(XlaiError::new(
                    ErrorKind::Unsupported,
                    "openai-compatible transcription requires inline audio bytes",
                ));
            }
        };

        let audio_bytes = STANDARD.decode(data_base64).map_err(|error| {
            XlaiError::new(
                ErrorKind::Validation,
                format!("transcription audio must be valid base64: {error}"),
            )
        })?;

        Ok(Self {
            model: model
                .or_else(|| config.transcription_model.clone())
                .unwrap_or_else(|| config.model.clone()),
            audio_bytes,
            mime_type: mime_type.unwrap_or(source_mime_type),
            filename: filename.unwrap_or_else(|| "audio".to_owned()),
            language,
            prompt,
            temperature,
        })
    }

    fn into_multipart_form(self) -> Result<Form, XlaiError> {
        let file_part = Part::bytes(self.audio_bytes)
            .file_name(self.filename)
            .mime_str(&self.mime_type)
            .map_err(|error| {
                XlaiError::new(
                    ErrorKind::Validation,
                    format!("invalid transcription MIME type: {error}"),
                )
            })?;

        let mut form = Form::new()
            .part("file", file_part)
            .text("model", self.model)
            .text("response_format", "json");

        if let Some(language) = self.language {
            form = form.text("language", language);
        }
        if let Some(prompt) = self.prompt {
            form = form.text("prompt", prompt);
        }
        if let Some(temperature) = self.temperature {
            form = form.text("temperature", temperature.to_string());
        }

        Ok(form)
    }
}

#[derive(Serialize)]
struct OpenAiRequestMessage {
    role: &'static str,
    content: Value,
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
            content: openai_request_content_value(&message),
            name: message.tool_name,
            tool_call_id: message.tool_call_id,
        }
    }
}

fn openai_request_content_value(message: &ChatMessage) -> Value {
    if message.role == MessageRole::Tool {
        return Value::String(message.content.text_parts_concatenated());
    }
    chat_content_to_openai_request_value(&message.content)
}

fn chat_content_to_openai_request_value(content: &ChatContent) -> Value {
    if let Some(text) = content.as_single_text() {
        return Value::String(text.to_owned());
    }
    let parts: Vec<Value> = content
        .parts
        .iter()
        .map(openai_request_part_value)
        .collect();
    Value::Array(parts)
}

fn openai_request_part_value(part: &ContentPart) -> Value {
    match part {
        ContentPart::Text { text } => serde_json::json!({
            "type": "text",
            "text": text,
        }),
        ContentPart::Image {
            source,
            mime_type,
            detail,
        } => {
            let url = match source {
                MediaSource::Url { url } => url.clone(),
                MediaSource::InlineData {
                    mime_type: inline_mime,
                    data_base64,
                } => {
                    let mime = mime_type.as_deref().unwrap_or(inline_mime.as_str());
                    format!("data:{mime};base64,{data_base64}")
                }
            };
            let mut image_url = serde_json::json!({ "url": url });
            if let Some(d) = detail {
                image_url["detail"] = Value::String(image_detail_openai(*d).to_owned());
            }
            serde_json::json!({
                "type": "image_url",
                "image_url": image_url,
            })
        }
        ContentPart::Audio { source, mime_type } => match source {
            MediaSource::Url { url } => serde_json::json!({
                "type": "text",
                "text": format!("(Attached audio URL: {url})"),
            }),
            MediaSource::InlineData {
                mime_type: inline_mime,
                data_base64,
            } => {
                let mime = mime_type.as_deref().unwrap_or(inline_mime.as_str());
                serde_json::json!({
                    "type": "file",
                    "file": {
                        "filename": "audio",
                        "file_data": format!("data:{mime};base64,{data_base64}"),
                    },
                })
            }
        },
        ContentPart::File {
            source,
            mime_type,
            filename,
        } => match source {
            MediaSource::Url { url } => serde_json::json!({
                "type": "text",
                "text": format!("(Attached file URL: {url})"),
            }),
            MediaSource::InlineData {
                mime_type: inline_mime,
                data_base64,
            } => {
                let mime = mime_type.as_deref().unwrap_or(inline_mime.as_str());
                let fname = filename.clone().unwrap_or_else(|| "attachment".to_owned());
                serde_json::json!({
                    "type": "file",
                    "file": {
                        "filename": fname,
                        "file_data": format!("data:{mime};base64,{data_base64}"),
                    },
                })
            }
        },
    }
}

fn image_detail_openai(detail: ImageDetail) -> &'static str {
    match detail {
        ImageDetail::Auto => "auto",
        ImageDetail::Low => "low",
        ImageDetail::High => "high",
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
            content: openai_response_content_to_chat_content(choice.message.content.as_ref()),
            tool_name: None,
            tool_call_id: None,
            metadata: BTreeMap::new(),
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
            metadata: BTreeMap::new(),
        })
    }
}

#[derive(Deserialize)]
struct OpenAiTranscriptionResponse {
    text: String,
    #[serde(flatten)]
    metadata: BTreeMap<String, Value>,
}

impl OpenAiTranscriptionResponse {
    fn into_core_response(self) -> TranscriptionResponse {
        TranscriptionResponse {
            text: self.text,
            metadata: self.metadata,
        }
    }
}

#[derive(Deserialize)]
struct OpenAiErrorEnvelope {
    error: OpenAiErrorDetail,
}

#[derive(Deserialize)]
struct OpenAiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    param: Option<String>,
    #[serde(default)]
    code: Option<String>,
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
    content: Option<Value>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

fn openai_response_content_to_chat_content(value: Option<&Value>) -> ChatContent {
    match value {
        None | Some(Value::Null) => ChatContent::text(""),
        Some(Value::String(s)) => ChatContent::text(s.clone()),
        Some(Value::Array(items)) => {
            let parts: Vec<ContentPart> = items
                .iter()
                .filter_map(parse_openai_response_content_part)
                .collect();
            if parts.is_empty() {
                ChatContent::text("")
            } else {
                ChatContent::from_parts(parts)
            }
        }
        Some(other) => ChatContent::text(other.to_string()),
    }
}

fn parse_openai_response_content_part(value: &Value) -> Option<ContentPart> {
    let obj = value.as_object()?;
    let ty = obj.get("type")?.as_str()?;
    match ty {
        "text" => {
            let text = obj.get("text")?.as_str()?.to_owned();
            Some(ContentPart::Text { text })
        }
        "image_url" => {
            let url = obj.get("image_url")?.get("url")?.as_str()?.to_owned();
            Some(ContentPart::Image {
                source: MediaSource::Url { url },
                mime_type: None,
                detail: None,
            })
        }
        _ => None,
    }
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
    #[serde(rename = "prompt_tokens")]
    prompt: u32,
    #[serde(rename = "completion_tokens")]
    completion: u32,
    #[serde(rename = "total_tokens")]
    total: u32,
}

impl From<OpenAiUsage> for TokenUsage {
    fn from(value: OpenAiUsage) -> Self {
        Self {
            input_tokens: value.prompt,
            output_tokens: value.completion,
            total_tokens: value.total,
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
                content: ChatContent::text(self.message_content),
                tool_name: None,
                tool_call_id: None,
                metadata: BTreeMap::new(),
            },
            tool_calls,
            usage: None,
            finish_reason: self.finish_reason,
            metadata: BTreeMap::new(),
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

#[cfg(test)]
mod openai_multimodal_tests {
    use std::collections::BTreeMap;

    use serde_json::json;
    use xlai_core::{
        ChatContent, ChatMessage, ChatRequest, ContentPart, MediaSource, MessageRole,
        TranscriptionRequest,
    };

    use super::{OpenAiChatRequest, OpenAiTranscriptionRequest, OpenAiTranscriptionResponse};

    fn test_config() -> super::OpenAiConfig {
        super::OpenAiConfig {
            base_url: "https://api.openai.com/v1".to_owned(),
            api_key: "k".to_owned(),
            model: "gpt-test".to_owned(),
            transcription_model: None,
        }
    }

    #[test]
    fn serializes_multimodal_user_message_as_content_array() {
        let config = test_config();
        let request = ChatRequest {
            model: None,
            system_prompt: None,
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: ChatContent::from_parts(vec![
                    ContentPart::Text {
                        text: "Describe the image.".to_owned(),
                    },
                    ContentPart::Image {
                        source: MediaSource::Url {
                            url: "https://example.com/a.png".to_owned(),
                        },
                        mime_type: None,
                        detail: None,
                    },
                ]),
                tool_name: None,
                tool_call_id: None,
                metadata: BTreeMap::new(),
            }],
            available_tools: Vec::new(),
            metadata: BTreeMap::new(),
            temperature: None,
            max_output_tokens: None,
        };

        let payload = OpenAiChatRequest::from_core_request(&config, request, false);
        let serialized = serde_json::to_value(&payload);
        assert!(serialized.is_ok(), "serialize payload");
        let Ok(v) = serialized else {
            return;
        };
        let content = &v["messages"][0]["content"];
        assert!(content.is_array());
        assert_eq!(content[0]["type"], json!("text"));
        assert_eq!(content[1]["type"], json!("image_url"));
    }

    #[test]
    fn serializes_plain_text_user_message_as_string_content() {
        let config = test_config();
        let request = ChatRequest {
            model: None,
            system_prompt: None,
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: ChatContent::text("hello"),
                tool_name: None,
                tool_call_id: None,
                metadata: BTreeMap::new(),
            }],
            available_tools: Vec::new(),
            metadata: BTreeMap::new(),
            temperature: None,
            max_output_tokens: None,
        };

        let payload = OpenAiChatRequest::from_core_request(&config, request, false);
        let serialized = serde_json::to_value(&payload);
        assert!(serialized.is_ok(), "serialize payload");
        let Ok(v) = serialized else {
            return;
        };
        assert_eq!(v["messages"][0]["content"], json!("hello"));
    }

    #[test]
    fn serializes_inline_audio_as_file_content_part() {
        let config = test_config();
        let request = ChatRequest {
            model: None,
            system_prompt: None,
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: ChatContent::from_parts(vec![ContentPart::Audio {
                    source: MediaSource::InlineData {
                        mime_type: "audio/wav".to_owned(),
                        data_base64: "UklGRg==".to_owned(),
                    },
                    mime_type: Some("audio/wav".to_owned()),
                }]),
                tool_name: None,
                tool_call_id: None,
                metadata: BTreeMap::new(),
            }],
            available_tools: Vec::new(),
            metadata: BTreeMap::new(),
            temperature: None,
            max_output_tokens: None,
        };

        let payload = OpenAiChatRequest::from_core_request(&config, request, false);
        let serialized = serde_json::to_value(&payload);
        assert!(serialized.is_ok(), "serialize payload");
        let Ok(v) = serialized else {
            return;
        };
        assert_eq!(v["messages"][0]["content"][0]["type"], json!("file"));
    }

    #[test]
    fn transcription_request_uses_configured_transcription_model_and_decodes_audio() {
        let config = test_config().with_transcription_model("gpt-4o-mini-transcribe");
        let request = TranscriptionRequest {
            model: None,
            audio: MediaSource::InlineData {
                mime_type: "audio/wav".to_owned(),
                data_base64: "UklGRg==".to_owned(),
            },
            mime_type: None,
            filename: Some("sample.wav".to_owned()),
            language: Some("en".to_owned()),
            prompt: Some("Speaker is concise.".to_owned()),
            temperature: Some(0.2),
            metadata: BTreeMap::new(),
        };

        let payload = OpenAiTranscriptionRequest::from_core_request(&config, request);
        assert!(payload.is_ok(), "build transcription request");
        let Ok(payload) = payload else {
            return;
        };

        assert_eq!(payload.model, "gpt-4o-mini-transcribe");
        assert_eq!(payload.filename, "sample.wav");
        assert_eq!(payload.mime_type, "audio/wav");
        assert_eq!(payload.audio_bytes, b"RIFF".to_vec());
    }

    #[test]
    fn transcription_request_rejects_url_audio_sources() {
        let config = test_config();
        let request = TranscriptionRequest {
            model: None,
            audio: MediaSource::Url {
                url: "https://example.com/audio.wav".to_owned(),
            },
            mime_type: Some("audio/wav".to_owned()),
            filename: None,
            language: None,
            prompt: None,
            temperature: None,
            metadata: BTreeMap::new(),
        };

        let result = OpenAiTranscriptionRequest::from_core_request(&config, request);
        assert!(result.is_err(), "url-based audio should be rejected");
        let Err(error) = result else {
            return;
        };
        assert_eq!(error.kind, xlai_core::ErrorKind::Unsupported);
        assert!(error.message.contains("inline audio"));
    }

    #[test]
    fn transcription_response_preserves_provider_metadata() {
        let response: Result<OpenAiTranscriptionResponse, _> = serde_json::from_value(json!({
            "text": "hello world",
            "language": "en",
            "usage": {
                "total_tokens": 42
            }
        }));
        assert!(response.is_ok(), "deserialize transcription response");
        let Ok(response) = response else {
            return;
        };

        let response = response.into_core_response();
        assert_eq!(response.text, "hello world");
        assert_eq!(response.metadata.get("language"), Some(&json!("en")));
        assert_eq!(
            response.metadata.get("usage"),
            Some(&json!({
                "total_tokens": 42
            }))
        );
    }

    #[test]
    fn provider_error_message_surfaces_openai_quota_details() {
        let message = super::format_provider_error_message(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            Some("req_123"),
            r#"{
                "error": {
                    "message": "You exceeded your current quota.",
                    "type": "insufficient_quota",
                    "param": "model",
                    "code": "insufficient_quota"
                }
            }"#,
        );

        assert!(message.contains("429 Too Many Requests"));
        assert!(message.contains("request_id=req_123"));
        assert!(message.contains("You exceeded your current quota."));
        assert!(message.contains("type=insufficient_quota"));
        assert!(message.contains("code=insufficient_quota"));
        assert!(message.contains("param=model"));
    }

    #[test]
    fn provider_error_message_falls_back_to_raw_body_for_non_json_errors() {
        let message = super::format_provider_error_message(
            reqwest::StatusCode::BAD_GATEWAY,
            None,
            "upstream gateway timeout",
        );

        assert!(message.contains("502 Bad Gateway"));
        assert!(message.contains("upstream gateway timeout"));
    }
}
