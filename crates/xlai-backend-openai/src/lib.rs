use async_stream::try_stream;
use futures_util::StreamExt;
use reqwest::Client;
use reqwest::StatusCode;
use xlai_core::{
    BoxFuture, BoxStream, ChatBackend, ChatChunk, ChatModel, ChatRequest, ChatResponse, ErrorKind,
    MessageRole, StreamTextDelta, TranscriptionBackend, TranscriptionModel, TranscriptionRequest,
    TranscriptionResponse, XlaiError,
};

mod request;
mod response;
mod stream;
mod transcription;

#[cfg(test)]
mod tests;

use request::OpenAiChatRequest;
use response::{OpenAiChatResponse, OpenAiErrorEnvelope};
use stream::{OpenAiStreamResponse, SseParser, StreamState, update_finish_reason};
use transcription::{OpenAiTranscriptionRequest, OpenAiTranscriptionResponse};

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
            let payload = OpenAiChatRequest::from_core_request(&self.config, request, false)?;

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
            let payload = OpenAiChatRequest::from_core_request(&self.config, request, true)?;

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
                        update_finish_reason(&mut state, Some(reason.as_str()));
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
