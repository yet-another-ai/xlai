use std::collections::BTreeMap;

use async_stream::try_stream;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures_util::{StreamExt, stream as stream_util};
use reqwest::Client;
use reqwest::StatusCode;
use xlai_core::{
    BoxFuture, BoxStream, ChatBackend, ChatChunk, ChatModel, ChatRequest, ChatResponse, ErrorKind,
    MediaSource, MessageRole, StreamTextDelta, TranscriptionBackend, TranscriptionModel,
    TranscriptionRequest, TranscriptionResponse, TtsBackend, TtsChunk, TtsDeliveryMode, TtsModel,
    TtsRequest, TtsResponse, XlaiError,
};

mod request;
mod response;
mod stream;
mod transcription;
mod tts;

#[cfg(test)]
mod tests;

use request::OpenAiChatRequest;
use response::{OpenAiChatResponse, OpenAiErrorEnvelope};
use stream::{OpenAiStreamResponse, SseParser, StreamState, update_finish_reason};
use transcription::{OpenAiTranscriptionRequest, OpenAiTranscriptionResponse};
use tts::{
    build_speech_json_body, merge_header_metadata, mime_for_tts_format, openai_model_supports_speech_sse,
    parse_speech_sse_data, resolved_tts_model, tts_response_from_unary_bytes, ParsedSpeechSse,
};

#[derive(Clone, Debug)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub transcription_model: Option<String>,
    pub tts_model: Option<String>,
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
            tts_model: None,
        }
    }

    #[must_use]
    pub fn with_transcription_model(mut self, model: impl Into<String>) -> Self {
        self.transcription_model = Some(model.into());
        self
    }

    #[must_use]
    pub fn with_tts_model(mut self, model: impl Into<String>) -> Self {
        self.tts_model = Some(model.into());
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

    fn audio_speech_url(&self) -> String {
        format!("{}/audio/speech", self.base_url.trim_end_matches('/'))
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

#[derive(Clone, Debug)]
pub struct OpenAiTtsModel {
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

impl OpenAiTtsModel {
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

impl TtsBackend for OpenAiConfig {
    type Model = OpenAiTtsModel;

    fn into_tts_model(self) -> Self::Model {
        OpenAiTtsModel::new(self)
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

impl TtsModel for OpenAiTtsModel {
    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }

    fn synthesize(&self, request: TtsRequest) -> BoxFuture<'_, Result<TtsResponse, XlaiError>> {
        Box::pin(async move {
            let endpoint = self.config.audio_speech_url();
            let body = build_speech_json_body(&self.config, &request, None)?;

            let response = self
                .client
                .post(endpoint)
                .bearer_auth(&self.config.api_key)
                .header(reqwest::header::ACCEPT, "application/octet-stream")
                .json(&body)
                .send()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            let meta = merge_header_metadata(response.headers());
            let response = require_success_response(response).await?;

            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);

            let bytes = response
                .bytes()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?
                .to_vec();

            let response_format = request.response_format.unwrap_or(xlai_core::TtsAudioFormat::Mp3);
            let mut tts = tts_response_from_unary_bytes(bytes, response_format, meta);
            if let Some(ct) = content_type {
                tts.mime_type = ct.clone();
                if let MediaSource::InlineData { mime_type, .. } = &mut tts.audio {
                    *mime_type = ct;
                }
            }
            Ok(tts)
        })
    }

    fn synthesize_stream(
        &self,
        request: TtsRequest,
    ) -> BoxStream<'_, Result<TtsChunk, XlaiError>> {
        match request.delivery {
            TtsDeliveryMode::Unary => Box::pin(try_stream! {
                let response = self.synthesize(request).await?;
                yield TtsChunk::Started {
                    mime_type: response.mime_type.clone(),
                    metadata: BTreeMap::new(),
                };
                let data_base64 = match &response.audio {
                    MediaSource::InlineData { data_base64, .. } => data_base64.clone(),
                    MediaSource::Url { .. } => Err(XlaiError::new(
                        ErrorKind::Unsupported,
                        "openai-compatible TTS stream fallback requires inline audio bytes",
                    ))?,
                };
                yield TtsChunk::AudioDelta { data_base64 };
                yield TtsChunk::Finished { response };
            }),
            TtsDeliveryMode::Stream => {
                let model = resolved_tts_model(&self.config, &request);
                if !openai_model_supports_speech_sse(&model) {
                    let message = format!(
                        "openai-compatible speech SSE streaming is not supported for model {model}"
                    );
                    return Box::pin(stream_util::once(async move {
                        Err(XlaiError::new(ErrorKind::Unsupported, message))
                    }));
                }

                let this = self.clone();
                Box::pin(try_stream! {
                    let endpoint = this.config.audio_speech_url();
                    let body = build_speech_json_body(&this.config, &request, Some("sse"))?;

                    let response = this
                        .client
                        .post(endpoint)
                        .bearer_auth(&this.config.api_key)
                        .header(reqwest::header::ACCEPT, "text/event-stream")
                        .json(&body)
                        .send()
                        .await
                        .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

                    let meta = merge_header_metadata(response.headers());
                    let response = require_success_response(response).await?;

                    let response_format =
                        request.response_format.unwrap_or(xlai_core::TtsAudioFormat::Mp3);
                    let mime_type = mime_for_tts_format(response_format).to_owned();
                    yield TtsChunk::Started {
                        mime_type: mime_type.clone(),
                        metadata: meta.clone(),
                    };

                    let mut parser = SseParser::default();
                    let mut assembled: Vec<u8> = Vec::new();
                    let mut bytes_stream = response.bytes_stream();

                    while let Some(chunk) = bytes_stream.next().await {
                        let chunk = chunk
                            .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

                        for event in parser.push(&chunk) {
                            if event == "[DONE]" {
                                let response_done = xlai_core::TtsResponse {
                                    audio: MediaSource::InlineData {
                                        mime_type: mime_type.clone(),
                                        data_base64: STANDARD.encode(&assembled),
                                    },
                                    mime_type: mime_type.clone(),
                                    metadata: meta.clone(),
                                };
                                yield TtsChunk::Finished {
                                    response: response_done,
                                };
                                return;
                            }

                            match parse_speech_sse_data(&event)? {
                                ParsedSpeechSse::DeltaBase64(b64) => {
                                    let decoded = STANDARD.decode(b64.as_str()).map_err(|error| {
                                        XlaiError::new(
                                            ErrorKind::Provider,
                                            format!("invalid base64 in speech SSE delta: {error}"),
                                        )
                                    })?;
                                    assembled.extend_from_slice(&decoded);
                                    yield TtsChunk::AudioDelta { data_base64: b64 };
                                }
                                ParsedSpeechSse::Done => {
                                    let response_done = xlai_core::TtsResponse {
                                        audio: MediaSource::InlineData {
                                            mime_type: mime_type.clone(),
                                            data_base64: STANDARD.encode(&assembled),
                                        },
                                        mime_type: mime_type.clone(),
                                        metadata: meta.clone(),
                                    };
                                    yield TtsChunk::Finished {
                                        response: response_done,
                                    };
                                    return;
                                }
                                ParsedSpeechSse::Ignored => {}
                            }
                        }
                    }

                    let response_done = xlai_core::TtsResponse {
                        audio: MediaSource::InlineData {
                            mime_type: mime_type.clone(),
                            data_base64: STANDARD.encode(&assembled),
                        },
                        mime_type: mime_type.clone(),
                        metadata: meta.clone(),
                    };
                    yield TtsChunk::Finished {
                        response: response_done,
                    };
                })
            }
        }
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
