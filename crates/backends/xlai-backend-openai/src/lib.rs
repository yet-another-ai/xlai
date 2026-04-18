use std::collections::BTreeMap;

use async_stream::try_stream;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures_util::{StreamExt, stream as stream_util};
use reqwest::Client;
use serde_json::Value;
use xlai_core::{
    BoxFuture, BoxStream, ChatBackend, ChatChunk, ChatModel, ChatRequest, ChatResponse,
    ChatRetryPolicy, ErrorKind, ImageGenerationBackend, ImageGenerationModel,
    ImageGenerationRequest, ImageGenerationResponse, MediaSource, MessageRole, StreamTextDelta,
    TranscriptionBackend, TranscriptionModel, TranscriptionRequest, TranscriptionResponse,
    TtsBackend, TtsChunk, TtsDeliveryMode, TtsModel, TtsRequest, TtsResponse, XlaiError,
};

mod chat_retry;
mod image_generation;
mod provider_response;
mod request;
mod response;
mod stream;
mod transcription;
mod tts;

#[cfg(test)]
mod tests;

use chat_retry::{
    backoff_delay_ms, retry_limits_for_chat_request, should_retry_xlai_error, sleep_ms,
};
use image_generation::{OpenAiImageGenerationRequest, OpenAiImageGenerationResponse};
use provider_response::{require_success_response, xlai_error_from_reqwest};
use request::OpenAiChatRequest;
use response::OpenAiChatResponse;
use stream::{SseParser, StreamState, maybe_completed_response, update_finish_reason};
use transcription::{OpenAiTranscriptionRequest, OpenAiTranscriptionResponse};
use tts::{
    ParsedSpeechSse, build_speech_json_body, merge_header_metadata, mime_for_tts_format,
    openai_model_supports_speech_sse, parse_speech_sse_data, resolved_tts_model,
    tts_response_from_unary_bytes,
};

#[derive(Clone, Debug)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub image_model: Option<String>,
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
            image_model: None,
            transcription_model: None,
            tts_model: None,
        }
    }

    #[must_use]
    pub fn with_image_model(mut self, model: impl Into<String>) -> Self {
        self.image_model = Some(model.into());
        self
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

    fn responses_url(&self) -> String {
        format!("{}/responses", self.base_url.trim_end_matches('/'))
    }

    fn images_generations_url(&self) -> String {
        format!("{}/images/generations", self.base_url.trim_end_matches('/'))
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
pub struct OpenAiImageGenerationModel {
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

    /// POST `/responses` with optional retries before the response body is consumed.
    async fn post_responses_checked(
        &self,
        payload: &OpenAiChatRequest,
        max_extra_attempts: u32,
        policy_for_backoff: Option<&ChatRetryPolicy>,
    ) -> Result<reqwest::Response, XlaiError> {
        let endpoint = self.config.responses_url();
        let mut failures = 0u32;
        loop {
            let attempt = async {
                let response = self
                    .client
                    .post(&endpoint)
                    .bearer_auth(&self.config.api_key)
                    .json(payload)
                    .send()
                    .await
                    .map_err(xlai_error_from_reqwest)?;
                require_success_response(response).await
            }
            .await;
            match attempt {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if failures >= max_extra_attempts || !should_retry_xlai_error(&e) {
                        return Err(e);
                    }
                    if let Some(policy) = policy_for_backoff {
                        sleep_ms(backoff_delay_ms(policy, failures)).await;
                    }
                    failures += 1;
                }
            }
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

impl OpenAiImageGenerationModel {
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

impl ImageGenerationBackend for OpenAiConfig {
    type Model = OpenAiImageGenerationModel;

    fn into_image_generation_model(self) -> Self::Model {
        OpenAiImageGenerationModel::new(self)
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
            let policy = request.retry_policy.clone();
            let (max_extra, policy_for_backoff) = retry_limits_for_chat_request(policy.as_ref());
            let payload = OpenAiChatRequest::from_core_request(&self.config, request, false)?;

            let response = self
                .post_responses_checked(&payload, max_extra, policy_for_backoff)
                .await?;

            let response: OpenAiChatResponse = response
                .json()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            response.into_core_response()
        })
    }

    fn generate_stream(&self, request: ChatRequest) -> BoxStream<'_, Result<ChatChunk, XlaiError>> {
        let this = self.clone();
        Box::pin(try_stream! {
            let policy = request.retry_policy.clone();
            let (max_extra, policy_for_backoff) = retry_limits_for_chat_request(policy.as_ref());
            let payload = OpenAiChatRequest::from_core_request(&this.config, request, true)?;

            let response = this
                .post_responses_checked(&payload, max_extra, policy_for_backoff)
                .await?;

            let mut bytes_stream = response.bytes_stream();
            let mut parser = SseParser::default();
            let mut state = StreamState::default();
            while let Some(chunk) = bytes_stream.next().await {
                let chunk = chunk
                    .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

                for event in parser.push(&chunk) {
                    if event == "[DONE]" {
                        let response = state.into_chat_response()?;
                        yield ChatChunk::Finished(response);
                        return;
                    }

                    let event_value: Value = serde_json::from_str(&event).map_err(|error| {
                        XlaiError::new(
                            ErrorKind::Provider,
                            format!("failed to parse stream event: {error}"),
                        )
                    })?;

                    if let Some(response) = maybe_completed_response(&event_value)? {
                        yield ChatChunk::Finished(response);
                        return;
                    }

                    match event_value.get("type").and_then(Value::as_str) {
                        Some("response.output_text.delta") => {
                            let message_index = event_value
                                .get("output_index")
                                .and_then(Value::as_u64)
                                .unwrap_or(0) as usize;
                            if state.mark_message_started(message_index) {
                                yield ChatChunk::MessageStart {
                                    role: MessageRole::Assistant,
                                    message_index,
                                };
                            }
                            if let Some(delta) = event_value.get("delta").and_then(Value::as_str) {
                                state.message_content.push_str(delta);
                                yield ChatChunk::ContentDelta(StreamTextDelta {
                                    message_index,
                                    part_index: event_value
                                        .get("content_index")
                                        .and_then(Value::as_u64)
                                        .unwrap_or(0) as usize,
                                    delta: delta.to_owned(),
                                });
                            }
                        }
                        Some("response.output_item.added") => {
                            if let Some(item) = event_value.get("item").and_then(Value::as_object)
                            {
                                let index = event_value
                                    .get("output_index")
                                    .and_then(Value::as_u64)
                                    .unwrap_or(0) as usize;
                                match item.get("type").and_then(Value::as_str) {
                                    Some("message") if state.mark_message_started(index) => {
                                        yield ChatChunk::MessageStart {
                                            role: MessageRole::Assistant,
                                            message_index: index,
                                        };
                                    }
                                    Some("function_call") => {
                                        let chunk = state.apply_tool_call_added(
                                            index,
                                            item.get("call_id").and_then(Value::as_str).map(str::to_owned),
                                            item.get("name").and_then(Value::as_str).map(str::to_owned),
                                            item.get("arguments").and_then(Value::as_str).map(str::to_owned),
                                        );
                                        yield ChatChunk::ToolCallDelta(chunk);
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Some("response.function_call_arguments.delta") => {
                            let index = event_value
                                .get("output_index")
                                .and_then(Value::as_u64)
                                .unwrap_or(0) as usize;
                            let delta = event_value
                                .get("delta")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_owned();
                            let chunk = state.apply_tool_delta(index, delta);
                            yield ChatChunk::ToolCallDelta(chunk);
                        }
                        Some("response.output_item.done") => {
                            if let Some(item) = event_value.get("item") {
                                state.push_output_item(item.clone());
                            }
                        }
                        Some("response.completed") => {
                            let has_tool_calls = state.has_tool_calls();
                            update_finish_reason(
                                &mut state,
                                event_value
                                    .get("response")
                                    .and_then(|r| r.get("status"))
                                    .and_then(Value::as_str),
                                event_value
                                    .get("response")
                                    .and_then(|r| r.get("incomplete_details"))
                                    .and_then(|d| d.get("reason"))
                                    .and_then(Value::as_str),
                                has_tool_calls,
                            );
                        }
                        Some("response.incomplete") => {
                            let has_tool_calls = state.has_tool_calls();
                            update_finish_reason(
                                &mut state,
                                Some("incomplete"),
                                event_value
                                    .get("response")
                                    .and_then(|r| r.get("incomplete_details"))
                                    .and_then(|d| d.get("reason"))
                                    .and_then(Value::as_str),
                                has_tool_calls,
                            );
                        }
                        Some("error") => {
                            let message = event_value
                                .get("error")
                                .and_then(|e| e.get("message"))
                                .and_then(Value::as_str)
                                .unwrap_or("openai-compatible stream returned an error event");
                            Err(XlaiError::new(ErrorKind::Provider, message))?;
                        }
                        _ => {}
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

impl ImageGenerationModel for OpenAiImageGenerationModel {
    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }

    fn generate_image(
        &self,
        request: ImageGenerationRequest,
    ) -> BoxFuture<'_, Result<ImageGenerationResponse, XlaiError>> {
        Box::pin(async move {
            let requested_format = request.output_format;
            let endpoint = self.config.images_generations_url();
            let payload = OpenAiImageGenerationRequest::from_core_request(&self.config, request)?;

            let response = self
                .client
                .post(endpoint)
                .bearer_auth(&self.config.api_key)
                .json(&payload)
                .send()
                .await
                .map_err(xlai_error_from_reqwest)?;

            let response = require_success_response(response).await?;

            let response: OpenAiImageGenerationResponse = response
                .json()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            response.into_core_response(requested_format)
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

            let response_format = request
                .response_format
                .unwrap_or(xlai_core::TtsAudioFormat::Mp3);
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

    fn synthesize_stream(&self, request: TtsRequest) -> BoxStream<'_, Result<TtsChunk, XlaiError>> {
        match request.delivery {
            TtsDeliveryMode::Unary => Box::pin(try_stream! {
                let response = self.synthesize(request).await?;
                yield TtsChunk::Started {
                    mime_type: response.mime_type.clone(),
                    metadata: BTreeMap::new(),
                };
                let data = match &response.audio {
                    MediaSource::InlineData { data, .. } => data.clone(),
                    MediaSource::Url { .. } => Err(XlaiError::new(
                        ErrorKind::Unsupported,
                        "openai-compatible TTS stream fallback requires inline audio bytes",
                    ))?,
                };
                yield TtsChunk::AudioDelta { data };
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
                                        data: assembled.clone(),
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
                                    yield TtsChunk::AudioDelta { data: decoded };
                                }
                                ParsedSpeechSse::Done => {
                                    let response_done = xlai_core::TtsResponse {
                                        audio: MediaSource::InlineData {
                                            mime_type: mime_type.clone(),
                                            data: assembled.clone(),
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
                            data: assembled,
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
