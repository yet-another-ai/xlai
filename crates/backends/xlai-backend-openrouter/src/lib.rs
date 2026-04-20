use async_stream::try_stream;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::Value;
use xlai_core::{
    BoxFuture, BoxStream, ChatBackend, ChatChunk, ChatModel, ChatRequest, ChatResponse,
    ChatRetryPolicy, ErrorKind, MessageRole, StreamTextDelta, XlaiError,
};

mod chat_retry;
mod provider_response;
mod request;
mod response;
mod stream;

#[cfg(test)]
mod tests;

use chat_retry::{
    backoff_delay_ms, retry_limits_for_chat_request, should_retry_xlai_error, sleep_ms,
};
use provider_response::{require_success_response, xlai_error_from_reqwest};
use request::OpenRouterChatRequest;
use response::OpenRouterChatResponse;
use stream::{SseParser, StreamState, maybe_completed_response, update_finish_reason};

#[derive(Clone, Debug)]
pub struct OpenRouterConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub http_referer: Option<String>,
    pub app_title: Option<String>,
    pub app_categories: Option<String>,
}

impl OpenRouterConfig {
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
            http_referer: None,
            app_title: None,
            app_categories: None,
        }
    }

    #[must_use]
    pub fn with_http_referer(mut self, http_referer: impl Into<String>) -> Self {
        self.http_referer = Some(http_referer.into());
        self
    }

    #[must_use]
    pub fn with_app_title(mut self, app_title: impl Into<String>) -> Self {
        self.app_title = Some(app_title.into());
        self
    }

    #[must_use]
    pub fn with_app_categories(mut self, app_categories: impl Into<String>) -> Self {
        self.app_categories = Some(app_categories.into());
        self
    }

    fn responses_url(&self) -> String {
        format!("{}/responses", self.base_url.trim_end_matches('/'))
    }
}

#[derive(Clone, Debug)]
pub struct OpenRouterChatModel {
    client: Client,
    config: OpenRouterConfig,
}

impl OpenRouterChatModel {
    #[must_use]
    pub fn new(config: OpenRouterConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    fn request_builder(&self, endpoint: &str) -> reqwest::RequestBuilder {
        let mut request = self.client.post(endpoint).bearer_auth(&self.config.api_key);

        if let Some(http_referer) = &self.config.http_referer {
            request = request.header("HTTP-Referer", http_referer);
        }
        if let Some(app_title) = &self.config.app_title {
            request = request.header("X-OpenRouter-Title", app_title);
        }
        if let Some(app_categories) = &self.config.app_categories {
            request = request.header("X-OpenRouter-Categories", app_categories);
        }

        request
    }

    async fn post_responses_checked(
        &self,
        payload: &OpenRouterChatRequest,
        max_extra_attempts: u32,
        policy_for_backoff: Option<&ChatRetryPolicy>,
    ) -> Result<reqwest::Response, XlaiError> {
        let endpoint = self.config.responses_url();
        let mut failures = 0u32;
        loop {
            let attempt = async {
                let response = self
                    .request_builder(&endpoint)
                    .json(payload)
                    .send()
                    .await
                    .map_err(xlai_error_from_reqwest)?;
                require_success_response(response).await
            }
            .await;
            match attempt {
                Ok(response) => return Ok(response),
                Err(error) => {
                    if failures >= max_extra_attempts || !should_retry_xlai_error(&error) {
                        return Err(error);
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

impl ChatBackend for OpenRouterConfig {
    type Model = OpenRouterChatModel;

    fn into_chat_model(self) -> Self::Model {
        OpenRouterChatModel::new(self)
    }
}

impl ChatModel for OpenRouterChatModel {
    fn provider_name(&self) -> &'static str {
        "openrouter"
    }

    fn generate(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, XlaiError>> {
        Box::pin(async move {
            let policy = request.retry_policy.clone();
            let (max_extra, policy_for_backoff) = retry_limits_for_chat_request(policy.as_ref());
            let payload = OpenRouterChatRequest::from_core_request(&self.config, request, false)?;

            let response = self
                .post_responses_checked(&payload, max_extra, policy_for_backoff)
                .await?;

            let response: OpenRouterChatResponse = response
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
            let payload = OpenRouterChatRequest::from_core_request(&this.config, request, true)?;

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
                                .unwrap_or("openrouter stream returned an error event");
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
