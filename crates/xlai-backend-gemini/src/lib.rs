use async_stream::try_stream;
use futures_util::stream::StreamExt;
use reqwest::Client;
use xlai_core::{
    BoxFuture, BoxStream, ChatBackend, ChatChunk, ChatModel, ChatRequest, ChatResponse, ErrorKind,
    XlaiError,
};

mod request;
mod response;
mod stream;

#[cfg(test)]
mod tests;

use request::GeminiChatRequest;
use response::GeminiChatResponse;
use stream::{SseParser, StreamState};

#[derive(Clone, Debug)]
pub struct GeminiConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl GeminiConfig {
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

    fn generate_content_url(&self) -> String {
        format!(
            "{}/models/{}:generateContent",
            self.base_url.trim_end_matches('/'),
            self.model
        )
    }

    fn stream_generate_content_url(&self) -> String {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url.trim_end_matches('/'),
            self.model
        )
    }
}

#[derive(Clone, Debug)]
pub struct GeminiChatModel {
    client: Client,
    config: GeminiConfig,
}

impl GeminiChatModel {
    #[must_use]
    pub fn new(config: GeminiConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }
}

impl ChatBackend for GeminiConfig {
    type Model = GeminiChatModel;

    fn into_chat_model(self) -> Self::Model {
        GeminiChatModel::new(self)
    }
}

impl ChatModel for GeminiChatModel {
    fn provider_name(&self) -> &'static str {
        "gemini"
    }

    fn generate(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, XlaiError>> {
        Box::pin(async move {
            let endpoint = self.config.generate_content_url();
            let payload = GeminiChatRequest::from_core_request(request)?;

            let response = self
                .client
                .post(&endpoint)
                .header("x-goog-api-key", &self.config.api_key)
                .json(&payload)
                .send()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            let status = response.status();
            if !status.is_success() {
                let text = response.text().await.unwrap_or_default();
                return Err(XlaiError::new(
                    ErrorKind::Provider,
                    format!("Gemini API error ({}): {}", status, text),
                ));
            }

            let response: GeminiChatResponse = response
                .json()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            response.into_core_response()
        })
    }

    fn generate_stream(&self, request: ChatRequest) -> BoxStream<'_, Result<ChatChunk, XlaiError>> {
        let this = self.clone();
        Box::pin(try_stream! {
            let endpoint = this.config.stream_generate_content_url();
            let payload = GeminiChatRequest::from_core_request(request)?;

            let response = this
                .client
                .post(&endpoint)
                .header("x-goog-api-key", &this.config.api_key)
                .json(&payload)
                .send()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            let status = response.status();
            if !status.is_success() {
                let text = response.text().await.unwrap_or_default();
                Err(XlaiError::new(
                    ErrorKind::Provider,
                    format!("Gemini API error ({}): {}", status, text),
                ))?;
                unreachable!();
            }

            let mut bytes_stream = response.bytes_stream();
            let mut parser = SseParser::default();
            let mut state = StreamState::default();

            while let Some(chunk) = bytes_stream.next().await {
                let chunk = chunk
                    .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

                for event in parser.push(&chunk) {
                    let chunks = state.process_event(&event)?;
                    for c in chunks {
                        yield c;
                    }
                }
            }

            let response = state.into_chat_response()?;
            yield ChatChunk::Finished(response);
        })
    }
}
