use async_stream::try_stream;
use futures_util::stream::StreamExt;
use reqwest::Client;
use xlai_core::{
    BoxFuture, BoxStream, ChatBackend, ChatChunk, ChatModel, ChatRequest, ChatResponse,
    EmbeddingBackend, EmbeddingModel, EmbeddingRequest, EmbeddingResponse, ErrorKind,
    ImageGenerationBackend, ImageGenerationModel, ImageGenerationRequest, ImageGenerationResponse,
    XlaiError,
};

mod embeddings;
mod image_generation;
mod request;
mod response;
mod stream;

#[cfg(test)]
mod tests;

use embeddings::{
    GeminiBatchEmbeddingRequest, GeminiBatchEmbeddingResponse, GeminiEmbeddingRequestPayload,
    GeminiEmbeddingResponse,
};
use image_generation::{GeminiImageGenerationRequest, GeminiImageGenerationResponse};
use request::GeminiChatRequest;
use response::GeminiChatResponse;
use stream::{SseParser, StreamState};

#[derive(Clone, Debug)]
pub struct GeminiConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub embedding_model: Option<String>,
    pub image_model: Option<String>,
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
            embedding_model: None,
            image_model: None,
        }
    }

    #[must_use]
    pub fn with_embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = Some(model.into());
        self
    }

    #[must_use]
    pub fn with_image_model(mut self, model: impl Into<String>) -> Self {
        self.image_model = Some(model.into());
        self
    }

    fn generate_content_url(&self) -> String {
        format!(
            "{}/models/{}:generateContent",
            self.base_url.trim_end_matches('/'),
            self.model
        )
    }

    fn embed_content_url(&self, model: &str) -> String {
        format!(
            "{}/models/{}:embedContent",
            self.base_url.trim_end_matches('/'),
            model
        )
    }

    fn batch_embed_contents_url(&self, model: &str) -> String {
        format!(
            "{}/models/{}:batchEmbedContents",
            self.base_url.trim_end_matches('/'),
            model
        )
    }

    fn generate_image_url(&self, model: &str) -> String {
        format!(
            "{}/models/{}:generateContent",
            self.base_url.trim_end_matches('/'),
            model
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

#[derive(Clone, Debug)]
pub struct GeminiEmbeddingModel {
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

impl GeminiEmbeddingModel {
    #[must_use]
    pub fn new(config: GeminiConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GeminiImageGenerationModel {
    client: Client,
    config: GeminiConfig,
}

impl GeminiImageGenerationModel {
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

impl EmbeddingBackend for GeminiConfig {
    type Model = GeminiEmbeddingModel;

    fn into_embedding_model(self) -> Self::Model {
        GeminiEmbeddingModel::new(self)
    }
}

impl ImageGenerationBackend for GeminiConfig {
    type Model = GeminiImageGenerationModel;

    fn into_image_generation_model(self) -> Self::Model {
        GeminiImageGenerationModel::new(self)
    }
}

impl EmbeddingModel for GeminiEmbeddingModel {
    fn provider_name(&self) -> &'static str {
        "gemini"
    }

    fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> BoxFuture<'_, Result<EmbeddingResponse, XlaiError>> {
        Box::pin(async move {
            if request.inputs.is_empty() {
                return Err(XlaiError::new(
                    ErrorKind::Validation,
                    "embedding request requires at least one input",
                ));
            }

            let model_name = request
                .model
                .clone()
                .or_else(|| self.config.embedding_model.clone())
                .unwrap_or_else(|| self.config.model.clone());

            if request.inputs.len() == 1 {
                let endpoint = self.config.embed_content_url(&model_name);
                let payload = GeminiEmbeddingRequestPayload::from_core_request(
                    model_name,
                    request.inputs.into_iter().next().unwrap_or_default(),
                    request.dimensions,
                );
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
                let response: GeminiEmbeddingResponse = response
                    .json()
                    .await
                    .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;
                return response.into_core_response();
            }

            let endpoint = self.config.batch_embed_contents_url(&model_name);
            let payload = GeminiBatchEmbeddingRequest::from_core_request(
                model_name,
                request.inputs,
                request.dimensions,
            );
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
            let response: GeminiBatchEmbeddingResponse = response
                .json()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            response.into_core_response()
        })
    }
}

impl ImageGenerationModel for GeminiImageGenerationModel {
    fn provider_name(&self) -> &'static str {
        "gemini"
    }

    fn generate_image(
        &self,
        request: ImageGenerationRequest,
    ) -> BoxFuture<'_, Result<ImageGenerationResponse, XlaiError>> {
        Box::pin(async move {
            let requested_format = request.output_format;
            let model_name = request
                .model
                .clone()
                .or_else(|| self.config.image_model.clone())
                .unwrap_or_else(|| self.config.model.clone());
            let endpoint = self.config.generate_image_url(&model_name);
            let payload = GeminiImageGenerationRequest::from_core_request(request)?;

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

            let response: GeminiImageGenerationResponse = response
                .json()
                .await
                .map_err(|error| XlaiError::new(ErrorKind::Provider, error.to_string()))?;

            response.into_core_response(requested_format)
        })
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
