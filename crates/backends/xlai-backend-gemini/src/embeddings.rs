use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::json;
use xlai_core::{EmbeddingResponse, TokenUsage, XlaiError};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiEmbeddingRequestPayload {
    model: String,
    content: GeminiContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<u32>,
}

impl GeminiEmbeddingRequestPayload {
    pub(crate) fn from_core_request(model: String, input: String, dimensions: Option<u32>) -> Self {
        Self {
            model,
            content: GeminiContent::from_text(input),
            output_dimensionality: dimensions,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiBatchEmbeddingRequest {
    requests: Vec<GeminiEmbeddingRequestPayload>,
}

impl GeminiBatchEmbeddingRequest {
    pub(crate) fn from_core_request(
        model: String,
        inputs: Vec<String>,
        dimensions: Option<u32>,
    ) -> Self {
        Self {
            requests: inputs
                .into_iter()
                .map(|input| {
                    GeminiEmbeddingRequestPayload::from_core_request(
                        model.clone(),
                        input,
                        dimensions,
                    )
                })
                .collect(),
        }
    }
}

#[derive(Serialize)]
struct GeminiContent {
    parts: Vec<GeminiTextPart>,
}

impl GeminiContent {
    fn from_text(text: String) -> Self {
        Self {
            parts: vec![GeminiTextPart { text }],
        }
    }
}

#[derive(Serialize)]
struct GeminiTextPart {
    text: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiEmbeddingResponse {
    embedding: GeminiEmbeddingValues,
    #[serde(default)]
    usage_metadata: Option<GeminiEmbeddingUsage>,
}

impl GeminiEmbeddingResponse {
    pub(crate) fn into_core_response(self) -> Result<EmbeddingResponse, XlaiError> {
        let mut metadata = BTreeMap::new();
        if let Some(usage_metadata) = &self.usage_metadata {
            metadata.insert("usage_metadata".to_owned(), json!(usage_metadata));
        }

        Ok(EmbeddingResponse {
            vectors: vec![self.embedding.values],
            usage: self.usage_metadata.map(Into::into),
            metadata,
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiBatchEmbeddingResponse {
    embeddings: Vec<GeminiEmbeddingValues>,
    #[serde(default)]
    usage_metadata: Option<GeminiEmbeddingUsage>,
}

impl GeminiBatchEmbeddingResponse {
    pub(crate) fn into_core_response(self) -> Result<EmbeddingResponse, XlaiError> {
        let mut metadata = BTreeMap::new();
        if let Some(usage_metadata) = &self.usage_metadata {
            metadata.insert("usage_metadata".to_owned(), json!(usage_metadata));
        }

        Ok(EmbeddingResponse {
            vectors: self
                .embeddings
                .into_iter()
                .map(|item| item.values)
                .collect(),
            usage: self.usage_metadata.map(Into::into),
            metadata,
        })
    }
}

#[derive(Deserialize)]
pub(crate) struct GeminiEmbeddingValues {
    values: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiEmbeddingUsage {
    #[serde(default)]
    prompt_token_count: Option<u32>,
    #[serde(default)]
    total_token_count: Option<u32>,
}

impl From<GeminiEmbeddingUsage> for TokenUsage {
    fn from(value: GeminiEmbeddingUsage) -> Self {
        let input_tokens = value.prompt_token_count.unwrap_or(0);
        Self {
            input_tokens,
            output_tokens: 0,
            total_tokens: value.total_token_count.unwrap_or(input_tokens),
        }
    }
}
