use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::json;
use xlai_core::{EmbeddingRequest, EmbeddingResponse, ErrorKind, TokenUsage, XlaiError};

use crate::OpenAiConfig;

#[derive(Serialize)]
pub(crate) struct OpenAiEmbeddingRequest {
    model: String,
    input: Vec<String>,
    encoding_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

impl OpenAiEmbeddingRequest {
    pub(crate) fn from_core_request(
        config: &OpenAiConfig,
        request: EmbeddingRequest,
    ) -> Result<Self, XlaiError> {
        if request.inputs.is_empty() {
            return Err(XlaiError::new(
                ErrorKind::Validation,
                "embedding request requires at least one input",
            ));
        }

        Ok(Self {
            model: request
                .model
                .or_else(|| config.embedding_model.clone())
                .unwrap_or_else(|| config.model.clone()),
            input: request.inputs,
            encoding_format: "float",
            dimensions: request.dimensions,
        })
    }
}

#[derive(Deserialize)]
pub(crate) struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingDatum>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<OpenAiEmbeddingUsage>,
}

impl OpenAiEmbeddingResponse {
    pub(crate) fn into_core_response(self) -> Result<EmbeddingResponse, XlaiError> {
        let vectors = self
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect::<Vec<_>>();

        let mut metadata = BTreeMap::new();
        if let Some(model) = self.model {
            metadata.insert("model".to_owned(), json!(model));
        }

        Ok(EmbeddingResponse {
            vectors,
            usage: self.usage.map(Into::into),
            metadata,
        })
    }
}

#[derive(Deserialize)]
struct OpenAiEmbeddingDatum {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingUsage {
    #[serde(rename = "prompt_tokens")]
    prompt_tokens: u32,
    #[serde(rename = "total_tokens")]
    total_tokens: u32,
}

impl From<OpenAiEmbeddingUsage> for TokenUsage {
    fn from(value: OpenAiEmbeddingUsage) -> Self {
        Self {
            input_tokens: value.prompt_tokens,
            output_tokens: 0,
            total_tokens: value.total_tokens,
        }
    }
}
