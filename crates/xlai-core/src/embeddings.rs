use serde::{Deserialize, Serialize};

use crate::chat::TokenUsage;
use crate::error::XlaiError;
use crate::metadata::Metadata;
use crate::runtime::{BoxFuture, RuntimeBound};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub inputs: Vec<String>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingResponse {
    pub vectors: Vec<Vec<f32>>,
    pub usage: Option<TokenUsage>,
    #[serde(default)]
    pub metadata: Metadata,
}

pub trait EmbeddingModel: RuntimeBound {
    fn provider_name(&self) -> &'static str;

    fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> BoxFuture<'_, Result<EmbeddingResponse, XlaiError>>;
}
