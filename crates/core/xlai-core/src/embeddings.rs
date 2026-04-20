use serde::{Deserialize, Serialize};

use crate::chat::TokenUsage;
use crate::error::XlaiError;
use crate::metadata::Metadata;
use crate::runtime::{BoxFuture, RuntimeBound};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub inputs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
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

pub trait EmbeddingBackend {
    type Model: EmbeddingModel + 'static;

    fn into_embedding_model(self) -> Self::Model;
}

impl EmbeddingRequest {
    /// Encode this request as CBOR.
    pub fn to_cbor_vec(&self) -> Result<Vec<u8>, String> {
        crate::cbor::to_cbor_vec(self)
    }

    /// Decode from CBOR bytes.
    pub fn from_cbor_slice(bytes: &[u8]) -> Result<Self, String> {
        crate::cbor::from_cbor_slice(bytes)
    }
}

impl EmbeddingResponse {
    /// Encode this response as CBOR.
    pub fn to_cbor_vec(&self) -> Result<Vec<u8>, String> {
        crate::cbor::to_cbor_vec(self)
    }

    /// Decode from CBOR bytes.
    pub fn from_cbor_slice(bytes: &[u8]) -> Result<Self, String> {
        crate::cbor::from_cbor_slice(bytes)
    }
}

impl<T> EmbeddingBackend for T
where
    T: EmbeddingModel + 'static,
{
    type Model = T;

    fn into_embedding_model(self) -> Self::Model {
        self
    }
}
