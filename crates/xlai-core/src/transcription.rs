use serde::{Deserialize, Serialize};

use crate::content::MediaSource;
use crate::error::XlaiError;
use crate::metadata::Metadata;
use crate::runtime::{BoxFuture, RuntimeBound};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionRequest {
    pub model: Option<String>,
    pub audio: MediaSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionResponse {
    pub text: String,
    #[serde(default)]
    pub metadata: Metadata,
}

pub trait TranscriptionModel: RuntimeBound {
    fn provider_name(&self) -> &'static str;

    fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> BoxFuture<'_, Result<TranscriptionResponse, XlaiError>>;
}

pub trait TranscriptionBackend {
    type Model: TranscriptionModel + 'static;

    fn into_transcription_model(self) -> Self::Model;
}

impl TranscriptionRequest {
    /// Encode this request as CBOR.
    pub fn to_cbor_vec(&self) -> Result<Vec<u8>, String> {
        crate::cbor::to_cbor_vec(self)
    }

    /// Decode from CBOR bytes.
    pub fn from_cbor_slice(bytes: &[u8]) -> Result<Self, String> {
        crate::cbor::from_cbor_slice(bytes)
    }
}

impl TranscriptionResponse {
    /// Encode this response as CBOR.
    pub fn to_cbor_vec(&self) -> Result<Vec<u8>, String> {
        crate::cbor::to_cbor_vec(self)
    }

    /// Decode from CBOR bytes.
    pub fn from_cbor_slice(bytes: &[u8]) -> Result<Self, String> {
        crate::cbor::from_cbor_slice(bytes)
    }
}

impl<T> TranscriptionBackend for T
where
    T: TranscriptionModel + 'static,
{
    type Model = T;

    fn into_transcription_model(self) -> Self::Model {
        self
    }
}
