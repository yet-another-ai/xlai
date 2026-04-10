use serde::{Deserialize, Serialize};

use crate::content::MediaSource;
use crate::error::XlaiError;
use crate::metadata::Metadata;
use crate::runtime::{BoxFuture, RuntimeBound};

/// Provider-neutral quality hint for generated images.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ImageGenerationQuality {
    Low,
    Medium,
    High,
}

/// Desired background compositing mode for generated images.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ImageGenerationBackground {
    Transparent,
    Opaque,
}

/// Output container format for generated images.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ImageGenerationOutputFormat {
    Png,
    Jpeg,
    Webp,
}

/// One generated image plus provider-supplied annotations.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GeneratedImage {
    pub image: MediaSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
    #[serde(default)]
    pub metadata: Metadata,
}

/// Provider-neutral image generation request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageGenerationRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<ImageGenerationQuality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<ImageGenerationBackground>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<ImageGenerationOutputFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    #[serde(default)]
    pub metadata: Metadata,
}

/// Provider-neutral image generation response.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageGenerationResponse {
    pub images: Vec<GeneratedImage>,
    #[serde(default)]
    pub metadata: Metadata,
}

pub trait ImageGenerationModel: RuntimeBound {
    fn provider_name(&self) -> &'static str;

    fn generate_image(
        &self,
        request: ImageGenerationRequest,
    ) -> BoxFuture<'_, Result<ImageGenerationResponse, XlaiError>>;
}

pub trait ImageGenerationBackend {
    type Model: ImageGenerationModel + 'static;

    fn into_image_generation_model(self) -> Self::Model;
}

impl ImageGenerationRequest {
    /// Encode this request as CBOR.
    pub fn to_cbor_vec(&self) -> Result<Vec<u8>, String> {
        crate::cbor::to_cbor_vec(self)
    }

    /// Decode from CBOR bytes.
    pub fn from_cbor_slice(bytes: &[u8]) -> Result<Self, String> {
        crate::cbor::from_cbor_slice(bytes)
    }
}

impl ImageGenerationResponse {
    /// Encode this response as CBOR.
    pub fn to_cbor_vec(&self) -> Result<Vec<u8>, String> {
        crate::cbor::to_cbor_vec(self)
    }

    /// Decode from CBOR bytes.
    pub fn from_cbor_slice(bytes: &[u8]) -> Result<Self, String> {
        crate::cbor::from_cbor_slice(bytes)
    }
}

impl GeneratedImage {
    /// Encode this image payload as CBOR.
    pub fn to_cbor_vec(&self) -> Result<Vec<u8>, String> {
        crate::cbor::to_cbor_vec(self)
    }

    /// Decode from CBOR bytes.
    pub fn from_cbor_slice(bytes: &[u8]) -> Result<Self, String> {
        crate::cbor::from_cbor_slice(bytes)
    }
}

impl<T> ImageGenerationBackend for T
where
    T: ImageGenerationModel + 'static,
{
    type Model = T;

    fn into_image_generation_model(self) -> Self::Model {
        self
    }
}
