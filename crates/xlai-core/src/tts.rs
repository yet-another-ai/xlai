use std::collections::BTreeMap;

use async_stream::try_stream;
use serde::{Deserialize, Serialize};

use crate::content::MediaSource;
use crate::error::{ErrorKind, XlaiError};
use crate::metadata::Metadata;
use crate::runtime::{BoxFuture, BoxStream, RuntimeBound};

/// Output container format for text-to-speech (OpenAI-compatible `response_format` values).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TtsAudioFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

/// Whether the caller wants a single response body or a stream of audio chunks.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TtsDeliveryMode {
    /// One complete audio payload (e.g. HTTP body).
    #[default]
    Unary,
    /// Incremental audio (e.g. SSE deltas); semantics depend on the provider.
    Stream,
}

/// One reference sample for voice cloning / conditioning (provider-specific).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VoiceReferenceSample {
    pub audio: MediaSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    #[serde(default)]
    pub metadata: Metadata,
}

/// Provider-neutral voice selection.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum VoiceSpec {
    /// Built-in voice name (e.g. OpenAI `alloy`).
    Preset { name: String },
    /// Provider-stored or custom voice identifier.
    ProviderRef {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider: Option<String>,
    },
    /// Reference-audio-based cloning / conditioning (not all backends support this).
    Clone {
        references: Vec<VoiceReferenceSample>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TtsRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub input: String,
    pub voice: VoiceSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<TtsAudioFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default)]
    pub delivery: TtsDeliveryMode,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TtsResponse {
    pub audio: MediaSource,
    pub mime_type: String,
    #[serde(default)]
    pub metadata: Metadata,
}

/// One item from a streaming TTS response.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum TtsChunk {
    Started {
        mime_type: String,
        #[serde(default)]
        metadata: Metadata,
    },
    AudioDelta {
        /// PCM/container chunk bytes. JSON: base64 string field `data`. CBOR: byte string.
        #[serde(with = "crate::serde_bytes_format")]
        data: Vec<u8>,
    },
    Finished {
        response: TtsResponse,
    },
}

pub trait TtsModel: RuntimeBound {
    fn provider_name(&self) -> &'static str;

    fn synthesize(&self, request: TtsRequest) -> BoxFuture<'_, Result<TtsResponse, XlaiError>>;

    /// Streaming synthesis. Default implementation calls [`Self::synthesize`] once and emits
    /// [`TtsChunk::Started`], one [`TtsChunk::AudioDelta`], then [`TtsChunk::Finished`].
    fn synthesize_stream(&self, request: TtsRequest) -> BoxStream<'_, Result<TtsChunk, XlaiError>> {
        Box::pin(try_stream! {
            let response = self.synthesize(request).await?;
            yield TtsChunk::Started {
                mime_type: response.mime_type.clone(),
                metadata: BTreeMap::new(),
            };
            let data = match &response.audio {
                MediaSource::InlineData { data, .. } => data.clone(),
                MediaSource::Url { .. } => Err(XlaiError::new(
                    ErrorKind::Unsupported,
                    "default TTS stream fallback requires inline audio bytes",
                ))?,
            };
            yield TtsChunk::AudioDelta { data };
            yield TtsChunk::Finished { response };
        })
    }
}

pub trait TtsBackend {
    type Model: TtsModel + 'static;

    fn into_tts_model(self) -> Self::Model;
}

impl TtsRequest {
    /// Encode this request as CBOR (efficient inline audio bytes).
    pub fn to_cbor_vec(&self) -> Result<Vec<u8>, String> {
        crate::cbor::to_cbor_vec(self)
    }

    /// Decode from CBOR bytes.
    pub fn from_cbor_slice(bytes: &[u8]) -> Result<Self, String> {
        crate::cbor::from_cbor_slice(bytes)
    }
}

impl TtsResponse {
    /// Encode this response as CBOR.
    pub fn to_cbor_vec(&self) -> Result<Vec<u8>, String> {
        crate::cbor::to_cbor_vec(self)
    }

    /// Decode from CBOR bytes.
    pub fn from_cbor_slice(bytes: &[u8]) -> Result<Self, String> {
        crate::cbor::from_cbor_slice(bytes)
    }
}

impl TtsChunk {
    /// Encode this chunk as CBOR.
    pub fn to_cbor_vec(&self) -> Result<Vec<u8>, String> {
        crate::cbor::to_cbor_vec(self)
    }

    /// Decode from CBOR bytes.
    pub fn from_cbor_slice(bytes: &[u8]) -> Result<Self, String> {
        crate::cbor::from_cbor_slice(bytes)
    }
}

impl<T> TtsBackend for T
where
    T: TtsModel + 'static,
{
    type Model = T;

    fn into_tts_model(self) -> Self::Model {
        self
    }
}
