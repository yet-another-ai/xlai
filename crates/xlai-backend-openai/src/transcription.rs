use std::collections::BTreeMap;

use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use serde_json::Value;
use xlai_core::{ErrorKind, MediaSource, TranscriptionRequest, TranscriptionResponse, XlaiError};

use crate::OpenAiConfig;

pub(crate) struct OpenAiTranscriptionRequest {
    pub(crate) model: String,
    pub(crate) audio_bytes: Vec<u8>,
    pub(crate) mime_type: String,
    pub(crate) filename: String,
    language: Option<String>,
    prompt: Option<String>,
    temperature: Option<f32>,
}

impl OpenAiTranscriptionRequest {
    pub(crate) fn from_core_request(
        config: &OpenAiConfig,
        request: TranscriptionRequest,
    ) -> Result<Self, XlaiError> {
        let TranscriptionRequest {
            model,
            audio,
            mime_type,
            filename,
            language,
            prompt,
            temperature,
            metadata: _,
        } = request;

        let (source_mime_type, audio_bytes) = match audio {
            MediaSource::InlineData { mime_type, data } => (mime_type, data),
            MediaSource::Url { .. } => {
                return Err(XlaiError::new(
                    ErrorKind::Unsupported,
                    "openai-compatible transcription requires inline audio bytes",
                ));
            }
        };

        Ok(Self {
            model: model
                .or_else(|| config.transcription_model.clone())
                .unwrap_or_else(|| config.model.clone()),
            audio_bytes,
            mime_type: mime_type.unwrap_or(source_mime_type),
            filename: filename.unwrap_or_else(|| "audio".to_owned()),
            language,
            prompt,
            temperature,
        })
    }

    pub(crate) fn into_multipart_form(self) -> Result<Form, XlaiError> {
        let file_part = Part::bytes(self.audio_bytes)
            .file_name(self.filename)
            .mime_str(&self.mime_type)
            .map_err(|error| {
                XlaiError::new(
                    ErrorKind::Validation,
                    format!("invalid transcription MIME type: {error}"),
                )
            })?;

        let mut form = Form::new()
            .part("file", file_part)
            .text("model", self.model)
            .text("response_format", "json");

        if let Some(language) = self.language {
            form = form.text("language", language);
        }
        if let Some(prompt) = self.prompt {
            form = form.text("prompt", prompt);
        }
        if let Some(temperature) = self.temperature {
            form = form.text("temperature", temperature.to_string());
        }

        Ok(form)
    }
}

#[derive(Deserialize)]
pub(crate) struct OpenAiTranscriptionResponse {
    text: String,
    #[serde(flatten)]
    metadata: BTreeMap<String, Value>,
}

impl OpenAiTranscriptionResponse {
    pub(crate) fn into_core_response(self) -> TranscriptionResponse {
        TranscriptionResponse {
            text: self.text,
            metadata: self.metadata,
        }
    }
}
