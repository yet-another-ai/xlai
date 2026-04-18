use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use xlai_core::{
    ErrorKind, GeneratedImage, ImageGenerationBackground, ImageGenerationOutputFormat,
    ImageGenerationQuality, ImageGenerationRequest, ImageGenerationResponse, MediaSource, Metadata,
    XlaiError,
};

use crate::OpenAiConfig;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OpenAiImageGenerationRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<OpenAiImageGenerationQuality>,
    #[serde(skip_serializing_if = "Option::is_none")]
    background: Option<OpenAiImageGenerationBackground>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_format: Option<OpenAiImageGenerationOutputFormat>,
    #[serde(rename = "n", skip_serializing_if = "Option::is_none")]
    count: Option<u32>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum OpenAiImageGenerationQuality {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum OpenAiImageGenerationBackground {
    Transparent,
    Opaque,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum OpenAiImageGenerationOutputFormat {
    Png,
    Jpeg,
    Webp,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct OpenAiImageGenerationResponse {
    #[serde(default)]
    created: Option<i64>,
    #[serde(default)]
    data: Vec<OpenAiGeneratedImage>,
    #[serde(flatten)]
    extra: Metadata,
}

#[derive(Clone, Debug, Deserialize)]
struct OpenAiGeneratedImage {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
    #[serde(flatten)]
    extra: Metadata,
}

impl OpenAiImageGenerationRequest {
    pub(crate) fn from_core_request(
        config: &OpenAiConfig,
        request: ImageGenerationRequest,
    ) -> Result<Self, XlaiError> {
        if matches!(request.count, Some(0)) {
            return Err(XlaiError::new(
                ErrorKind::Validation,
                "image generation count must be at least 1",
            ));
        }

        Ok(Self {
            model: request
                .model
                .or_else(|| config.image_model.clone())
                .unwrap_or_else(|| config.model.clone()),
            prompt: request.prompt,
            size: request.size,
            quality: request.quality.map(Into::into),
            background: request.background.map(Into::into),
            output_format: request.output_format.map(Into::into),
            count: request.count,
        })
    }
}

impl OpenAiImageGenerationResponse {
    pub(crate) fn into_core_response(
        self,
        requested_format: Option<ImageGenerationOutputFormat>,
    ) -> Result<ImageGenerationResponse, XlaiError> {
        let mut metadata = self.extra;
        if let Some(created) = self.created {
            metadata.insert("created".to_owned(), Value::from(created));
        }

        let mut images = Vec::with_capacity(self.data.len());
        for image in self.data {
            images.push(image.into_core_image(requested_format)?);
        }

        Ok(ImageGenerationResponse { images, metadata })
    }
}

impl OpenAiGeneratedImage {
    fn into_core_image(
        self,
        requested_format: Option<ImageGenerationOutputFormat>,
    ) -> Result<GeneratedImage, XlaiError> {
        let mime_type = requested_format
            .map(mime_for_output_format)
            .map(str::to_owned);
        let image = if let Some(data) = self.b64_json {
            MediaSource::InlineData {
                mime_type: mime_type
                    .clone()
                    .unwrap_or_else(|| "application/octet-stream".to_owned()),
                data: STANDARD.decode(data).map_err(|error| {
                    XlaiError::new(
                        ErrorKind::Provider,
                        format!("failed to decode OpenAI image payload: {error}"),
                    )
                })?,
            }
        } else if let Some(url) = self.url {
            MediaSource::Url { url }
        } else {
            return Err(XlaiError::new(
                ErrorKind::Provider,
                "OpenAI image response item missing both b64_json and url",
            ));
        };

        Ok(GeneratedImage {
            image,
            mime_type,
            revised_prompt: self.revised_prompt,
            metadata: self.extra,
        })
    }
}

fn mime_for_output_format(format: ImageGenerationOutputFormat) -> &'static str {
    match format {
        ImageGenerationOutputFormat::Png => "image/png",
        ImageGenerationOutputFormat::Jpeg => "image/jpeg",
        ImageGenerationOutputFormat::Webp => "image/webp",
    }
}

impl From<ImageGenerationQuality> for OpenAiImageGenerationQuality {
    fn from(value: ImageGenerationQuality) -> Self {
        match value {
            ImageGenerationQuality::Low => Self::Low,
            ImageGenerationQuality::Medium => Self::Medium,
            ImageGenerationQuality::High => Self::High,
        }
    }
}

impl From<ImageGenerationBackground> for OpenAiImageGenerationBackground {
    fn from(value: ImageGenerationBackground) -> Self {
        match value {
            ImageGenerationBackground::Transparent => Self::Transparent,
            ImageGenerationBackground::Opaque => Self::Opaque,
        }
    }
}

impl From<ImageGenerationOutputFormat> for OpenAiImageGenerationOutputFormat {
    fn from(value: ImageGenerationOutputFormat) -> Self {
        match value {
            ImageGenerationOutputFormat::Png => Self::Png,
            ImageGenerationOutputFormat::Jpeg => Self::Jpeg,
            ImageGenerationOutputFormat::Webp => Self::Webp,
        }
    }
}
