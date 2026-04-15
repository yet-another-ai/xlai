use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use xlai_core::{
    ErrorKind, GeneratedImage, ImageGenerationOutputFormat, ImageGenerationRequest,
    ImageGenerationResponse, MediaSource, XlaiError,
};

#[derive(Debug, Serialize)]
pub(crate) struct GeminiImageGenerationRequest {
    pub contents: Vec<GeminiImageContent>,
    #[serde(rename = "generationConfig")]
    pub generation_config: GeminiImageGenerationConfig,
}

#[derive(Debug, Serialize)]
pub(crate) struct GeminiImageContent {
    pub parts: Vec<GeminiImagePart>,
}

#[derive(Debug, Serialize)]
pub(crate) struct GeminiImagePart {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct GeminiImageGenerationConfig {
    #[serde(rename = "responseModalities")]
    pub response_modalities: Vec<String>,
    #[serde(rename = "imageConfig", skip_serializing_if = "Option::is_none")]
    pub image_config: Option<GeminiImageConfig>,
}

#[derive(Debug, Serialize)]
pub(crate) struct GeminiImageConfig {
    #[serde(rename = "aspectRatio", skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(rename = "imageSize", skip_serializing_if = "Option::is_none")]
    pub image_size: Option<String>,
    #[serde(rename = "numberOfImages", skip_serializing_if = "Option::is_none")]
    pub number_of_images: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiImageGenerationResponse {
    pub candidates: Option<Vec<GeminiImageCandidate>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiImageCandidate {
    pub content: Option<GeminiImageContentResponse>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiImageContentResponse {
    pub parts: Option<Vec<GeminiImagePartResponse>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiImagePartResponse {
    #[serde(rename = "inlineData")]
    pub inline_data: Option<GeminiImageInlineDataResponse>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiImageInlineDataResponse {
    #[serde(rename = "mimeType")]
    pub mime_type: Option<String>,
    pub data: Option<String>,
}

impl GeminiImageGenerationRequest {
    pub(crate) fn from_core_request(request: ImageGenerationRequest) -> Result<Self, XlaiError> {
        if matches!(request.count, Some(0)) {
            return Err(XlaiError::new(
                ErrorKind::Validation,
                "image generation count must be at least 1",
            ));
        }

        if request.background.is_some() {
            return Err(XlaiError::new(
                ErrorKind::Unsupported,
                "Gemini backend does not support specifying image background",
            ));
        }

        let mut image_config = GeminiImageConfig {
            aspect_ratio: None,
            image_size: None,
            number_of_images: request.count,
        };

        if let Some(size) = request.size {
            // Gemini expects aspect ratios like "1:1", "16:9", etc. or imageSize like "1K", "2K".
            // We'll pass the size directly as aspectRatio if it contains a colon, otherwise as imageSize.
            if size.contains(':') {
                image_config.aspect_ratio = Some(size);
            } else {
                image_config.image_size = Some(size);
            }
        }

        Ok(Self {
            contents: vec![GeminiImageContent {
                parts: vec![GeminiImagePart {
                    text: request.prompt,
                }],
            }],
            generation_config: GeminiImageGenerationConfig {
                response_modalities: vec!["IMAGE".to_owned()],
                image_config: Some(image_config),
            },
        })
    }
}

impl GeminiImageGenerationResponse {
    pub(crate) fn into_core_response(
        self,
        requested_format: Option<ImageGenerationOutputFormat>,
    ) -> Result<ImageGenerationResponse, XlaiError> {
        let mut images = Vec::new();

        if let Some(candidates) = self.candidates {
            if let Some(candidate) = candidates.into_iter().next() {
                if let Some(content) = candidate.content {
                    if let Some(parts) = content.parts {
                        for part in parts {
                            if let Some(inline_data) = part.inline_data {
                                if let (Some(mime_type), Some(data)) =
                                    (inline_data.mime_type, inline_data.data)
                                {
                                    let mut decoded = STANDARD.decode(data).map_err(|error| {
                                        XlaiError::new(
                                            ErrorKind::Provider,
                                            format!("failed to decode Gemini image payload: {error}"),
                                        )
                                    })?;

                                    let mut final_mime_type = mime_type.clone();

                                    if let Some(format) = requested_format {
                                        let target_mime = match format {
                                            ImageGenerationOutputFormat::Png => "image/png",
                                            ImageGenerationOutputFormat::Jpeg => "image/jpeg",
                                            ImageGenerationOutputFormat::Webp => "image/webp",
                                        };

                                        if target_mime != mime_type {
                                            let img = image::load_from_memory(&decoded).map_err(|error| {
                                                XlaiError::new(
                                                    ErrorKind::Provider,
                                                    format!("failed to decode image for format conversion: {error}"),
                                                )
                                            })?;

                                            let mut cursor = std::io::Cursor::new(Vec::new());
                                            let output_format = match format {
                                                ImageGenerationOutputFormat::Png => image::ImageFormat::Png,
                                                ImageGenerationOutputFormat::Jpeg => image::ImageFormat::Jpeg,
                                                ImageGenerationOutputFormat::Webp => image::ImageFormat::WebP,
                                            };

                                            img.write_to(&mut cursor, output_format).map_err(|error| {
                                                XlaiError::new(
                                                    ErrorKind::Provider,
                                                    format!("failed to encode image to requested format: {error}"),
                                                )
                                            })?;

                                            decoded = cursor.into_inner();
                                            final_mime_type = target_mime.to_owned();
                                        }
                                    }

                                    images.push(GeneratedImage {
                                        image: MediaSource::InlineData {
                                            mime_type: final_mime_type.clone(),
                                            data: decoded,
                                        },
                                        mime_type: Some(final_mime_type),
                                        revised_prompt: None,
                                        metadata: Default::default(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        if images.is_empty() {
            return Err(XlaiError::new(
                ErrorKind::Provider,
                "Gemini image response contained no images",
            ));
        }

        Ok(ImageGenerationResponse {
            images,
            metadata: Default::default(),
        })
    }
}
