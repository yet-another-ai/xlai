use serde::Serialize;
use xlai_core::{ChatRequest, ContentPart, ErrorKind, MessageRole, XlaiError};

#[derive(Debug, Serialize)]
pub(crate) struct GeminiChatRequest {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
    // TODO: tool support
}

#[derive(Debug, Serialize)]
pub(crate) struct GeminiContent {
    pub role: &'static str,
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum GeminiPart {
    Text { text: String },
    InlineData { inline_data: GeminiInlineData },
}

#[derive(Debug, Serialize)]
pub(crate) struct GeminiInlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Serialize, Default)]
pub(crate) struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
}

impl GeminiChatRequest {
    pub(crate) fn from_core_request(request: ChatRequest) -> Result<Self, XlaiError> {
        let mut contents = Vec::new();
        let mut system_parts = Vec::new();

        for message in request.messages {
            match message.role {
                MessageRole::System => {
                    for part in message.content.parts {
                        if let ContentPart::Text { text } = part {
                            system_parts.push(GeminiPart::Text { text });
                        } else {
                            return Err(XlaiError::new(
                                ErrorKind::Unsupported,
                                "Gemini backend only supports text in system instructions",
                            ));
                        }
                    }
                }
                MessageRole::User => {
                    contents.push(GeminiContent {
                        role: "user",
                        parts: message
                            .content
                            .parts
                            .into_iter()
                            .map(gemini_part_from_core)
                            .collect::<Result<Vec<_>, _>>()?,
                    });
                }
                MessageRole::Assistant => {
                    contents.push(GeminiContent {
                        role: "model",
                        parts: message
                            .content
                            .parts
                            .into_iter()
                            .map(gemini_part_from_core)
                            .collect::<Result<Vec<_>, _>>()?,
                    });
                }
                MessageRole::Tool => {
                    return Err(XlaiError::new(
                        ErrorKind::Unsupported,
                        "Gemini backend does not yet support tool messages",
                    ));
                }
            }
        }

        let system_instruction = if system_parts.is_empty() {
            None
        } else {
            Some(GeminiContent {
                role: "user", // System instruction role is typically "user" or omitted, but Gemini API accepts it as a Content object.
                parts: system_parts,
            })
        };

        let mut generation_config = GeminiGenerationConfig::default();
        let mut has_config = false;

        if let Some(temp) = request.temperature {
            generation_config.temperature = Some(temp);
            has_config = true;
        }
        if let Some(max_tokens) = request.max_output_tokens {
            generation_config.max_output_tokens = Some(max_tokens);
            has_config = true;
        }

        Ok(Self {
            contents,
            system_instruction,
            generation_config: if has_config {
                Some(generation_config)
            } else {
                None
            },
        })
    }
}

fn gemini_part_from_core(part: ContentPart) -> Result<GeminiPart, XlaiError> {
    match part {
        ContentPart::Text { text } => Ok(GeminiPart::Text { text }),
        ContentPart::Image { source, .. } => match source {
            xlai_core::MediaSource::InlineData { mime_type, data } => {
                use base64::{Engine as _, engine::general_purpose::STANDARD};
                Ok(GeminiPart::InlineData {
                    inline_data: GeminiInlineData {
                        mime_type,
                        data: STANDARD.encode(data),
                    },
                })
            }
            xlai_core::MediaSource::Url { .. } => Err(XlaiError::new(
                ErrorKind::Unsupported,
                "Gemini backend does not support image URLs",
            )),
        },
        _ => Err(XlaiError::new(
            ErrorKind::Unsupported,
            "Gemini backend does not yet support this content part",
        )),
    }
}
