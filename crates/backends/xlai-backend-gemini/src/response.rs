use std::collections::BTreeMap;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::json;
use xlai_core::{
    ChatContent, ChatMessage, ChatResponse, ContentPart, FinishReason, MessageRole, TokenUsage,
    TokenUsageSource, XlaiError,
};

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiChatResponse {
    pub candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiCandidate {
    pub content: Option<GeminiContentResponse>,
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiContentResponse {
    pub parts: Option<Vec<GeminiPartResponse>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiPartResponse {
    pub text: Option<String>,
    #[serde(rename = "inlineData")]
    pub inline_data: Option<GeminiInlineDataResponse>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiInlineDataResponse {
    #[serde(rename = "mimeType")]
    pub mime_type: Option<String>,
    pub data: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiUsageMetadata {
    #[serde(default)]
    prompt_token_count: Option<u32>,
    #[serde(default)]
    candidates_token_count: Option<u32>,
    #[serde(default)]
    total_token_count: Option<u32>,
}

impl From<GeminiUsageMetadata> for TokenUsage {
    fn from(value: GeminiUsageMetadata) -> Self {
        let input_tokens = value.prompt_token_count.unwrap_or(0);
        let output_tokens = value.candidates_token_count.unwrap_or(0);
        Self {
            input_tokens,
            output_tokens,
            total_tokens: value
                .total_token_count
                .unwrap_or_else(|| input_tokens.saturating_add(output_tokens)),
            source: Some(TokenUsageSource::ProviderReported),
        }
    }
}

impl GeminiChatResponse {
    pub(crate) fn into_core_response(self) -> Result<ChatResponse, XlaiError> {
        let mut parts = Vec::new();
        let mut finish_reason_str = None;
        let mut metadata = BTreeMap::new();
        if let Some(usage_metadata) = &self.usage_metadata {
            metadata.insert("usage_metadata".to_owned(), json!(usage_metadata));
        }

        if let Some(candidates) = self.candidates
            && let Some(candidate) = candidates.into_iter().next()
        {
            finish_reason_str = candidate.finish_reason;
            if let Some(content) = candidate.content
                && let Some(gemini_parts) = content.parts
            {
                for part in gemini_parts {
                    if let Some(text) = part.text {
                        parts.push(ContentPart::Text { text });
                    } else if let Some(inline_data) = part.inline_data
                        && let (Some(mime_type), Some(data)) =
                            (inline_data.mime_type, inline_data.data)
                        && let Ok(decoded) = STANDARD.decode(data)
                    {
                        parts.push(ContentPart::Image {
                            source: xlai_core::MediaSource::InlineData {
                                mime_type: mime_type.clone(),
                                data: decoded,
                            },
                            mime_type: Some(mime_type),
                            detail: None,
                        });
                    }
                }
            }
        }

        let finish_reason = match finish_reason_str.as_deref() {
            Some("STOP") => FinishReason::Completed,
            Some("MAX_TOKENS") => FinishReason::Length,
            _ => FinishReason::Stopped,
        };

        Ok(ChatResponse {
            message: ChatMessage {
                role: MessageRole::Assistant,
                content: ChatContent { parts },
                tool_name: None,
                tool_call_id: None,
                metadata: Default::default(),
            },
            tool_calls: Vec::new(),
            usage: self.usage_metadata.map(Into::into),
            finish_reason,
            metadata,
        })
    }
}
