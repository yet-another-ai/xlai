use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Vision / image detail hint for providers that support it (e.g. OpenAI).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    #[default]
    Auto,
    Low,
    High,
}

/// Binary or remote reference for multimodal parts.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum MediaSource {
    Url {
        url: String,
    },
    InlineData {
        mime_type: String,
        /// Inline bytes. JSON: base64 string field `data`. CBOR: byte string.
        #[serde(with = "crate::serde_bytes_format")]
        data: Vec<u8>,
    },
}

/// One segment of a multimodal user/assistant/system message.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    Audio {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
    File {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

/// Message body as an ordered list of multimodal parts.
///
/// For JSON, a plain string still deserializes as a single [`ContentPart::Text`].
#[derive(Clone, Debug, PartialEq)]
pub struct ChatContent {
    pub parts: Vec<ContentPart>,
}

impl ChatContent {
    #[must_use]
    pub fn empty() -> Self {
        Self { parts: Vec::new() }
    }

    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            parts: vec![ContentPart::Text { text: text.into() }],
        }
    }

    #[must_use]
    pub fn from_parts(parts: Vec<ContentPart>) -> Self {
        Self { parts }
    }

    /// Concatenates all [`ContentPart::Text`] segments in order.
    #[must_use]
    pub fn text_parts_concatenated(&self) -> String {
        self.parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns the single text part if this is exactly one text block.
    #[must_use]
    pub fn as_single_text(&self) -> Option<&str> {
        match self.parts.as_slice() {
            [ContentPart::Text { text }] => Some(text.as_str()),
            _ => None,
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamTextDelta {
    /// Index into the assembled message [`ChatContent::parts`] for this stream.
    pub part_index: usize,
    pub delta: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
enum ChatContentSerde {
    Plain(String),
    WithParts {
        parts: Vec<ContentPart>,
    },
    /// JSON array of content parts, e.g. `[{"type":"text","text":"hi"}]`.
    PartsOnly(Vec<ContentPart>),
}

impl Serialize for ChatContent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if let Some(text) = self.as_single_text() {
            serializer.serialize_str(text)
        } else {
            ChatContentSerde::WithParts {
                parts: self.parts.clone(),
            }
            .serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for ChatContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match ChatContentSerde::deserialize(deserializer)? {
            ChatContentSerde::Plain(text) => Ok(Self::text(text)),
            ChatContentSerde::WithParts { parts } | ChatContentSerde::PartsOnly(parts) => {
                Ok(Self { parts })
            }
        }
    }
}

impl From<String> for ChatContent {
    fn from(value: String) -> Self {
        Self::text(value)
    }
}

impl From<&str> for ChatContent {
    fn from(value: &str) -> Self {
        Self::text(value)
    }
}
