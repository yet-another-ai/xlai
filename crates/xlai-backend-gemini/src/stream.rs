use serde_json::Value;
use xlai_core::{
    ChatChunk, ChatMessage, ChatResponse, ContentPart, ErrorKind, FinishReason, MessageRole,
    StreamTextDelta, XlaiError,
};

#[derive(Default)]
pub(crate) struct SseParser {
    buffer: String,
}

impl SseParser {
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        let mut events = Vec::new();
        self.buffer.push_str(&String::from_utf8_lossy(chunk));

        while let Some(index) = self.buffer.find("\n\n") {
            let event_str = self.buffer[..index].to_owned();
            self.buffer = self.buffer[index + 2..].to_owned();

            for line in event_str.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    events.push(data.to_owned());
                }
            }
        }

        events
    }
}

#[derive(Default)]
pub(crate) struct StreamState {
    message_started: bool,
    content: String,
    finish_reason: Option<String>,
}

impl StreamState {
    pub(crate) fn process_event(&mut self, event: &str) -> Result<Vec<ChatChunk>, XlaiError> {
        let mut chunks = Vec::new();

        if event == "[DONE]" {
            return Ok(chunks);
        }

        let value: Value = serde_json::from_str(event).map_err(|error| {
            XlaiError::new(
                ErrorKind::Provider,
                format!("failed to parse stream event: {error}"),
            )
        })?;

        if let Some(candidates) = value.get("candidates").and_then(Value::as_array)
            && let Some(candidate) = candidates.first() {
                if let Some(content) = candidate.get("content")
                    && let Some(parts) = content.get("parts").and_then(Value::as_array) {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(Value::as_str) {
                                if !self.message_started {
                                    self.message_started = true;
                                    chunks.push(ChatChunk::MessageStart {
                                        role: MessageRole::Assistant,
                                        message_index: 0,
                                    });
                                }

                                self.content.push_str(text);
                                chunks.push(ChatChunk::ContentDelta(StreamTextDelta {
                                    message_index: 0,
                                    part_index: 0,
                                    delta: text.to_owned(),
                                }));
                            }
                        }
                    }

                if let Some(finish_reason) = candidate.get("finishReason").and_then(Value::as_str) {
                    self.finish_reason = Some(finish_reason.to_owned());
                }
            }

        Ok(chunks)
    }

    pub(crate) fn into_chat_response(self) -> Result<ChatResponse, XlaiError> {
        let finish_reason = match self.finish_reason.as_deref() {
            Some("STOP") => FinishReason::Completed,
            Some("MAX_TOKENS") => FinishReason::Length,
            _ => FinishReason::Stopped,
        };

        Ok(ChatResponse {
            message: ChatMessage {
                role: MessageRole::Assistant,
                content: xlai_core::ChatContent {
                    parts: vec![ContentPart::Text { text: self.content }],
                },
                tool_name: None,
                tool_call_id: None,
                metadata: Default::default(),
            },
            tool_calls: Vec::new(),
            usage: None,
            finish_reason,
            metadata: Default::default(),
        })
    }
}
