use std::collections::BTreeMap;

use serde::Deserialize;
use xlai_core::{
    ChatContent, ChatMessage, ChatResponse, ErrorKind, FinishReason, MessageRole, ToolCall,
    ToolCallChunk, XlaiError,
};

use crate::response::finish_reason_from_api;

pub(crate) struct StreamState {
    pub(crate) message_content: String,
    tool_calls: Vec<PartialToolCall>,
    pub(crate) finish_reason: FinishReason,
}

impl Default for StreamState {
    fn default() -> Self {
        Self {
            message_content: String::new(),
            tool_calls: Vec::new(),
            finish_reason: FinishReason::Completed,
        }
    }
}

impl StreamState {
    pub(crate) fn apply_tool_delta(&mut self, delta: OpenAiStreamToolCallDelta) -> ToolCallChunk {
        let index = delta.index.unwrap_or(0);

        while self.tool_calls.len() <= index {
            self.tool_calls.push(PartialToolCall::default());
        }

        let tool_call = &mut self.tool_calls[index];
        if let Some(id) = delta.id {
            tool_call.id = Some(id);
        }

        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                tool_call.tool_name = Some(name);
            }

            if let Some(arguments) = function.arguments {
                tool_call.arguments.push_str(&arguments);
                return ToolCallChunk {
                    index,
                    id: tool_call.id.clone(),
                    tool_name: tool_call.tool_name.clone(),
                    arguments_delta: arguments,
                };
            }
        }

        ToolCallChunk {
            index,
            id: tool_call.id.clone(),
            tool_name: tool_call.tool_name.clone(),
            arguments_delta: String::new(),
        }
    }

    pub(crate) fn into_chat_response(self) -> Result<ChatResponse, XlaiError> {
        let tool_calls = self
            .tool_calls
            .into_iter()
            .enumerate()
            .map(|(index, call)| call.into_tool_call(index))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ChatResponse {
            message: ChatMessage {
                role: MessageRole::Assistant,
                content: ChatContent::text(self.message_content),
                tool_name: None,
                tool_call_id: None,
                metadata: BTreeMap::new(),
            }
            .with_assistant_tool_calls(&tool_calls),
            tool_calls,
            usage: None,
            finish_reason: self.finish_reason,
            metadata: BTreeMap::new(),
        })
    }
}

#[derive(Default)]
struct PartialToolCall {
    id: Option<String>,
    tool_name: Option<String>,
    arguments: String,
}

impl PartialToolCall {
    fn into_tool_call(self, index: usize) -> Result<ToolCall, XlaiError> {
        let tool_name = self.tool_name.ok_or_else(|| {
            XlaiError::new(
                ErrorKind::Provider,
                format!("streamed tool call at index {index} was missing a name"),
            )
        })?;
        let arguments = serde_json::from_str(&self.arguments).map_err(|error| {
            XlaiError::new(
                ErrorKind::Provider,
                format!("failed to parse streamed tool arguments: {error}"),
            )
        })?;

        Ok(ToolCall {
            id: self.id.unwrap_or_else(|| format!("tool_call_{index}")),
            tool_name,
            arguments,
        })
    }
}

#[derive(Default)]
pub(crate) struct SseParser {
    buffer: String,
}

impl SseParser {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));

        let mut events = Vec::new();
        loop {
            let boundary = self
                .buffer
                .find("\r\n\r\n")
                .map(|i| (i, 4usize))
                .or_else(|| self.buffer.find("\n\n").map(|i| (i, 2usize)));
            let Some((index, sep_len)) = boundary else {
                break;
            };

            let raw_event = self.buffer[..index].to_owned();
            self.buffer.drain(..index + sep_len);

            let data = raw_event
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim)
                .collect::<Vec<_>>()
                .join("");

            if !data.is_empty() {
                events.push(data);
            }
        }

        events
    }
}

#[derive(Deserialize)]
pub(crate) struct OpenAiStreamResponse {
    pub(crate) choices: Vec<OpenAiStreamChoice>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiStreamChoice {
    pub(crate) delta: OpenAiStreamDelta,
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiStreamDelta {
    #[serde(default)]
    pub(crate) content: Option<String>,
    #[serde(default)]
    pub(crate) tool_calls: Option<Vec<OpenAiStreamToolCallDelta>>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiStreamToolCallDelta {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiStreamFunctionDelta>,
}

#[derive(Deserialize)]
struct OpenAiStreamFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

pub(crate) fn update_finish_reason(state: &mut StreamState, reason: Option<&str>) {
    state.finish_reason = finish_reason_from_api(reason);
}

#[cfg(test)]
mod sse_parser_tests {
    use super::SseParser;

    #[test]
    fn splits_events_on_lf_only_delimiter() {
        let mut p = SseParser::default();
        let ev = p.push(b"data: hello\n\n");
        assert_eq!(ev, vec!["hello".to_owned()]);
        assert!(p.buffer.is_empty());
    }

    #[test]
    fn splits_events_on_crlf_delimiter() {
        let mut p = SseParser::default();
        let ev = p.push(b"data: first\r\n\r\ndata: second\r\n\r\n");
        assert_eq!(ev, vec!["first".to_owned(), "second".to_owned()]);
        assert!(p.buffer.is_empty());
    }

    #[test]
    fn crlf_delimiter_does_not_require_lf_lf_substring() {
        // Regression: `\r\n\r\n` contains no `\n\n`, so LF-only splitting never fired.
        let mut p = SseParser::default();
        let payload = b"data: {\"type\":\"speech.audio.delta\",\"audio\":\"YWI=\"}\r\n\r\n";
        let ev = p.push(payload);
        assert_eq!(
            ev,
            vec![r#"{"type":"speech.audio.delta","audio":"YWI="}"#.to_owned()]
        );
    }
}
