use std::collections::BTreeMap;

use serde_json::Value;
use xlai_core::{
    ChatContent, ChatMessage, ChatResponse, ErrorKind, FinishReason, MessageRole, ToolCall,
    ToolCallChunk, XlaiError,
};

use crate::response::{
    OpenAiChatResponse, attach_response_output_items, finish_reason_from_api,
    openai_response_output_to_chat,
};

pub(crate) struct StreamState {
    pub(crate) message_content: String,
    tool_calls: Vec<PartialToolCall>,
    pub(crate) finish_reason: FinishReason,
    output_items: Vec<Value>,
}

impl Default for StreamState {
    fn default() -> Self {
        Self {
            message_content: String::new(),
            tool_calls: Vec::new(),
            finish_reason: FinishReason::Completed,
            output_items: Vec::new(),
        }
    }
}

impl StreamState {
    pub(crate) fn ensure_tool_call_slot(&mut self, index: usize) {
        while self.tool_calls.len() <= index {
            self.tool_calls.push(PartialToolCall::default());
        }
    }

    pub(crate) fn apply_tool_call_added(
        &mut self,
        index: usize,
        call_id: Option<String>,
        tool_name: Option<String>,
        arguments: Option<String>,
    ) -> ToolCallChunk {
        self.ensure_tool_call_slot(index);
        let tool_call = &mut self.tool_calls[index];
        if let Some(id) = call_id {
            tool_call.id = Some(id);
        }
        if let Some(name) = tool_name {
            tool_call.tool_name = Some(name);
        }
        if let Some(arguments) = arguments {
            tool_call.arguments.push_str(&arguments);
        }
        ToolCallChunk {
            index,
            id: tool_call.id.clone(),
            tool_name: tool_call.tool_name.clone(),
            arguments_delta: String::new(),
        }
    }

    pub(crate) fn apply_tool_delta(&mut self, index: usize, delta: String) -> ToolCallChunk {
        self.ensure_tool_call_slot(index);
        let tool_call = &mut self.tool_calls[index];
        tool_call.arguments.push_str(&delta);
        ToolCallChunk {
            index,
            id: tool_call.id.clone(),
            tool_name: tool_call.tool_name.clone(),
            arguments_delta: delta,
        }
    }

    pub(crate) fn push_output_item(&mut self, item: Value) {
        self.output_items.push(item);
    }

    pub(crate) fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    pub(crate) fn into_chat_response(self) -> Result<ChatResponse, XlaiError> {
        let (content, tool_calls) = if self.output_items.is_empty() {
            let tool_calls = self
                .tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, call)| call.into_tool_call(index))
                .collect::<Result<Vec<_>, _>>()?;
            (ChatContent::text(self.message_content), tool_calls)
        } else {
            openai_response_output_to_chat(&self.output_items)?
        };

        Ok(ChatResponse {
            message: attach_response_output_items(
                ChatMessage {
                    role: MessageRole::Assistant,
                    content,
                    tool_name: None,
                    tool_call_id: None,
                    metadata: BTreeMap::new(),
                }
                .with_assistant_tool_calls(&tool_calls),
                &self.output_items,
            ),
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

pub(crate) fn update_finish_reason(
    state: &mut StreamState,
    status: Option<&str>,
    incomplete_reason: Option<&str>,
    has_tool_calls: bool,
) {
    state.finish_reason = finish_reason_from_api(status, incomplete_reason, has_tool_calls);
}

pub(crate) fn maybe_completed_response(event: &Value) -> Result<Option<ChatResponse>, XlaiError> {
    let Some("response.completed") = event.get("type").and_then(Value::as_str) else {
        return Ok(None);
    };
    let Some(response) = event.get("response") else {
        return Ok(None);
    };
    let parsed: OpenAiChatResponse = serde_json::from_value(response.clone()).map_err(|error| {
        XlaiError::new(
            ErrorKind::Provider,
            format!("failed to parse completed response event: {error}"),
        )
    })?;
    parsed.into_core_response().map(Some)
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
