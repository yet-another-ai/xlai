//! Runtime tests: streaming chat.
use std::sync::Arc;

use futures_util::StreamExt;
use serde_json::json;
use xlai_core::{
    ChatChunk, ChatResponse, FinishReason, MessageRole, StreamTextDelta, ToolCall, XlaiError,
};

use super::common::*;
use crate::{ChatExecutionEvent, RuntimeBuilder};

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn chat_stream_emits_model_and_tool_events() -> Result<(), XlaiError> {
    let model = Arc::new(StreamingChatModel::new(vec![
        vec![
            ChatChunk::MessageStart {
                role: MessageRole::Assistant,
            },
            ChatChunk::ContentDelta(StreamTextDelta {
                part_index: 0,
                delta: "Looking up weather".to_owned(),
            }),
            ChatChunk::Finished(ChatResponse {
                message: assistant_message("Looking up weather"),
                tool_calls: vec![ToolCall {
                    id: "tool_stream_1".to_owned(),
                    tool_name: "lookup_weather".to_owned(),
                    arguments: json!({ "city": "Paris" }),
                }],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            }),
        ],
        vec![
            ChatChunk::MessageStart {
                role: MessageRole::Assistant,
            },
            ChatChunk::ContentDelta(StreamTextDelta {
                part_index: 0,
                delta: "Paris is sunny.".to_owned(),
            }),
            ChatChunk::Finished(ChatResponse {
                message: assistant_message("Paris is sunny."),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            }),
        ],
    ]));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;

    let mut chat = runtime.chat_session();
    chat.register_tool(weather_tool_definition(), |arguments| async move {
        Ok(xlai_core::ToolResult {
            tool_name: "lookup_weather".to_owned(),
            content: format!(
                "weather for {}: sunny",
                arguments["city"].as_str().unwrap_or("unknown")
            ),
            is_error: false,
            metadata: empty_metadata(),
        })
    });

    let mut stream = chat.stream_prompt("Stream the weather.");
    let mut content_deltas = Vec::new();
    let mut saw_tool_call = false;
    let mut saw_tool_result = false;
    let mut finished_messages = Vec::new();

    while let Some(event) = stream.next().await {
        match event? {
            ChatExecutionEvent::Model(ChatChunk::ContentDelta(StreamTextDelta {
                delta, ..
            })) => {
                content_deltas.push(delta);
            }
            ChatExecutionEvent::Model(ChatChunk::Finished(response)) => {
                finished_messages.push(response.message.content.text_parts_concatenated());
            }
            ChatExecutionEvent::ToolCall(call) => {
                saw_tool_call = true;
                assert_eq!(call.tool_name, "lookup_weather");
            }
            ChatExecutionEvent::ToolResult(result) => {
                saw_tool_result = true;
                assert_eq!(result.content, "weather for Paris: sunny");
            }
            ChatExecutionEvent::Model(
                ChatChunk::MessageStart { .. } | ChatChunk::ToolCallDelta(_),
            ) => {}
        }
    }

    assert_eq!(
        content_deltas,
        vec!["Looking up weather", "Paris is sunny."]
    );
    assert!(saw_tool_call);
    assert!(saw_tool_result);
    assert_eq!(
        finished_messages,
        vec!["Looking up weather", "Paris is sunny."]
    );

    Ok(())
}
