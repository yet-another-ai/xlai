//! Execution hint merge, cancellation, and runtime defaults.

use std::sync::Arc;

use futures_util::StreamExt;
use xlai_core::{
    CancellationSignal, ChatChunk, ChatContent, ChatExecutionConfig, ChatExecutionOverrides,
    ChatMessage, ChatRequest, ChatResponse, ErrorKind, ExecutionLatencyMode, FinishReason,
    MessageRole, StreamTextDelta, XlaiError,
};

use super::common::*;
use crate::RuntimeBuilder;

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn execution_defaults_merge_into_outgoing_request() -> Result<(), XlaiError> {
    let reqs = Arc::new(std::sync::Mutex::new(Vec::new()));
    let runtime = RuntimeBuilder::new()
        .with_chat_model(Arc::new(RecordingChatModel::new(
            reqs.clone(),
            vec![ChatResponse {
                message: assistant_message("ok"),
                tool_calls: vec![],
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            }],
        )))
        .with_chat_execution_defaults(ChatExecutionOverrides {
            latency_mode: Some(ExecutionLatencyMode::Interactive),
            ..Default::default()
        })
        .build()?;

    let chat = runtime
        .chat_session()
        .with_chat_execution_overrides(ChatExecutionOverrides {
            cancel_on_drop: Some(true),
            ..Default::default()
        });

    chat.execute(vec![ChatMessage {
        role: MessageRole::User,
        content: ChatContent::text("hi"),
        tool_name: None,
        tool_call_id: None,
        metadata: Default::default(),
    }])
    .await?;

    let r = lock_unpoisoned(&reqs);
    let req = &r[0];
    let exec = req.execution.as_ref().ok_or_else(|| {
        XlaiError::new(
            ErrorKind::Provider,
            "expected merged execution on outgoing ChatRequest",
        )
    })?;
    assert_eq!(exec.latency_mode, ExecutionLatencyMode::Interactive);
    assert!(exec.cancel_on_drop);
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn stream_chat_stops_on_cancellation_signal() -> Result<(), XlaiError> {
    let model = Arc::new(StreamingChatModel::new(vec![vec![
        ChatChunk::MessageStart {
            role: MessageRole::Assistant,
            message_index: 0,
        },
        ChatChunk::ContentDelta(StreamTextDelta {
            message_index: 0,
            part_index: 0,
            delta: "a".to_owned(),
        }),
        ChatChunk::ContentDelta(StreamTextDelta {
            message_index: 0,
            part_index: 0,
            delta: "b".to_owned(),
        }),
        ChatChunk::Finished(ChatResponse {
            message: assistant_message("ab"),
            tool_calls: vec![],
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }),
    ]]));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;

    let cancel = CancellationSignal::new();
    let mut request = ChatRequest {
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::text("hi"),
            tool_name: None,
            tool_call_id: None,
            metadata: Default::default(),
        }],
        cancellation: Some(cancel.clone()),
        ..Default::default()
    };
    request.execution = Some(ChatExecutionConfig::default());

    let mut stream = runtime.stream_chat(request)?;
    let first = stream
        .next()
        .await
        .ok_or_else(|| XlaiError::new(ErrorKind::Provider, "stream ended before first chunk"))?;
    first?;
    cancel.cancel();
    let second = stream
        .next()
        .await
        .ok_or_else(|| XlaiError::new(ErrorKind::Provider, "stream ended before cancel error"))?;
    let err = second.err().ok_or_else(|| {
        XlaiError::new(
            ErrorKind::Provider,
            "expected cancelled error on second chunk, got Ok",
        )
    })?;
    assert_eq!(err.kind, ErrorKind::Cancelled);
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_exposes_default_max_tool_round_trips() -> Result<(), XlaiError> {
    let runtime = RuntimeBuilder::new()
        .with_chat_model(Arc::new(RecordingChatModel::new(
            Arc::new(std::sync::Mutex::new(Vec::new())),
            vec![ChatResponse {
                message: assistant_message("x"),
                tool_calls: vec![],
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            }],
        )))
        .with_default_max_tool_round_trips(11)
        .build()?;

    assert_eq!(runtime.default_max_tool_round_trips(), Some(11));
    Ok(())
}
