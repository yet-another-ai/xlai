//! Runtime tests: agent and MCP.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use serde_json::json;
use xlai_core::{
    ChatContent, ChatMessage, ChatResponse, ErrorKind, FinishReason, MessageRole, ToolCall,
    XlaiError,
};

use super::common::*;
use crate::RuntimeBuilder;

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_session_injects_skill_tool_when_skill_store_is_configured() -> Result<(), XlaiError>
{
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("agent reply"),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new()
        .with_chat_model(model)
        .with_skill_store(seed_markdown_skill_store().await?)
        .build()?;

    let agent = runtime.agent_session()?;
    let response = agent.prompt("Hello").await?;
    assert_eq!(
        response.message.content.as_single_text(),
        Some("agent reply")
    );

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0]
            .available_tools
            .iter()
            .any(|tool| tool.name == "skill"),
        "expected agent sessions to expose the built-in skill tool"
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_skill_tool_uses_configured_skill_store() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![ToolCall {
                    id: "skill_1".to_owned(),
                    tool_name: "skill".to_owned(),
                    arguments: json!({
                        "skill": "review.code",
                        "args": "Focus on correctness."
                    }),
                }],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("skill loaded"),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let runtime = RuntimeBuilder::new()
        .with_chat_model(model)
        .with_skill_store(seed_markdown_skill_store().await?)
        .build()?;

    let agent = runtime.agent_session()?;
    let response = agent_stream_prompt_final_response(&agent, "Review this patch").await?;
    assert_eq!(
        response.message.content.as_single_text(),
        Some("skill loaded")
    );

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 2);
    // Second model round: [user, assistant, system_reminder, tool]
    assert_eq!(requests[1].messages.len(), 4);
    assert_eq!(requests[1].messages[3].tool_name.as_deref(), Some("skill"));
    assert!(
        requests[1].messages[3]
            .content
            .text_parts_concatenated()
            .contains("Prioritize bugs, regressions, and missing tests."),
        "expected the skill tool result to include the resolved prompt fragment"
    );
    assert!(
        requests[1].messages[3]
            .content
            .text_parts_concatenated()
            .contains("Focus on correctness."),
        "expected the skill tool result to include the supplied skill arguments"
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_mcp_registry_exposes_registered_tools_alongside_builtins() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("agent reply"),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new()
        .with_chat_model(model)
        .with_skill_store(seed_markdown_skill_store().await?)
        .build()?;

    let mut agent = runtime.agent_session()?;
    agent
        .mcp_registry()
        .register_tool(weather_tool_definition(), |_| async {
            Ok(xlai_core::ToolResult {
                tool_name: "lookup_weather".to_owned(),
                content: "weather for Paris: sunny".to_owned(),
                is_error: false,
                metadata: empty_metadata(),
            })
        });

    let mcp_tools = agent.mcp_registry().tool_definitions();
    assert_eq!(mcp_tools.len(), 1);
    assert_eq!(mcp_tools[0].name, "lookup_weather");

    let response = agent.prompt("Hello").await?;
    assert_eq!(
        response.message.content.as_single_text(),
        Some("agent reply")
    );

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0]
            .available_tools
            .iter()
            .any(|tool| tool.name == "skill"),
        "expected agent sessions to keep exposing built-in tools"
    );
    assert!(
        requests[0]
            .available_tools
            .iter()
            .any(|tool| tool.name == "lookup_weather"),
        "expected agent sessions to expose MCP-registered tools"
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_mcp_registry_executes_registered_tool_calls() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![ToolCall {
                    id: "tool_1".to_owned(),
                    tool_name: "lookup_weather".to_owned(),
                    arguments: json!({ "city": "Paris" }),
                }],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("Paris is sunny."),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let mut agent = runtime.agent_session()?;
    agent
        .mcp_registry()
        .register_tool(weather_tool_definition(), |arguments| async move {
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

    let response =
        agent_stream_prompt_final_response(&agent, "What's the weather in Paris?").await?;
    assert_eq!(
        response.message.content.as_single_text(),
        Some("Paris is sunny.")
    );

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].messages.len(), 3);
    assert_eq!(
        requests[1].messages[2].tool_name.as_deref(),
        Some("lookup_weather")
    );
    assert_eq!(
        requests[1].messages[2].content.as_single_text(),
        Some("weather for Paris: sunny")
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_prompt_never_executes_tool_callbacks() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("need tool"),
            tool_calls: vec![ToolCall {
                id: "w1".to_owned(),
                tool_name: "lookup_weather".to_owned(),
                arguments: json!({ "city": "Paris" }),
            }],
            usage: None,
            finish_reason: FinishReason::ToolCalls,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let mut agent = runtime.agent_session()?;
    let invoked = Arc::new(Mutex::new(false));
    let invoked_cb = invoked.clone();
    agent.register_tool(weather_tool_definition(), move |_arguments| {
        let invoked_cb = invoked_cb.clone();
        async move {
            *lock_unpoisoned(&invoked_cb) = true;
            Ok(xlai_core::ToolResult {
                tool_name: "lookup_weather".to_owned(),
                content: "nope".to_owned(),
                is_error: false,
                metadata: empty_metadata(),
            })
        }
    });

    let response = agent.prompt("Weather?").await?;
    assert_eq!(response.finish_reason, FinishReason::ToolCalls);
    assert!(!response.tool_calls.is_empty());
    assert!(!*lock_unpoisoned(&invoked));
    assert_eq!(lock_unpoisoned(&requests).len(), 1);

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_register_tool_shorthand_routes_through_mcp_registry() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![ToolCall {
                    id: "tool_1".to_owned(),
                    tool_name: "lookup_weather".to_owned(),
                    arguments: json!({ "city": "Paris" }),
                }],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("Paris is sunny."),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let mut agent = runtime.agent_session()?;
    agent.register_tool(weather_tool_definition(), |arguments| async move {
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

    let mcp_tools = agent.mcp_registry().tool_definitions();
    assert_eq!(mcp_tools.len(), 1);
    assert_eq!(mcp_tools[0].name, "lookup_weather");

    let response =
        agent_stream_prompt_final_response(&agent, "What's the weather in Paris?").await?;
    assert_eq!(
        response.message.content.as_single_text(),
        Some("Paris is sunny.")
    );

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 2);
    assert!(
        requests[0]
            .available_tools
            .iter()
            .any(|tool| tool.name == "lookup_weather"),
        "expected shorthand registration to expose an MCP tool to the model"
    );
    assert_eq!(
        requests[1].messages[2].tool_name.as_deref(),
        Some("lookup_weather")
    );
    assert_eq!(
        requests[1].messages[2].content.as_single_text(),
        Some("weather for Paris: sunny")
    );

    Ok(())
}

fn user_message(text: &str) -> ChatMessage {
    ChatMessage {
        role: MessageRole::User,
        content: ChatContent::text(text),
        tool_name: None,
        tool_call_id: None,
        metadata: empty_metadata(),
    }
}

/// Simulates a leaked internal reminder row that must not appear in user-managed history.
fn leaked_internal_system_reminder(content: &str) -> ChatMessage {
    let mut metadata = empty_metadata();
    metadata.insert("xlai_system_reminder".to_owned(), json!(true));
    ChatMessage {
        role: MessageRole::System,
        content: ChatContent::text(content),
        tool_name: None,
        tool_call_id: None,
        metadata,
    }
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_context_compressor_runs_once_per_stream_round() -> Result<(), XlaiError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![ToolCall {
                    id: "t1".to_owned(),
                    tool_name: "lookup_weather".to_owned(),
                    arguments: json!({ "city": "Paris" }),
                }],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("done"),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let calls_cb = Arc::clone(&calls);
    let mut agent = runtime
        .agent_session()?
        .with_context_compressor(move |msgs, est| {
            let calls_cb = Arc::clone(&calls_cb);
            async move {
                calls_cb.fetch_add(1, Ordering::SeqCst);
                assert!(
                    est.is_some(),
                    "expected a best-effort token estimate for non-empty context"
                );
                Ok(msgs)
            }
        });
    agent.register_tool(weather_tool_definition(), |arguments| async move {
        Ok(xlai_core::ToolResult {
            tool_name: "lookup_weather".to_owned(),
            content: format!(
                "weather for {}: ok",
                arguments["city"].as_str().unwrap_or("?")
            ),
            is_error: false,
            metadata: empty_metadata(),
        })
    });

    agent_stream_prompt_final_response(&agent, "What's the weather in Paris?").await?;
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_context_compressor_sees_growing_history() -> Result<(), XlaiError> {
    let lens = Arc::new(Mutex::new(Vec::new()));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![ToolCall {
                    id: "t1".to_owned(),
                    tool_name: "lookup_weather".to_owned(),
                    arguments: json!({ "city": "Paris" }),
                }],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("ok"),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let lens_cb = Arc::clone(&lens);
    let mut agent = runtime
        .agent_session()?
        .with_context_compressor(move |msgs, _est| {
            let lens_cb = Arc::clone(&lens_cb);
            async move {
                lock_unpoisoned(&lens_cb).push(msgs.len());
                Ok(msgs)
            }
        });
    agent.register_tool(weather_tool_definition(), |arguments| async move {
        Ok(xlai_core::ToolResult {
            tool_name: "lookup_weather".to_owned(),
            content: format!("weather for {}", arguments["city"].as_str().unwrap_or("?")),
            is_error: false,
            metadata: empty_metadata(),
        })
    });

    agent_stream_prompt_final_response(&agent, "Paris weather?").await?;
    let seen = lock_unpoisoned(&lens);
    assert_eq!(&*seen, &vec![1_usize, 3]);
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_context_compressor_rewritten_messages_reach_model() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![ToolCall {
                    id: "t1".to_owned(),
                    tool_name: "lookup_weather".to_owned(),
                    arguments: json!({ "city": "Paris" }),
                }],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("final"),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let mut agent = runtime
        .agent_session()?
        .with_context_compressor(|mut msgs, _est| async move {
            // Only forward the first user turn to the model; internal history still grows.
            let idx = msgs
                .iter()
                .position(|m| m.role == MessageRole::User)
                .ok_or_else(|| {
                    XlaiError::new(
                        ErrorKind::Provider,
                        "context compressor test: missing user message",
                    )
                })?;
            let u = msgs.remove(idx);
            Ok(vec![u])
        });
    agent.register_tool(weather_tool_definition(), |_arguments| async move {
        Ok(xlai_core::ToolResult {
            tool_name: "lookup_weather".to_owned(),
            content: "tool ok".to_owned(),
            is_error: false,
            metadata: empty_metadata(),
        })
    });

    agent_stream_prompt_final_response(&agent, "Paris?").await?;
    let reqs = lock_unpoisoned(&requests);
    assert_eq!(reqs.len(), 2);
    assert_eq!(reqs[0].messages.len(), 1);
    assert_eq!(reqs[0].messages[0].role, MessageRole::User);
    assert_eq!(reqs[1].messages.len(), 1);
    assert_eq!(reqs[1].messages[0].role, MessageRole::User);
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_loop_injects_continue_when_compressor_removes_all_user_messages()
-> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let rounds = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![ToolCall {
                    id: "t1".to_owned(),
                    tool_name: "lookup_weather".to_owned(),
                    arguments: json!({ "city": "Paris" }),
                }],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("final"),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let rounds_cb = Arc::clone(&rounds);
    let mut agent = runtime
        .agent_session()?
        .with_context_compressor(move |msgs, _est| {
            let round = rounds_cb.fetch_add(1, Ordering::SeqCst);
            async move {
                if round == 0 {
                    Ok(vec![
                        msgs.into_iter()
                            .find(|message| message.role == MessageRole::User)
                            .ok_or_else(|| {
                                XlaiError::new(
                                    ErrorKind::Provider,
                                    "context compressor test: missing user message",
                                )
                            })?,
                    ])
                } else {
                    Ok(msgs
                        .into_iter()
                        .filter(|message| message.role != MessageRole::User)
                        .collect())
                }
            }
        });
    agent.register_tool(weather_tool_definition(), |_arguments| async move {
        Ok(xlai_core::ToolResult {
            tool_name: "lookup_weather".to_owned(),
            content: "tool ok".to_owned(),
            is_error: false,
            metadata: empty_metadata(),
        })
    });

    agent_stream_prompt_final_response(&agent, "Paris?").await?;

    let reqs = lock_unpoisoned(&requests);
    assert_eq!(reqs.len(), 2);
    assert_eq!(reqs[1].messages.len(), 3);
    assert_eq!(reqs[1].messages[0].role, MessageRole::Assistant);
    assert_eq!(reqs[1].messages[1].role, MessageRole::Tool);
    assert_eq!(reqs[1].messages[2].role, MessageRole::User);
    assert_eq!(
        reqs[1].messages[2].content.as_single_text(),
        Some(
            "Continue. If you feel nothing else could be further done, just summarize your work without any tool calling."
        )
    );
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_loop_does_not_inject_continue_when_user_message_still_exists()
-> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![ToolCall {
                    id: "t1".to_owned(),
                    tool_name: "lookup_weather".to_owned(),
                    arguments: json!({ "city": "Paris" }),
                }],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("final"),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let mut agent = runtime.agent_session()?;
    agent.register_tool(weather_tool_definition(), |_arguments| async move {
        Ok(xlai_core::ToolResult {
            tool_name: "lookup_weather".to_owned(),
            content: "tool ok".to_owned(),
            is_error: false,
            metadata: empty_metadata(),
        })
    });

    agent_stream_prompt_final_response(&agent, "Paris?").await?;

    let reqs = lock_unpoisoned(&requests);
    assert_eq!(reqs.len(), 2);
    assert_eq!(reqs[1].messages.len(), 3);
    assert_eq!(reqs[1].messages[0].role, MessageRole::User);
    assert_eq!(reqs[1].messages[1].role, MessageRole::Assistant);
    assert_eq!(reqs[1].messages[2].role, MessageRole::Tool);
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_context_compressor_not_invoked_on_unary_prompt() -> Result<(), XlaiError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("hi"),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let calls_cb = Arc::clone(&calls);
    let agent = runtime
        .agent_session()?
        .with_context_compressor(move |_msgs, _est| {
            let calls_cb = Arc::clone(&calls_cb);
            async move {
                calls_cb.fetch_add(1, Ordering::SeqCst);
                Ok(vec![user_message("x")])
            }
        });

    agent.prompt("hello").await?;
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_context_compressor_error_propagates_from_stream() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(requests.clone(), vec![]));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let agent = runtime
        .agent_session()?
        .with_context_compressor(|_msgs, _est| async {
            Err(XlaiError::new(ErrorKind::Provider, "compressor failed"))
        });

    let mut stream = agent.stream_prompt("x");
    let Some(first) = stream.next().await else {
        return Err(XlaiError::new(
            ErrorKind::Provider,
            "expected stream to yield compressor error",
        ));
    };
    match first {
        Err(e) => {
            assert_eq!(e.kind, ErrorKind::Provider);
            assert!(e.message.contains("compressor failed"));
        }
        Ok(_) => {
            return Err(XlaiError::new(
                ErrorKind::Provider,
                "expected compressor error, got event",
            ));
        }
    }
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_context_compressor_empty_output_errors_stream() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(requests.clone(), vec![]));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let agent = runtime
        .agent_session()?
        .with_context_compressor(|_msgs, _est| async { Ok(Vec::new()) });

    let mut stream = agent.stream_prompt("x");
    let Some(first) = stream.next().await else {
        return Err(XlaiError::new(
            ErrorKind::Provider,
            "expected stream to yield error for empty compressor output",
        ));
    };
    match first {
        Err(e) => assert_eq!(e.kind, ErrorKind::Provider),
        Ok(_) => {
            return Err(XlaiError::new(
                ErrorKind::Provider,
                "expected error for empty compressor output",
            ));
        }
    }
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_context_compressor_skipped_when_agent_loop_disabled() -> Result<(), XlaiError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let chunks = vec![xlai_core::ChatChunk::Finished(ChatResponse {
        message: assistant_message("one shot"),
        tool_calls: Vec::new(),
        usage: None,
        finish_reason: FinishReason::Completed,
        metadata: empty_metadata(),
    })];
    let model = Arc::new(StreamingChatModel::new(vec![chunks]));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let calls_cb = Arc::clone(&calls);
    let agent = runtime
        .agent_session()?
        .with_agent_loop_enabled(false)
        .with_context_compressor(move |_msgs, _est| {
            let calls_cb = Arc::clone(&calls_cb);
            async move {
                calls_cb.fetch_add(1, Ordering::SeqCst);
                Ok(vec![user_message("should not run")])
            }
        });

    let _ = agent_stream_prompt_final_response(&agent, "x").await?;
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_system_reminder_unary_inserts_before_user() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("ok"),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let agent = runtime
        .agent_session()?
        .with_system_reminder(|_msgs| async { Ok("remember: be brief".to_owned()) });

    agent.prompt("hello").await?;

    let reqs = lock_unpoisoned(&requests);
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].messages.len(), 2);
    assert_eq!(reqs[0].messages[0].role, MessageRole::System);
    assert!(
        reqs[0].messages[0]
            .metadata
            .get("xlai_system_reminder")
            .and_then(|v| v.as_bool())
            == Some(true),
        "expected synthetic reminder metadata marker"
    );
    assert!(
        reqs[0].messages[0]
            .content
            .as_single_text()
            .is_some_and(|t| t.contains("remember: be brief")),
        "expected reminder to include user hook text"
    );
    assert_eq!(reqs[0].messages[1].role, MessageRole::User);
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_system_reminder_strips_inbound_internal_rows_from_history() -> Result<(), XlaiError>
{
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("ok"),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let mut agent = runtime.agent_session()?;
    agent.register_system_reminder(|msgs| async move {
        assert_eq!(
            msgs.len(),
            1,
            "callback must not see internal reminder rows"
        );
        assert!(
            msgs.iter().all(|m| {
                m.metadata
                    .get("xlai_system_reminder")
                    .and_then(|v| v.as_bool())
                    != Some(true)
            }),
            "callback transcript must exclude internal reminders"
        );
        Ok("new-reminder".to_owned())
    });

    agent
        .execute(vec![
            user_message("hi"),
            leaked_internal_system_reminder("leaked-secret-body"),
        ])
        .await?;

    let reqs = lock_unpoisoned(&requests);
    assert_eq!(reqs[0].messages.len(), 2);
    let Some(reminder_text) = reqs[0].messages[0].content.as_single_text() else {
        return Err(XlaiError::new(
            ErrorKind::Provider,
            "expected plain-text system reminder",
        ));
    };
    assert!(
        reminder_text.contains("new-reminder"),
        "expected fresh composed reminder"
    );
    assert!(
        !reminder_text.contains("leaked-secret-body"),
        "leaked internal reminder must not be forwarded"
    );
    assert_eq!(reqs[0].messages[1].role, MessageRole::User);
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_system_reminder_inserts_before_non_user_tail() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("tail"),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let agent = runtime
        .agent_session()?
        .with_system_reminder(|_| async { Ok("reminder-body".to_owned()) });

    agent
        .execute(vec![user_message("u"), assistant_message("a")])
        .await?;

    let reqs = lock_unpoisoned(&requests);
    assert_eq!(reqs[0].messages.len(), 3);
    assert_eq!(reqs[0].messages[0].role, MessageRole::User);
    assert_eq!(reqs[0].messages[1].role, MessageRole::System);
    assert!(
        reqs[0].messages[1]
            .content
            .as_single_text()
            .is_some_and(|t| t.contains("reminder-body")),
        "expected reminder before non-user tail"
    );
    assert_eq!(reqs[0].messages[2].role, MessageRole::Assistant);
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_system_reminder_does_not_split_assistant_tool_block() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("ok"),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let agent = runtime
        .agent_session()?
        .with_system_reminder(|_| async { Ok("reminder-body".to_owned()) });

    agent
        .execute(vec![
            user_message("u"),
            assistant_message("").with_assistant_tool_calls(&[ToolCall {
                id: "call_1".to_owned(),
                tool_name: "skill".to_owned(),
                arguments: json!({ "skill_id": "review.code" }),
            }]),
            {
                let mut metadata = empty_metadata();
                metadata.insert("skill_id".to_owned(), json!("review.code"));
                metadata.insert("skill_name".to_owned(), json!("Code Review"));
                ChatMessage {
                    role: MessageRole::Tool,
                    content: ChatContent::text("resolved"),
                    tool_name: Some("skill".to_owned()),
                    tool_call_id: Some("call_1".to_owned()),
                    metadata,
                }
            },
        ])
        .await?;

    let reqs = lock_unpoisoned(&requests);
    assert_eq!(reqs[0].messages.len(), 4);
    assert_eq!(reqs[0].messages[0].role, MessageRole::User);
    assert_eq!(reqs[0].messages[1].role, MessageRole::System);
    assert_eq!(reqs[0].messages[2].role, MessageRole::Assistant);
    assert_eq!(reqs[0].messages[3].role, MessageRole::Tool);
    assert!(
        reqs[0].messages[1]
            .content
            .as_single_text()
            .is_some_and(|t| t.contains("reminder-body")),
        "expected reminder before assistant+tool tail block"
    );
    assert!(
        reqs[0].messages[2].assistant_tool_calls().is_some(),
        "expected assistant tool calls metadata to be preserved"
    );
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_system_reminder_skips_when_nothing_to_add() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("ok"),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let agent = runtime
        .agent_session()?
        .with_system_reminder(|_| async { Ok("  \n\t".to_owned()) });

    agent.prompt("hi").await?;

    let reqs = lock_unpoisoned(&requests);
    assert_eq!(reqs[0].messages.len(), 1);
    assert_eq!(reqs[0].messages[0].role, MessageRole::User);
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_system_reminder_stream_runs_once_per_loop_round() -> Result<(), XlaiError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![ToolCall {
                    id: "t1".to_owned(),
                    tool_name: "lookup_weather".to_owned(),
                    arguments: json!({ "city": "Paris" }),
                }],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("done"),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let calls_cb = Arc::clone(&calls);
    let mut agent = runtime.agent_session()?.with_system_reminder(move |_msgs| {
        let calls_cb = Arc::clone(&calls_cb);
        async move {
            calls_cb.fetch_add(1, Ordering::SeqCst);
            Ok("ping".to_owned())
        }
    });
    agent.register_tool(weather_tool_definition(), |arguments| async move {
        Ok(xlai_core::ToolResult {
            tool_name: "lookup_weather".to_owned(),
            content: format!("weather for {}", arguments["city"].as_str().unwrap_or("?")),
            is_error: false,
            metadata: empty_metadata(),
        })
    });

    agent_stream_prompt_final_response(&agent, "Paris?").await?;
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    let reqs = lock_unpoisoned(&requests);
    assert_eq!(reqs.len(), 2);
    for r in reqs.iter() {
        assert!(
            r.messages.iter().any(|m| {
                m.role == MessageRole::System
                    && m.metadata
                        .get("xlai_system_reminder")
                        .and_then(|v| v.as_bool())
                        == Some(true)
            }),
            "expected a system reminder on each outgoing request"
        );
    }
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_system_reminder_includes_available_skills_from_store() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("ok"),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new()
        .with_chat_model(model)
        .with_skill_store(seed_markdown_skill_store().await?)
        .build()?;
    let agent = runtime.agent_session()?;

    agent.prompt("use skills").await?;

    let reqs = lock_unpoisoned(&requests);
    assert_eq!(reqs[0].messages.len(), 2);
    assert_eq!(reqs[0].messages[0].role, MessageRole::System);
    let Some(text) = reqs[0].messages[0].content.as_single_text() else {
        return Err(XlaiError::new(
            ErrorKind::Provider,
            "expected plain-text available-skills reminder",
        ));
    };
    assert!(
        text.contains("review.code"),
        "expected available-skills reminder, got: {text:?}"
    );
    Ok(())
}

fn skill_tool_message(skill_id: &str) -> ChatMessage {
    let mut metadata = empty_metadata();
    metadata.insert("skill_id".to_owned(), json!(skill_id));
    metadata.insert("skill_name".to_owned(), json!("Code Review"));
    ChatMessage {
        role: MessageRole::Tool,
        content: ChatContent::text("resolved"),
        tool_name: Some("skill".to_owned()),
        tool_call_id: Some("c1".to_owned()),
        metadata,
    }
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_system_reminder_includes_invoked_skills_section() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("ok"),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new()
        .with_chat_model(model)
        .with_skill_store(seed_markdown_skill_store().await?)
        .build()?;
    let agent = runtime.agent_session()?;

    agent
        .execute(vec![
            user_message("prior"),
            skill_tool_message("review.code"),
        ])
        .await?;

    let reqs = lock_unpoisoned(&requests);
    let reminder_idx = reqs[0].messages.len() - 2;
    let Some(text) = reqs[0].messages[reminder_idx].content.as_single_text() else {
        return Err(XlaiError::new(
            ErrorKind::Provider,
            "expected plain-text invoked-skills reminder",
        ));
    };
    assert!(
        text.contains("were invoked"),
        "expected invoked-skills section, got: {text:?}"
    );
    assert!(text.contains("review.code"));
    Ok(())
}
