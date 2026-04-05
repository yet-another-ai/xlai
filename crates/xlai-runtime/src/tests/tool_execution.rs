//! Runtime tests: tool concurrency.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::json;
use tokio::time::{Duration, sleep};
use xlai_core::{ChatResponse, FinishReason, ToolCall, XlaiError};

use super::common::{agent_stream_prompt_final_response, *};
use crate::RuntimeBuilder;

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_executes_multiple_tool_calls_concurrently_by_default() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![
                    ToolCall {
                        id: "tool_1".to_owned(),
                        tool_name: "lookup_weather".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    },
                    ToolCall {
                        id: "tool_2".to_owned(),
                        tool_name: "lookup_time".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    },
                ],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("Paris is sunny and 9am."),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let active_calls = Arc::new(AtomicUsize::new(0));
    let max_active_calls = Arc::new(AtomicUsize::new(0));
    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;

    let mut agent = runtime.agent_session()?;
    {
        let active_calls = active_calls.clone();
        let max_active_calls = max_active_calls.clone();
        agent.register_tool(weather_tool_definition(), move |_| {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            async move {
                let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_calls.fetch_max(current, Ordering::SeqCst);
                sleep(Duration::from_millis(25)).await;
                active_calls.fetch_sub(1, Ordering::SeqCst);
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_weather".to_owned(),
                    content: "weather for Paris: sunny".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            }
        });
    }
    {
        let active_calls = active_calls.clone();
        let max_active_calls = max_active_calls.clone();
        agent.register_tool(time_tool_definition(), move |_| {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            async move {
                let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_calls.fetch_max(current, Ordering::SeqCst);
                sleep(Duration::from_millis(25)).await;
                active_calls.fetch_sub(1, Ordering::SeqCst);
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_time".to_owned(),
                    content: "time for Paris: 9am".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            }
        });
    }

    let response =
        agent_stream_prompt_final_response(&agent, "What's the weather and time in Paris?").await?;
    assert_eq!(
        response.message.content.as_single_text(),
        Some("Paris is sunny and 9am.")
    );

    assert!(
        max_active_calls.load(Ordering::SeqCst) >= 2,
        "default mode should overlap tool executions",
    );

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].messages.len(), 4);
    assert_eq!(
        requests[1].messages[2].tool_name.as_deref(),
        Some("lookup_weather")
    );
    assert_eq!(
        requests[1].messages[3].tool_name.as_deref(),
        Some("lookup_time")
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_can_execute_multiple_tool_calls_concurrently() -> Result<(), XlaiError> {
    let model = Arc::new(RecordingChatModel::new(
        Arc::new(Mutex::new(Vec::new())),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![
                    ToolCall {
                        id: "tool_1".to_owned(),
                        tool_name: "lookup_weather".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    },
                    ToolCall {
                        id: "tool_2".to_owned(),
                        tool_name: "lookup_time".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    },
                ],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("Paris is sunny and 9am."),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let active_calls = Arc::new(AtomicUsize::new(0));
    let max_active_calls = Arc::new(AtomicUsize::new(0));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;

    let mut agent = runtime.agent_session()?;

    {
        let active_calls = active_calls.clone();
        let max_active_calls = max_active_calls.clone();
        agent.register_tool(weather_tool_definition(), move |_| {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            async move {
                let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_calls.fetch_max(current, Ordering::SeqCst);
                sleep(Duration::from_millis(25)).await;
                active_calls.fetch_sub(1, Ordering::SeqCst);
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_weather".to_owned(),
                    content: "weather for Paris: sunny".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            }
        });
    }
    {
        let active_calls = active_calls.clone();
        let max_active_calls = max_active_calls.clone();
        agent.register_tool(time_tool_definition(), move |_| {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            async move {
                let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_calls.fetch_max(current, Ordering::SeqCst);
                sleep(Duration::from_millis(25)).await;
                active_calls.fetch_sub(1, Ordering::SeqCst);
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_time".to_owned(),
                    content: "time for Paris: 9am".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            }
        });
    }

    let response =
        agent_stream_prompt_final_response(&agent, "What's the weather and time in Paris?").await?;
    assert_eq!(
        response.message.content.as_single_text(),
        Some("Paris is sunny and 9am.")
    );
    assert!(
        max_active_calls.load(Ordering::SeqCst) >= 2,
        "concurrent mode should overlap tool executions",
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_runs_tool_batch_sequentially_when_any_tool_is_sequential() -> Result<(), XlaiError> {
    let execution_order = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        Arc::new(Mutex::new(Vec::new())),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![
                    ToolCall {
                        id: "tool_1".to_owned(),
                        tool_name: "lookup_weather".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    },
                    ToolCall {
                        id: "tool_2".to_owned(),
                        tool_name: "lookup_time".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    },
                ],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("Paris is sunny and 9am."),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;

    let mut agent = runtime.agent_session()?;
    {
        let execution_order = execution_order.clone();
        agent.register_tool(weather_tool_definition_sequential(), move |arguments| {
            let execution_order = execution_order.clone();
            async move {
                lock_unpoisoned(&execution_order).push(format!(
                    "weather:{}",
                    arguments["city"].as_str().unwrap_or("unknown")
                ));
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_weather".to_owned(),
                    content: "weather for Paris: sunny".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            }
        });
    }
    {
        let execution_order = execution_order.clone();
        agent.register_tool(time_tool_definition(), move |arguments| {
            let execution_order = execution_order.clone();
            async move {
                lock_unpoisoned(&execution_order).push(format!(
                    "time:{}",
                    arguments["city"].as_str().unwrap_or("unknown")
                ));
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_time".to_owned(),
                    content: "time for Paris: 9am".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            }
        });
    }

    let response =
        agent_stream_prompt_final_response(&agent, "What's the weather and time in Paris?").await?;
    assert_eq!(
        response.message.content.as_single_text(),
        Some("Paris is sunny and 9am.")
    );

    let execution_order = lock_unpoisoned(&execution_order);
    assert_eq!(
        *execution_order,
        vec!["weather:Paris".to_owned(), "time:Paris".to_owned()]
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_runs_mixed_tool_batch_sequentially_in_model_order() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![
                    ToolCall {
                        id: "tool_1".to_owned(),
                        tool_name: "lookup_weather".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    },
                    ToolCall {
                        id: "tool_2".to_owned(),
                        tool_name: "lookup_time".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    },
                    ToolCall {
                        id: "tool_3".to_owned(),
                        tool_name: "lookup_calendar".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    },
                ],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("Paris schedule assembled."),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let active_calls = Arc::new(AtomicUsize::new(0));
    let max_active_calls = Arc::new(AtomicUsize::new(0));
    let execution_graph = Arc::new(Mutex::new(Vec::new()));
    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;

    let mut agent = runtime.agent_session()?;
    {
        let active_calls = active_calls.clone();
        let max_active_calls = max_active_calls.clone();
        let execution_graph = execution_graph.clone();
        agent.register_tool(weather_tool_definition(), move |_| {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            let execution_graph = execution_graph.clone();
            async move {
                lock_unpoisoned(&execution_graph).push("weather:start".to_owned());
                let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_calls.fetch_max(current, Ordering::SeqCst);
                sleep(Duration::from_millis(20)).await;
                active_calls.fetch_sub(1, Ordering::SeqCst);
                lock_unpoisoned(&execution_graph).push("weather:end".to_owned());
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_weather".to_owned(),
                    content: "weather for Paris: sunny".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            }
        });
    }
    {
        let active_calls = active_calls.clone();
        let max_active_calls = max_active_calls.clone();
        let execution_graph = execution_graph.clone();
        agent.register_tool(time_tool_definition_sequential(), move |_| {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            let execution_graph = execution_graph.clone();
            async move {
                lock_unpoisoned(&execution_graph).push("time:start".to_owned());
                let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_calls.fetch_max(current, Ordering::SeqCst);
                sleep(Duration::from_millis(20)).await;
                active_calls.fetch_sub(1, Ordering::SeqCst);
                lock_unpoisoned(&execution_graph).push("time:end".to_owned());
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_time".to_owned(),
                    content: "time for Paris: 9am".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            }
        });
    }
    {
        let active_calls = active_calls.clone();
        let max_active_calls = max_active_calls.clone();
        let execution_graph = execution_graph.clone();
        agent.register_tool(calendar_tool_definition(), move |_| {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            let execution_graph = execution_graph.clone();
            async move {
                lock_unpoisoned(&execution_graph).push("calendar:start".to_owned());
                let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_calls.fetch_max(current, Ordering::SeqCst);
                sleep(Duration::from_millis(20)).await;
                active_calls.fetch_sub(1, Ordering::SeqCst);
                lock_unpoisoned(&execution_graph).push("calendar:end".to_owned());
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_calendar".to_owned(),
                    content: "calendar for Paris: free after 10am".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            }
        });
    }

    let response =
        agent_stream_prompt_final_response(&agent, "Build my Paris schedule.").await?;
    assert_eq!(
        response.message.content.as_single_text(),
        Some("Paris schedule assembled.")
    );
    assert_eq!(
        max_active_calls.load(Ordering::SeqCst),
        1,
        "a sequential tool should serialize the whole batch",
    );
    assert_eq!(
        *lock_unpoisoned(&execution_graph),
        vec![
            "weather:start".to_owned(),
            "weather:end".to_owned(),
            "time:start".to_owned(),
            "time:end".to_owned(),
            "calendar:start".to_owned(),
            "calendar:end".to_owned(),
        ]
    );

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[1]
            .messages
            .iter()
            .filter_map(|message| message.tool_name.clone())
            .collect::<Vec<_>>(),
        vec![
            "lookup_weather".to_owned(),
            "lookup_time".to_owned(),
            "lookup_calendar".to_owned(),
        ]
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn agent_runs_mixed_batch_sequentially_when_multiple_tools_are_sequential()
-> Result<(), XlaiError> {
    let model = Arc::new(RecordingChatModel::new(
        Arc::new(Mutex::new(Vec::new())),
        vec![
            ChatResponse {
                message: assistant_message(""),
                tool_calls: vec![
                    ToolCall {
                        id: "tool_1".to_owned(),
                        tool_name: "lookup_weather".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    },
                    ToolCall {
                        id: "tool_2".to_owned(),
                        tool_name: "lookup_time".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    },
                    ToolCall {
                        id: "tool_3".to_owned(),
                        tool_name: "lookup_calendar".to_owned(),
                        arguments: json!({ "city": "Paris" }),
                    },
                ],
                usage: None,
                finish_reason: FinishReason::ToolCalls,
                metadata: empty_metadata(),
            },
            ChatResponse {
                message: assistant_message("Paris plan completed."),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            },
        ],
    ));

    let active_calls = Arc::new(AtomicUsize::new(0));
    let max_active_calls = Arc::new(AtomicUsize::new(0));
    let execution_order = Arc::new(Mutex::new(Vec::new()));
    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;

    let mut agent = runtime.agent_session()?;
    {
        let active_calls = active_calls.clone();
        let max_active_calls = max_active_calls.clone();
        let execution_order = execution_order.clone();
        agent.register_tool(weather_tool_definition_sequential(), move |_| {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            let execution_order = execution_order.clone();
            async move {
                let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_calls.fetch_max(current, Ordering::SeqCst);
                lock_unpoisoned(&execution_order).push("weather".to_owned());
                sleep(Duration::from_millis(15)).await;
                active_calls.fetch_sub(1, Ordering::SeqCst);
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_weather".to_owned(),
                    content: "weather for Paris: sunny".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            }
        });
    }
    {
        let active_calls = active_calls.clone();
        let max_active_calls = max_active_calls.clone();
        let execution_order = execution_order.clone();
        agent.register_tool(time_tool_definition(), move |_| {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            let execution_order = execution_order.clone();
            async move {
                let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_calls.fetch_max(current, Ordering::SeqCst);
                lock_unpoisoned(&execution_order).push("time".to_owned());
                sleep(Duration::from_millis(15)).await;
                active_calls.fetch_sub(1, Ordering::SeqCst);
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_time".to_owned(),
                    content: "time for Paris: 9am".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            }
        });
    }
    {
        let active_calls = active_calls.clone();
        let max_active_calls = max_active_calls.clone();
        let execution_order = execution_order.clone();
        agent.register_tool(calendar_tool_definition_sequential(), move |_| {
            let active_calls = active_calls.clone();
            let max_active_calls = max_active_calls.clone();
            let execution_order = execution_order.clone();
            async move {
                let current = active_calls.fetch_add(1, Ordering::SeqCst) + 1;
                max_active_calls.fetch_max(current, Ordering::SeqCst);
                lock_unpoisoned(&execution_order).push("calendar".to_owned());
                sleep(Duration::from_millis(15)).await;
                active_calls.fetch_sub(1, Ordering::SeqCst);
                Ok(xlai_core::ToolResult {
                    tool_name: "lookup_calendar".to_owned(),
                    content: "calendar for Paris: free after 10am".to_owned(),
                    is_error: false,
                    metadata: empty_metadata(),
                })
            }
        });
    }

    let response = agent_stream_prompt_final_response(&agent, "Plan my Paris day.").await?;
    assert_eq!(
        response.message.content.as_single_text(),
        Some("Paris plan completed.")
    );
    assert_eq!(
        max_active_calls.load(Ordering::SeqCst),
        1,
        "mixed batches with multiple sequential tools must still avoid overlap",
    );
    assert_eq!(
        *lock_unpoisoned(&execution_order),
        vec![
            "weather".to_owned(),
            "time".to_owned(),
            "calendar".to_owned(),
        ]
    );

    Ok(())
}
