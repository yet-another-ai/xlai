//! Runtime tests: agent and MCP.
use std::sync::{Arc, Mutex};

use serde_json::json;
use xlai_core::{ChatResponse, FinishReason, ToolCall, XlaiError};

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
    assert_eq!(requests[1].messages.len(), 3);
    assert_eq!(requests[1].messages[2].tool_name.as_deref(), Some("skill"));
    assert!(
        requests[1].messages[2]
            .content
            .text_parts_concatenated()
            .contains("Prioritize bugs, regressions, and missing tests."),
        "expected the skill tool result to include the resolved prompt fragment"
    );
    assert!(
        requests[1].messages[2]
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
