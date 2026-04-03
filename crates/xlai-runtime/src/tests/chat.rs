//! Runtime tests: chat session.
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use xlai_core::{
    ChatContent, ChatMessage, ChatResponse, ContentPart, FinishReason, MediaSource, MessageRole,
    StructuredOutput, StructuredOutputFormat, ToolCall, XlaiError,
};

use super::common::*;
use crate::{EmbeddedPromptStore, PromptContext, RuntimeBuilder};

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn chat_executes_registered_tools_across_round_trips() -> Result<(), XlaiError> {
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

    let mut chat = runtime.chat_session();
    chat.register_tool(weather_tool_definition(), |arguments| async move {
        Ok(xlai_core::ToolResult {
            tool_name: "ignored_by_runtime".to_owned(),
            content: format!(
                "weather for {}: sunny",
                arguments["city"].as_str().unwrap_or("unknown")
            ),
            is_error: false,
            metadata: empty_metadata(),
        })
    });

    let response = chat.prompt("What's the weather in Paris?").await?;

    assert_eq!(
        response.message.content.as_single_text(),
        Some("Paris is sunny.")
    );

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].available_tools.len(), 1);
    assert_eq!(requests[1].messages.len(), 3);
    assert_eq!(requests[1].messages[2].role, MessageRole::Tool);
    assert_eq!(
        requests[1].messages[2].tool_name.as_deref(),
        Some("lookup_weather")
    );
    assert_eq!(
        requests[1].messages[2].tool_call_id.as_deref(),
        Some("tool_1")
    );
    assert_eq!(
        requests[1].messages[2].content.as_single_text(),
        Some("weather for Paris: sunny")
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn chat_prompt_parts_preserves_multimodal_user_message_in_request() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("noted."),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let chat = runtime.chat_session();
    chat.prompt_parts(vec![
        ContentPart::Text {
            text: "What is in this image?".to_owned(),
        },
        ContentPart::Image {
            source: MediaSource::Url {
                url: "https://example.com/picture.png".to_owned(),
            },
            mime_type: None,
            detail: None,
        },
    ])
    .await?;

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].messages.len(), 1);
    assert_eq!(requests[0].messages[0].role, MessageRole::User);
    assert_eq!(requests[0].messages[0].content.parts.len(), 2);

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn chat_prompt_parts_preserves_audio_user_message_in_request() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("got audio."),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let chat = runtime.chat_session();
    chat.prompt_parts(vec![ContentPart::Audio {
        source: MediaSource::InlineData {
            mime_type: "audio/wav".to_owned(),
            data_base64: "UklGRg==".to_owned(),
        },
        mime_type: Some("audio/wav".to_owned()),
    }])
    .await?;

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].messages[0].content.parts.len(), 1);
    assert!(matches!(
        requests[0].messages[0].content.parts[0],
        ContentPart::Audio { .. }
    ));

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn chat_execute_preserves_structured_history_metadata() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("Reminder acknowledged."),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let chat = runtime.chat_session();

    let mut metadata = empty_metadata();
    metadata.insert(
        "reminder".to_owned(),
        json!({
            "kind": "system_reminder",
            "editable": true,
            "scope": {
                "session": "future"
            }
        }),
    );

    let response = chat
        .execute(vec![ChatMessage {
            role: MessageRole::System,
            content: ChatContent::text("Remind the assistant to stay concise."),
            tool_name: None,
            tool_call_id: None,
            metadata: metadata.clone(),
        }])
        .await?;

    assert_eq!(
        response.message.content.as_single_text(),
        Some("Reminder acknowledged.")
    );

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].messages[0].metadata.get("reminder"),
        Some(&json!({
            "kind": "system_reminder",
            "editable": true,
            "scope": {
                "session": "future"
            }
        }))
    );

    if requests[0].messages[0]
        .metadata
        .get("reminder")
        .and_then(Value::as_object)
        .and_then(|reminder| reminder.get("kind"))
        .and_then(Value::as_str)
        != Some("system_reminder")
    {
        return Err(XlaiError::new(
            xlai_core::ErrorKind::Validation,
            "expected structured reminder metadata to be preserved in chat history",
        ));
    }

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn chat_can_load_embedded_system_prompt_assets() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("Prompt loaded."),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let chat = runtime
        .chat_session()
        .with_system_prompt_asset("system/tool-description-skill.md")?;

    let response = chat.prompt("Say something brief.").await?;
    assert_eq!(
        response.message.content.as_single_text(),
        Some("Prompt loaded.")
    );

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 1);
    let system_prompt = requests[0].system_prompt.as_deref();
    assert_eq!(
        system_prompt,
        Some(EmbeddedPromptStore::load("system/tool-description-skill.md")?.as_str())
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn chat_can_render_embedded_system_prompt_templates() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("Rendered prompt loaded."),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let mut context = PromptContext::new();
    context.insert("skill_tag_name", "tool_skill");
    let chat = runtime
        .chat_session()
        .with_system_prompt_template("system/tool-description-skill.md", &context)?;

    let response = chat.prompt("Say something brief.").await?;
    assert_eq!(
        response.message.content.as_single_text(),
        Some("Rendered prompt loaded.")
    );

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0]
            .system_prompt
            .as_deref()
            .is_some_and(|prompt| prompt.contains("tool_skill"))
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn chat_session_propagates_structured_output_requests() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingChatModel::new(
        requests.clone(),
        vec![ChatResponse {
            message: assistant_message("{\"name\":\"Ada\"}"),
            tool_calls: Vec::new(),
            usage: None,
            finish_reason: FinishReason::Completed,
            metadata: empty_metadata(),
        }],
    ));

    let runtime = RuntimeBuilder::new().with_chat_model(model).build()?;
    let chat = runtime.chat_session().with_structured_output(
        StructuredOutput::lark_grammar("start: NAME\nNAME: /[A-Z][a-z]+/")
            .with_name("person")
            .with_description("Capitalized name"),
    );

    let _response = chat.prompt("Return a name.").await?;

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 1);
    let structured_output = requests[0].structured_output.as_ref();
    assert!(
        structured_output.is_some(),
        "structured output should be propagated"
    );
    let Some(structured_output) = structured_output else {
        return Ok(());
    };
    assert_eq!(structured_output.name.as_deref(), Some("person"));
    assert_eq!(
        structured_output.description.as_deref(),
        Some("Capitalized name")
    );
    assert!(matches!(
        &structured_output.format,
        StructuredOutputFormat::LarkGrammar { grammar }
            if grammar == "start: NAME\nNAME: /[A-Z][a-z]+/"
    ));

    Ok(())
}
