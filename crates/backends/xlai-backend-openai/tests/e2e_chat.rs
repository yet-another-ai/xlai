use xlai_backend_openai::OpenAiConfig;
use xlai_core::MessageRole;
use xlai_core::{ErrorKind, XlaiError};
use xlai_runtime::RuntimeBuilder;

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY in the protected e2e environment"]
async fn openai_chat_smoke_test() -> Result<(), XlaiError> {
    let _ = dotenvy::dotenv();

    let api_key = require_env("OPENAI_API_KEY")?;
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_owned());

    let runtime = RuntimeBuilder::new()
        .with_chat_backend(OpenAiConfig::new(base_url, api_key, model))
        .build()?;

    let chat = runtime
        .chat_session()
        .with_system_prompt("Reply concisely and truthfully.");

    let response = chat
        .prompt("Reply with a short greeting for the xlai e2e smoke test.")
        .await?;

    assert_eq!(response.message.role, MessageRole::Assistant);
    assert!(
        !response
            .message
            .content
            .text_parts_concatenated()
            .trim()
            .is_empty(),
        "assistant response should not be empty",
    );
    assert!(
        response.tool_calls.is_empty(),
        "smoke test should not require tool calls",
    );

    Ok(())
}

fn require_env(name: &str) -> Result<String, XlaiError> {
    std::env::var(name).map_err(|error| {
        XlaiError::new(
            ErrorKind::Configuration,
            format!("{name} must be set for e2e tests: {error}"),
        )
    })
}
