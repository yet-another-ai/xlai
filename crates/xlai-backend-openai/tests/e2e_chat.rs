use xlai_core::MessageRole;
use xlai_runtime::{OpenAiConfig, RuntimeBuilder};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY in the protected e2e environment"]
async fn openai_chat_smoke_test() {
    let _ = dotenvy::dotenv();

    let api_key = require_env("OPENAI_API_KEY");
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_owned());

    let runtime = RuntimeBuilder::new()
        .with_openai_chat(OpenAiConfig::new(base_url, api_key, model))
        .build()
        .expect("runtime should build");

    let chat = runtime
        .chat_session()
        .with_system_prompt("Reply concisely and truthfully.");

    let response = chat
        .prompt("Reply with a short greeting for the xlai e2e smoke test.")
        .await
        .expect("chat request should succeed");

    assert_eq!(response.message.role, MessageRole::Assistant);
    assert!(
        !response.message.content.trim().is_empty(),
        "assistant response should not be empty",
    );
    assert!(
        response.tool_calls.is_empty(),
        "smoke test should not require tool calls",
    );
}

fn require_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set for e2e tests"))
}
