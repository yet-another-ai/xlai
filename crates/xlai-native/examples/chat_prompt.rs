//! Minimal native chat example. Run with a real key:
//! `OPENAI_API_KEY=... cargo run -p xlai-native --example chat_prompt`

use xlai_native::{OpenAiConfig, RuntimeBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let runtime = RuntimeBuilder::new()
        .with_chat_backend(OpenAiConfig::new(
            std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_owned()),
            api_key,
            std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_owned()),
        ))
        .build()?;

    let response = runtime
        .chat_session()
        .with_system_prompt("Reply in at most five words.")
        .prompt("Say hello.")
        .await?;

    println!(
        "{}",
        response
            .message
            .content
            .as_single_text()
            .unwrap_or_default()
    );
    Ok(())
}
