use futures_util::StreamExt;
use xlai_backend_gemini::GeminiConfig;
use xlai_core::{ChatBackend, ChatContent, ChatModel, ChatRequest, MessageRole, XlaiError};

#[tokio::test]
async fn gemini_chat_smoke_test() -> Result<(), XlaiError> {
    let api_key = match std::env::var("GEMINI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            println!("Skipping gemini_chat_smoke_test because GEMINI_API_KEY is not set");
            return Ok(());
        }
    };

    let base_url = std::env::var("GEMINI_BASE_URL")
        .unwrap_or_else(|_| "https://generativelanguage.googleapis.com/v1beta".to_owned());
    let model_name =
        std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_owned());

    let config = GeminiConfig::new(base_url, api_key, model_name);
    let model = config.into_chat_model();

    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![xlai_core::ChatMessage {
            role: MessageRole::User,
            content: ChatContent::text("Say exactly 'hello world' and nothing else."),
            tool_name: None,
            tool_call_id: None,
            metadata: Default::default(),
        }],
        available_tools: Vec::new(),
        structured_output: None,
        metadata: Default::default(),
        temperature: None,
        max_output_tokens: None,
        reasoning_effort: None,
        retry_policy: None,
    };

    let response = model.generate(request).await?;
    assert!(
        response
            .message
            .content
            .text_parts_concatenated()
            .to_lowercase()
            .contains("hello world"),
        "Response should contain 'hello world', got: {:?}",
        response.message.content.text_parts_concatenated()
    );

    Ok(())
}

#[tokio::test]
async fn gemini_chat_stream_smoke_test() -> Result<(), XlaiError> {
    let api_key = match std::env::var("GEMINI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            println!("Skipping gemini_chat_stream_smoke_test because GEMINI_API_KEY is not set");
            return Ok(());
        }
    };

    let base_url = std::env::var("GEMINI_BASE_URL")
        .unwrap_or_else(|_| "https://generativelanguage.googleapis.com/v1beta".to_owned());
    let model_name =
        std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_owned());

    let config = GeminiConfig::new(base_url, api_key, model_name);
    let model = config.into_chat_model();

    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![xlai_core::ChatMessage {
            role: MessageRole::User,
            content: ChatContent::text("Say exactly 'hello stream' and nothing else."),
            tool_name: None,
            tool_call_id: None,
            metadata: Default::default(),
        }],
        available_tools: Vec::new(),
        structured_output: None,
        metadata: Default::default(),
        temperature: None,
        max_output_tokens: None,
        reasoning_effort: None,
        retry_policy: None,
    };

    let mut stream = model.generate_stream(request);
    let mut full_text = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let xlai_core::ChatChunk::ContentDelta(delta) = chunk {
            full_text.push_str(&delta.delta);
        }
    }

    assert!(
        full_text.to_lowercase().contains("hello stream"),
        "Streamed response should contain 'hello stream', got: {:?}",
        full_text
    );

    Ok(())
}
