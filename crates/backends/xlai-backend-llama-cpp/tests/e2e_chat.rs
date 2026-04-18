use std::path::PathBuf;

use xlai_backend_llama_cpp::LlamaCppConfig;
use xlai_core::MessageRole;
use xlai_core::{ErrorKind, XlaiError};
use xlai_runtime::RuntimeBuilder;

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
#[ignore = "requires a local GGUF fixture or LLAMA_CPP_MODEL override"]
async fn llama_cpp_chat_smoke_test() -> Result<(), XlaiError> {
    let model_path = resolve_model_path()?;

    let runtime = RuntimeBuilder::new()
        .with_chat_backend(LlamaCppConfig::new(model_path).with_max_output_tokens(64))
        .build()?;

    let chat = runtime
        .chat_session()
        .with_system_prompt("Reply concisely and truthfully.");

    let response = chat
        .prompt("Reply with a short greeting for the xlai llama.cpp smoke test.")
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

fn resolve_model_path() -> Result<String, XlaiError> {
    if let Ok(path) = std::env::var("LLAMA_CPP_MODEL") {
        return Ok(path);
    }

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/llama.cpp/Qwen3.5-0.8B-Q4_0.gguf");
    if fixture.exists() {
        return Ok(fixture.to_string_lossy().into_owned());
    }

    Err(XlaiError::new(
        ErrorKind::Configuration,
        format!(
            "set LLAMA_CPP_MODEL or place the test model at {}",
            fixture.display()
        ),
    ))
}
