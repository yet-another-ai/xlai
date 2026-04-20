use xlai_backend_openai::OpenAiConfig;
use xlai_core::{EmbeddingRequest, ErrorKind, RuntimeCapability, XlaiError};
use xlai_runtime::RuntimeBuilder;

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL in the protected e2e environment"]
async fn openai_embeddings_smoke_test() -> Result<(), XlaiError> {
    let _ = dotenvy::dotenv();

    let runtime = build_embedding_runtime()?;
    let response = runtime
        .embed(EmbeddingRequest {
            model: None,
            inputs: vec!["A short smoke-test sentence for embeddings.".to_owned()],
            dimensions: None,
            metadata: Default::default(),
        })
        .await?;

    assert_eq!(response.vectors.len(), 1);
    assert!(
        !response.vectors[0].is_empty(),
        "embedding response should contain a non-empty vector",
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL in the protected e2e environment"]
async fn openai_embeddings_runtime_reports_capability() -> Result<(), XlaiError> {
    let _ = dotenvy::dotenv();

    let runtime = build_embedding_runtime()?;
    assert!(runtime.has_capability(RuntimeCapability::Embeddings));

    Ok(())
}

fn build_embedding_runtime() -> Result<xlai_runtime::XlaiRuntime, XlaiError> {
    let api_key = require_env("OPENAI_API_KEY")?;
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());
    let chat_model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_owned());
    let embedding_model = require_env("OPENAI_EMBEDDING_MODEL")?;

    RuntimeBuilder::new()
        .with_embedding_backend(
            OpenAiConfig::new(base_url, api_key, chat_model).with_embedding_model(embedding_model),
        )
        .build()
}

fn require_env(name: &str) -> Result<String, XlaiError> {
    std::env::var(name).map_err(|error| {
        XlaiError::new(
            ErrorKind::Configuration,
            format!("{name} must be set for e2e tests: {error}"),
        )
    })
}
