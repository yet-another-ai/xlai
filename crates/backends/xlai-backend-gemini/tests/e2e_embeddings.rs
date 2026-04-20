use xlai_backend_gemini::GeminiConfig;
use xlai_core::{EmbeddingBackend, EmbeddingModel, EmbeddingRequest, XlaiError};

#[tokio::test]
async fn gemini_embeddings_smoke_test() -> Result<(), XlaiError> {
    let api_key = match std::env::var("GEMINI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            println!("Skipping gemini_embeddings_smoke_test because GEMINI_API_KEY is not set");
            return Ok(());
        }
    };

    let base_url = std::env::var("GEMINI_BASE_URL")
        .unwrap_or_else(|_| "https://generativelanguage.googleapis.com/v1beta".to_owned());
    let model_name = std::env::var("GEMINI_EMBEDDING_MODEL")
        .unwrap_or_else(|_| "gemini-embedding-001".to_owned());

    let config =
        GeminiConfig::new(base_url, api_key, "gemini-2.5-flash").with_embedding_model(model_name);
    let model = config.into_embedding_model();

    let response = model
        .embed(EmbeddingRequest {
            model: None,
            inputs: vec!["A short smoke-test sentence for embeddings.".to_owned()],
            dimensions: Some(768),
            metadata: Default::default(),
        })
        .await?;

    assert_eq!(response.vectors.len(), 1);
    assert!(
        !response.vectors[0].is_empty(),
        "Response should contain a non-empty embedding vector"
    );

    Ok(())
}
