#![allow(clippy::expect_used, clippy::panic)]

//! Unit tests for the llama.cpp backend.

mod prompt_and_request;
mod structured_validation;
mod tool_parsing;

use xlai_core::{EmbeddingModel, EmbeddingRequest, ErrorKind};

use crate::{LlamaCppConfig, LlamaCppEmbeddingModel};

#[tokio::test]
async fn llama_embeddings_reject_custom_dimensions() {
    let model = LlamaCppEmbeddingModel::new(LlamaCppConfig::new("fake.gguf"));
    let error = model
        .embed(EmbeddingRequest {
            model: None,
            inputs: vec!["hello".to_owned()],
            dimensions: Some(256),
            metadata: Default::default(),
        })
        .await
        .expect_err("custom dimensions should be rejected");

    assert_eq!(error.kind, ErrorKind::Unsupported);
    assert!(error.message.contains("dimensions"));
}
