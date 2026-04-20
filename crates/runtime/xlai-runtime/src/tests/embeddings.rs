#![allow(clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use xlai_core::{EmbeddingRequest, EmbeddingResponse, ErrorKind, XlaiError};

use super::common::{RecordingEmbeddingModel, empty_metadata, lock_unpoisoned};
use crate::RuntimeBuilder;

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_embed_forwards_requests_to_embedding_model() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let runtime = RuntimeBuilder::new()
        .with_embedding_model(Arc::new(RecordingEmbeddingModel::new(
            requests.clone(),
            vec![EmbeddingResponse {
                vectors: vec![vec![0.1, 0.2, 0.3]],
                usage: None,
                metadata: empty_metadata(),
            }],
        )))
        .build()?;

    let response = runtime
        .embed(EmbeddingRequest {
            model: Some("test-embedding".to_owned()),
            inputs: vec!["hello".to_owned()],
            dimensions: Some(3),
            metadata: empty_metadata(),
        })
        .await?;

    assert_eq!(response.vectors, vec![vec![0.1, 0.2, 0.3]]);

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].dimensions, Some(3));

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_builder_with_embedding_backend_adds_capability() -> Result<(), XlaiError> {
    let runtime = RuntimeBuilder::new()
        .with_embedding_backend(RecordingEmbeddingModel::new(
            Arc::new(Mutex::new(Vec::new())),
            vec![EmbeddingResponse {
                vectors: vec![vec![1.0]],
                usage: None,
                metadata: empty_metadata(),
            }],
        ))
        .build()?;

    assert!(runtime.has_capability(xlai_core::RuntimeCapability::Embeddings));
    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_embed_requires_configured_embedding_model() {
    let runtime = RuntimeBuilder::new()
        .with_chat_model(Arc::new(super::common::RecordingChatModel::new(
            Arc::new(Mutex::new(Vec::new())),
            Vec::new(),
        )))
        .build()
        .expect("runtime should build");

    let error = runtime
        .embed(EmbeddingRequest {
            model: None,
            inputs: vec!["hello".to_owned()],
            dimensions: None,
            metadata: empty_metadata(),
        })
        .await
        .expect_err("missing embedding model should error");

    assert_eq!(error.kind, ErrorKind::Unsupported);
    assert!(error.message.contains("embedding model"));
}
