#![allow(clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use serde_json::json;
use xlai_core::EmbeddingRequest;

use crate::embeddings::{OpenAiEmbeddingRequest, OpenAiEmbeddingResponse};

use super::common::test_config;

#[test]
fn embedding_request_uses_explicit_or_configured_model_and_dimensions() {
    let config = test_config().with_embedding_model("text-embedding-3-small");
    let request = EmbeddingRequest {
        model: None,
        inputs: vec!["hello".to_owned(), "world".to_owned()],
        dimensions: Some(768),
        metadata: BTreeMap::new(),
    };

    let payload = OpenAiEmbeddingRequest::from_core_request(&config, request.clone());
    assert!(payload.is_ok(), "build embedding request");
    let payload_json = serde_json::to_value(payload.unwrap_or_else(|_| unreachable!()));
    assert!(payload_json.is_ok(), "serialize payload");
    let payload_json = payload_json.unwrap_or_else(|_| json!({}));
    assert_eq!(payload_json["model"], json!("text-embedding-3-small"));
    assert_eq!(payload_json["input"], json!(["hello", "world"]));
    assert_eq!(payload_json["encoding_format"], json!("float"));
    assert_eq!(payload_json["dimensions"], json!(768));

    let mut explicit = request;
    explicit.model = Some("text-embedding-3-large".to_owned());
    let payload = OpenAiEmbeddingRequest::from_core_request(&config, explicit);
    assert!(payload.is_ok(), "build explicit embedding request");
    let payload_json = serde_json::to_value(payload.unwrap_or_else(|_| unreachable!()));
    assert!(payload_json.is_ok(), "serialize explicit payload");
    assert_eq!(
        payload_json.unwrap_or_else(|_| json!({}))["model"],
        json!("text-embedding-3-large")
    );
}

#[test]
fn embedding_response_preserves_order_usage_and_model_metadata() {
    let response: Result<OpenAiEmbeddingResponse, _> = serde_json::from_value(json!({
        "data": [
            { "embedding": [0.1, 0.2], "index": 0 },
            { "embedding": [0.3, 0.4], "index": 1 }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8
        }
    }));
    assert!(response.is_ok(), "deserialize embedding response");
    let Ok(response) = response else {
        return;
    };

    let response = response.into_core_response();
    assert!(response.is_ok(), "map embedding response");
    let Ok(response) = response else {
        return;
    };

    assert_eq!(response.vectors, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    assert_eq!(
        response.metadata.get("model"),
        Some(&json!("text-embedding-3-small"))
    );
    let usage = response.usage.expect("usage should be present");
    assert_eq!(usage.input_tokens, 8);
    assert_eq!(usage.output_tokens, 0);
    assert_eq!(usage.total_tokens, 8);
}
