use crate::embeddings::{
    GeminiBatchEmbeddingRequest, GeminiBatchEmbeddingResponse, GeminiEmbeddingRequestPayload,
    GeminiEmbeddingResponse,
};
use crate::request::GeminiChatRequest;
use serde_json::json;
use xlai_core::{ChatContent, ChatRequest, MessageRole};

#[test]
fn serializes_basic_request() {
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![
            xlai_core::ChatMessage {
                role: MessageRole::System,
                content: ChatContent::text("You are a helpful assistant."),
                tool_name: None,
                tool_call_id: None,
                metadata: Default::default(),
            },
            xlai_core::ChatMessage {
                role: MessageRole::User,
                content: ChatContent::text("Hello!"),
                tool_name: None,
                tool_call_id: None,
                metadata: Default::default(),
            },
        ],
        available_tools: Vec::new(),
        structured_output: None,
        metadata: Default::default(),
        temperature: None,
        max_output_tokens: None,
        reasoning_effort: None,
        retry_policy: None,
        ..Default::default()
    };

    let gemini_req = GeminiChatRequest::from_core_request(request).unwrap();

    assert!(gemini_req.system_instruction.is_some());
    assert_eq!(gemini_req.contents.len(), 1);
    assert_eq!(gemini_req.contents[0].role, "user");
}

#[test]
fn gemini_embedding_request_serializes_single_and_batch_payloads() {
    let single = GeminiEmbeddingRequestPayload::from_core_request(
        "gemini-embedding-001".to_owned(),
        "hello".to_owned(),
        Some(768),
    );
    let single_json = serde_json::to_value(single).expect("serialize single payload");
    assert_eq!(single_json["model"], json!("gemini-embedding-001"));
    assert_eq!(single_json["content"]["parts"][0]["text"], json!("hello"));
    assert_eq!(single_json["outputDimensionality"], json!(768));

    let batch = GeminiBatchEmbeddingRequest::from_core_request(
        "gemini-embedding-001".to_owned(),
        vec!["hello".to_owned(), "world".to_owned()],
        Some(1536),
    );
    let batch_json = serde_json::to_value(batch).expect("serialize batch payload");
    assert_eq!(
        batch_json["requests"][0]["model"],
        json!("gemini-embedding-001")
    );
    assert_eq!(
        batch_json["requests"][1]["content"]["parts"][0]["text"],
        json!("world")
    );
    assert_eq!(
        batch_json["requests"][0]["outputDimensionality"],
        json!(1536)
    );
}

#[test]
fn gemini_embedding_response_maps_vectors_and_usage() {
    let single: GeminiEmbeddingResponse = serde_json::from_value(json!({
        "embedding": { "values": [0.1, 0.2] },
        "usageMetadata": {
            "promptTokenCount": 8,
            "totalTokenCount": 8
        }
    }))
    .expect("deserialize single embedding response");
    let single = single.into_core_response().expect("map single response");
    assert_eq!(single.vectors, vec![vec![0.1, 0.2]]);
    assert_eq!(
        single.metadata.get("usage_metadata"),
        Some(&json!({
            "promptTokenCount": 8,
            "totalTokenCount": 8
        }))
    );
    assert_eq!(single.usage.expect("usage").total_tokens, 8);

    let batch: GeminiBatchEmbeddingResponse = serde_json::from_value(json!({
        "embeddings": [
            { "values": [0.1, 0.2] },
            { "values": [0.3, 0.4] }
        ]
    }))
    .expect("deserialize batch embedding response");
    let batch = batch.into_core_response().expect("map batch response");
    assert_eq!(batch.vectors, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
}
