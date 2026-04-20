//! Integration tests for responses API retries against a mock HTTP server.

#![allow(clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};
use xlai_backend_openrouter::{OpenRouterChatModel, OpenRouterConfig};
use xlai_core::{
    ChatContent, ChatMessage, ChatModel, ChatRequest, ChatRetryPolicy, ErrorKind, MessageRole,
};

fn minimal_success_json() -> serde_json::Value {
    json!({
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": "ok",
                "annotations": []
            }]
        }]
    })
}

fn minimal_chat_request(retry_policy: Option<ChatRetryPolicy>) -> ChatRequest {
    ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: ChatContent::text("hi"),
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
        retry_policy,
        ..Default::default()
    }
}

#[tokio::test]
async fn retry_policy_recovers_after_transient_503() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();
    Mock::given(method("POST"))
        .and(path("/api/v1/responses"))
        .and(header("HTTP-Referer", "https://xlai.example"))
        .and(header("X-OpenRouter-Title", "xlai"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = count_clone.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                ResponseTemplate::new(503).set_body_string("unavailable")
            } else {
                ResponseTemplate::new(200).set_body_json(minimal_success_json())
            }
        })
        .expect(2)
        .mount(&server)
        .await;

    let policy = ChatRetryPolicy::default()
        .with_max_retries(2)
        .with_initial_backoff_ms(0)
        .with_max_backoff_ms(0);

    let config = OpenRouterConfig::new(format!("{}/api/v1", server.uri()), "k", "gpt-test")
        .with_http_referer("https://xlai.example")
        .with_app_title("xlai");
    let model = OpenRouterChatModel::new(config);
    let out = model.generate(minimal_chat_request(Some(policy))).await;
    let response = out.expect("retry should succeed");

    assert_eq!(response.message.content.as_single_text(), Some("ok"));
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn does_not_retry_non_retryable_http_error() {
    let server = MockServer::start().await;
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_clone = hits.clone();
    Mock::given(method("POST"))
        .and(path("/api/v1/responses"))
        .respond_with(move |_req: &wiremock::Request| {
            hits_clone.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(401).set_body_string("nope")
        })
        .expect(1)
        .mount(&server)
        .await;

    let policy = ChatRetryPolicy::default()
        .with_max_retries(3)
        .with_initial_backoff_ms(0)
        .with_max_backoff_ms(0);

    let config = OpenRouterConfig::new(format!("{}/api/v1", server.uri()), "k", "gpt-test");
    let model = OpenRouterChatModel::new(config);
    let out = model.generate(minimal_chat_request(Some(policy))).await;
    let err = out.expect_err("401 should not succeed");

    assert_eq!(err.kind, ErrorKind::Provider);
    assert_eq!(err.retryable, Some(false));
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}
