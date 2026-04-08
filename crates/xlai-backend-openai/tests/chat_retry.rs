//! Integration tests for chat completion retries against a mock HTTP server.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};
use xlai_backend_openai::{OpenAiChatModel, OpenAiConfig};
use xlai_core::{
    ChatContent, ChatMessage, ChatModel, ChatRequest, ChatRetryPolicy, ErrorKind, MessageRole,
};

fn minimal_success_json() -> serde_json::Value {
    json!({
        "choices": [{
            "message": { "content": "ok" },
            "finish_reason": "stop"
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
    }
}

#[tokio::test]
async fn no_policy_single_503_fails_without_second_request() {
    let server = MockServer::start().await;
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_clone = hits.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            hits_clone.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(503).set_body_string("unavailable")
        })
        .expect(1)
        .mount(&server)
        .await;

    let config = OpenAiConfig::new(format!("{}/v1", server.uri()), "k", "gpt-test");
    let model = OpenAiChatModel::new(config);
    let out = model.generate(minimal_chat_request(None)).await;
    assert!(out.is_err(), "expected provider error");
    let Err(err) = out else {
        return;
    };

    assert_eq!(err.kind, ErrorKind::Provider);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn retry_policy_recovers_after_transient_503() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
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

    let config = OpenAiConfig::new(format!("{}/v1", server.uri()), "k", "gpt-test");
    let model = OpenAiChatModel::new(config);
    let out = model.generate(minimal_chat_request(Some(policy))).await;
    assert!(out.is_ok(), "retry should succeed");
    let Ok(response) = out else {
        return;
    };

    assert_eq!(response.message.content.as_single_text(), Some("ok"));
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn does_not_retry_when_policy_disabled() {
    let server = MockServer::start().await;
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_clone = hits.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            hits_clone.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(503).set_body_string("unavailable")
        })
        .expect(1)
        .mount(&server)
        .await;

    let policy = ChatRetryPolicy::disabled();
    let config = OpenAiConfig::new(format!("{}/v1", server.uri()), "k", "gpt-test");
    let model = OpenAiChatModel::new(config);
    let out = model.generate(minimal_chat_request(Some(policy))).await;
    assert!(out.is_err(), "disabled retry should fail on 503");

    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn does_not_retry_non_retryable_http_error() {
    let server = MockServer::start().await;
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_clone = hits.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
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

    let config = OpenAiConfig::new(format!("{}/v1", server.uri()), "k", "gpt-test");
    let model = OpenAiChatModel::new(config);
    let out = model.generate(minimal_chat_request(Some(policy))).await;
    assert!(out.is_err(), "401 should not succeed");
    let Err(err) = out else {
        return;
    };

    assert_eq!(err.kind, ErrorKind::Provider);
    assert_eq!(err.retryable, Some(false));
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}
