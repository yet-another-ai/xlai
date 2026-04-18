#[test]
fn provider_error_message_surfaces_openai_quota_details() {
    let message = crate::provider_response::format_provider_error_message(
        reqwest::StatusCode::TOO_MANY_REQUESTS,
        Some("req_123"),
        r#"{
                "error": {
                    "message": "You exceeded your current quota.",
                    "type": "insufficient_quota",
                    "param": "model",
                    "code": "insufficient_quota"
                }
            }"#,
    );

    assert!(message.contains("429 Too Many Requests"));
    assert!(message.contains("request_id=req_123"));
    assert!(message.contains("You exceeded your current quota."));
    assert!(message.contains("type=insufficient_quota"));
    assert!(message.contains("code=insufficient_quota"));
    assert!(message.contains("param=model"));
}

#[test]
fn provider_error_message_falls_back_to_raw_body_for_non_json_errors() {
    let message = crate::provider_response::format_provider_error_message(
        reqwest::StatusCode::BAD_GATEWAY,
        None,
        "upstream gateway timeout",
    );

    assert!(message.contains("502 Bad Gateway"));
    assert!(message.contains("upstream gateway timeout"));
}
