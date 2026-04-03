use reqwest::StatusCode;
use xlai_core::{ErrorKind, XlaiError};

use crate::response::OpenAiErrorEnvelope;

pub(crate) async fn require_success_response(
    response: reqwest::Response,
) -> Result<reqwest::Response, XlaiError> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let request_id = response
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("<failed to read provider error body: {error}>"));

    let message = format_provider_error_message(status, request_id.as_deref(), &body);
    let status_u16 = status.as_u16();
    let mut err = XlaiError::new(ErrorKind::Provider, message).with_http_status(status_u16);
    if let Some(rid) = request_id {
        err = err.with_request_id(rid);
    }
    if let Ok(envelope) = serde_json::from_str::<OpenAiErrorEnvelope>(&body)
        && let Some(code) = envelope.error.code
    {
        err = err.with_provider_code(code);
    }
    err = err.with_retryable(http_status_suggests_retry(status_u16));
    Err(err)
}

fn http_status_suggests_retry(status: u16) -> bool {
    matches!(status, 408 | 409 | 425 | 429 | 500 | 502 | 503 | 504)
}

pub(crate) fn format_provider_error_message(
    status: StatusCode,
    request_id: Option<&str>,
    body: &str,
) -> String {
    let body = body.trim();

    let mut message = format!("openai-compatible request failed with {status}");
    if let Some(request_id) = request_id {
        message.push_str(&format!(" (request_id={request_id})"));
    }

    if body.is_empty() {
        return message;
    }

    if let Ok(envelope) = serde_json::from_str::<OpenAiErrorEnvelope>(body) {
        message.push_str(": ");
        message.push_str(&envelope.error.message);

        let mut details = Vec::new();
        if let Some(kind) = envelope.error.kind.as_deref() {
            details.push(format!("type={kind}"));
        }
        if let Some(code) = envelope.error.code.as_deref() {
            details.push(format!("code={code}"));
        }
        if let Some(param) = envelope.error.param.as_deref() {
            details.push(format!("param={param}"));
        }

        if !details.is_empty() {
            message.push_str(" [");
            message.push_str(&details.join(", "));
            message.push(']');
        }

        return message;
    }

    message.push_str(": ");
    message.push_str(body);
    message
}
