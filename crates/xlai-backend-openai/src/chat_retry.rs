//! Bounded retries for chat completions (unary and stream handshake).

use std::time::Duration;

use xlai_core::{ChatRetryPolicy, XlaiError};

pub(crate) fn should_retry_xlai_error(err: &XlaiError) -> bool {
    match err.retryable {
        Some(false) => false,
        Some(true) => true,
        None => false,
    }
}

/// Returns `(max_extra_attempts_after_failure, policy_for_backoff)`.
pub(crate) fn retry_limits_for_chat_request(
    policy: Option<&ChatRetryPolicy>,
) -> (u32, Option<&ChatRetryPolicy>) {
    match policy {
        None => (0, None),
        Some(p) if !p.enabled => (0, None),
        Some(p) => (p.max_retries, Some(p)),
    }
}

pub(crate) fn backoff_delay_ms(policy: &ChatRetryPolicy, failure_index: u32) -> u64 {
    let shift = failure_index.min(31);
    let multiplied = policy.initial_backoff_ms.saturating_mul(1u64 << shift);
    multiplied.min(policy.max_backoff_ms)
}

pub(crate) async fn sleep_ms(ms: u64) {
    if ms == 0 {
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    tokio::time::sleep(Duration::from_millis(ms)).await;
    #[cfg(target_arch = "wasm32")]
    gloo_timers::future::sleep(Duration::from_millis(ms)).await;
}
