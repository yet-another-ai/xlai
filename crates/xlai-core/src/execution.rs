//! Advisory execution hints for latency-sensitive and in-process workloads.
//!
//! Backends and runtimes may interpret these fields; unsupported combinations should be ignored.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

/// Shared cooperative cancellation flag for in-process operations.
///
/// Clone the signal to pass into concurrent tasks; cancel by calling [`Self::cancel`].
#[derive(Clone, Debug, Default)]
pub struct CancellationSignal {
    cancelled: Arc<AtomicBool>,
}

impl PartialEq for CancellationSignal {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.cancelled, &other.cancelled)
    }
}

impl Eq for CancellationSignal {}

impl CancellationSignal {
    #[must_use]
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Marks the operation as cancelled. Idempotent.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Clears the cancelled state. Intended for tests and pooled workers.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::Release);
    }
}

/// Latency vs throughput trade-off hint for local runtimes and backends.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionLatencyMode {
    /// Prefer steady throughput (batching, fewer yields).
    #[default]
    Balanced,
    /// Prefer low tail latency and frequent yields (interactive / in-game).
    Interactive,
    /// Prefer maximum tokens/sec when the GPU is otherwise idle.
    Throughput,
}

/// Partial per-field overrides merged in order: runtime defaults, then session, then request.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ChatExecutionOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_mode: Option<ExecutionLatencyMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streaming_preferred: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup_on_create: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancel_on_drop: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_per_second: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_preference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_fallback_order: Option<Vec<String>>,
}

impl ChatExecutionOverrides {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.latency_mode.is_none()
            && self.streaming_preferred.is_none()
            && self.warmup_on_create.is_none()
            && self.cancel_on_drop.is_none()
            && self.max_tokens_per_second.is_none()
            && self.backend_preference.is_none()
            && self.backend_fallback_order.is_none()
    }
}

/// Resolved execution configuration after merging override layers.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct ChatExecutionConfig {
    pub latency_mode: ExecutionLatencyMode,
    pub streaming_preferred: bool,
    pub warmup_on_create: bool,
    pub cancel_on_drop: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_per_second: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_preference: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backend_fallback_order: Vec<String>,
}

impl ChatExecutionConfig {
    /// Merges optional override layers in order: `runtime`, then `session`, then `request`.
    /// Later layers win per field when `Some`.
    #[must_use]
    pub fn merge_optional_layers(
        runtime: Option<&ChatExecutionOverrides>,
        session: Option<&ChatExecutionOverrides>,
        request: Option<&ChatExecutionOverrides>,
    ) -> Self {
        let mut out = Self::default();
        for layer in [runtime, session, request] {
            let Some(layer) = layer else {
                continue;
            };
            if let Some(v) = layer.latency_mode {
                out.latency_mode = v;
            }
            if let Some(v) = layer.streaming_preferred {
                out.streaming_preferred = v;
            }
            if let Some(v) = layer.warmup_on_create {
                out.warmup_on_create = v;
            }
            if let Some(v) = layer.cancel_on_drop {
                out.cancel_on_drop = v;
            }
            if layer.max_tokens_per_second.is_some() {
                out.max_tokens_per_second = layer.max_tokens_per_second;
            }
            if layer.backend_preference.is_some() {
                out.backend_preference = layer.backend_preference.clone();
            }
            if let Some(fallback) = &layer.backend_fallback_order {
                out.backend_fallback_order.clone_from(fallback);
            }
        }
        out
    }
}

/// Optional overrides for TTS synthesis (mirrors chat execution hints where applicable).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TtsExecutionOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_mode: Option<ExecutionLatencyMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancel_on_drop: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_preference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_fallback_order: Option<Vec<String>>,
}

impl TtsExecutionOverrides {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.latency_mode.is_none()
            && self.cancel_on_drop.is_none()
            && self.backend_preference.is_none()
            && self.backend_fallback_order.is_none()
    }
}

/// Resolved TTS execution configuration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct TtsExecutionConfig {
    pub latency_mode: ExecutionLatencyMode,
    pub cancel_on_drop: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_preference: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backend_fallback_order: Vec<String>,
}

impl TtsExecutionConfig {
    /// Converts this resolved config into a dense override layer (all fields `Some`).
    #[must_use]
    pub fn to_overrides(&self) -> TtsExecutionOverrides {
        TtsExecutionOverrides {
            latency_mode: Some(self.latency_mode),
            cancel_on_drop: Some(self.cancel_on_drop),
            backend_preference: self.backend_preference.clone(),
            backend_fallback_order: Some(self.backend_fallback_order.clone()),
        }
    }

    #[must_use]
    pub fn merge_optional_layers(
        runtime: Option<&TtsExecutionOverrides>,
        session: Option<&TtsExecutionOverrides>,
        request: Option<&TtsExecutionOverrides>,
    ) -> Self {
        let mut out = Self::default();
        for layer in [runtime, session, request] {
            let Some(layer) = layer else {
                continue;
            };
            if let Some(v) = layer.latency_mode {
                out.latency_mode = v;
            }
            if let Some(v) = layer.cancel_on_drop {
                out.cancel_on_drop = v;
            }
            if layer.backend_preference.is_some() {
                out.backend_preference = layer.backend_preference.clone();
            }
            if let Some(fallback) = &layer.backend_fallback_order {
                out.backend_fallback_order.clone_from(fallback);
            }
        }
        out
    }
}

#[cfg(test)]
mod merge_tests {
    use super::*;

    #[test]
    fn chat_execution_merge_respects_layer_order() {
        let runtime = ChatExecutionOverrides {
            latency_mode: Some(ExecutionLatencyMode::Balanced),
            streaming_preferred: Some(true),
            ..Default::default()
        };
        let session = ChatExecutionOverrides {
            latency_mode: Some(ExecutionLatencyMode::Interactive),
            cancel_on_drop: Some(true),
            ..Default::default()
        };
        let request = ChatExecutionOverrides {
            max_tokens_per_second: Some(30.0),
            ..Default::default()
        };

        let merged = ChatExecutionConfig::merge_optional_layers(
            Some(&runtime),
            Some(&session),
            Some(&request),
        );

        assert_eq!(merged.latency_mode, ExecutionLatencyMode::Interactive);
        assert!(merged.streaming_preferred);
        assert!(merged.cancel_on_drop);
        assert_eq!(merged.max_tokens_per_second, Some(30.0));
    }
}
