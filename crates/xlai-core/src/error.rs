use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ErrorKind {
    Configuration,
    Validation,
    Provider,
    Tool,
    Skill,
    Knowledge,
    Vector,
    FileSystem,
    Unsupported,
}

/// Structured error for xlai APIs. Optional fields are omitted from JSON when unset so older
/// clients keep working; deserializing legacy `{"kind","message"}` JSON fills them with `None`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct XlaiError {
    pub kind: ErrorKind,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Hint for callers: whether retrying the same request may succeed (best-effort).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl XlaiError {
    #[must_use]
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            provider_code: None,
            http_status: None,
            request_id: None,
            retryable: None,
            details: None,
        }
    }

    #[must_use]
    pub fn with_provider_code(mut self, code: impl Into<String>) -> Self {
        self.provider_code = Some(code.into());
        self
    }

    #[must_use]
    pub fn with_http_status(mut self, status: u16) -> Self {
        self.http_status = Some(status);
        self
    }

    #[must_use]
    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    #[must_use]
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = Some(retryable);
        self
    }

    #[must_use]
    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }
}

impl Display for XlaiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)?;
        if let Some(status) = self.http_status {
            write!(f, " (http_status={status})")?;
        }
        if let Some(id) = self.request_id.as_deref() {
            write!(f, " (request_id={id})")?;
        }
        Ok(())
    }
}

impl Error for XlaiError {}
