//! Poll-friendly async handle over chat execution streams with optional cooperative cancellation.

use futures_util::StreamExt;
use xlai_core::{BoxStream, ErrorKind, XlaiError};

use crate::ChatExecutionEvent;

/// Handle for driving a [`ChatExecutionEvent`] stream with an explicit cancellation signal.
///
/// Prefer [`Self::next_event`] from an async context (game loop task, UI runtime). Dropping the
/// handle without finishing may trigger cancellation when `cancel_on_drop` was set at creation.
pub struct ChatExecutionHandle {
    stream: BoxStream<'static, Result<ChatExecutionEvent, XlaiError>>,
    cancellation: xlai_core::CancellationSignal,
    cancel_on_drop: bool,
    finished: bool,
}

impl ChatExecutionHandle {
    pub(crate) fn new(
        stream: BoxStream<'static, Result<ChatExecutionEvent, XlaiError>>,
        cancellation: xlai_core::CancellationSignal,
        cancel_on_drop: bool,
    ) -> Self {
        Self {
            stream,
            cancellation,
            cancel_on_drop,
            finished: false,
        }
    }

    /// Shared cancellation signal for cooperative backends.
    #[must_use]
    pub fn cancellation(&self) -> xlai_core::CancellationSignal {
        self.cancellation.clone()
    }

    /// Returns the next event, or [`None`] when the stream has ended successfully.
    ///
    /// When cancelled, returns `Some(Err(XlaiError { kind: Cancelled, .. }))`.
    pub async fn next_event(&mut self) -> Option<Result<ChatExecutionEvent, XlaiError>> {
        if self.cancellation.is_cancelled() {
            self.finished = true;
            return Some(Err(XlaiError::new(
                ErrorKind::Cancelled,
                "chat execution was cancelled",
            )));
        }

        let item = match self.stream.next().await {
            None => {
                self.finished = true;
                return None;
            }
            Some(item) => item,
        };

        if self.cancellation.is_cancelled() {
            self.finished = true;
            return Some(Err(XlaiError::new(
                ErrorKind::Cancelled,
                "chat execution was cancelled",
            )));
        }

        Some(item)
    }
}

impl Drop for ChatExecutionHandle {
    fn drop(&mut self) {
        if self.finished || !self.cancel_on_drop {
            return;
        }
        self.cancellation.cancel();
    }
}
