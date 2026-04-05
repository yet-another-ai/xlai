//! Bridges `llama.cpp` / ggml log callbacks into the [`tracing`] crate.

use std::borrow::Cow;
use std::ffi::{CStr, c_char, c_void};
use std::sync::Mutex;

use super::raw;

struct LogState {
    buf: String,
    pending_start_level: Option<raw::ggml_log_level>,
}

impl LogState {
    const fn new() -> Self {
        Self {
            buf: String::new(),
            pending_start_level: None,
        }
    }
}

static LOG_STATE: Mutex<LogState> = Mutex::new(LogState::new());

/// Installs a global `llama_log_set` handler that forwards to `tracing`.
///
/// Safe to call once per process; [`super::ensure_backend_initialized`] does this before
/// `llama_backend_init`.
pub(crate) fn install() {
    // SAFETY: `llama_log_set` is documented as setting a process-wide callback; we pass a
    // function pointer with C ABI and null user data.
    unsafe {
        raw::llama_log_set(Some(llama_tracing_log_callback), std::ptr::null_mut());
    }
}

unsafe extern "C" fn llama_tracing_log_callback(
    level: raw::ggml_log_level,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    let text: Cow<'_, str> = if text.is_null() {
        Cow::Borrowed("")
    } else {
        // SAFETY: llama.cpp passes a NUL-terminated fragment for the duration of the callback.
        unsafe { CStr::from_ptr(text).to_string_lossy() }
    };
    on_native_fragment(level, text.as_ref());
}

fn on_native_fragment(level: raw::ggml_log_level, text: &str) {
    let Ok(mut st) = LOG_STATE.lock() else {
        return;
    };

    if level == raw::ggml_log_level_GGML_LOG_LEVEL_CONT {
        st.buf.push_str(text);
        return;
    }

    if !st.buf.is_empty() {
        let emit_level = st.pending_start_level.unwrap_or(level);
        let drained = std::mem::take(&mut st.buf);
        st.pending_start_level = None;
        emit_full_message(emit_level, &drained);
    }

    st.pending_start_level = Some(level);
    st.buf.push_str(text);

    while let Some(pos) = st.buf.find('\n') {
        let line = st.buf[..pos].to_string();
        st.buf.drain(..=pos);
        let lvl = st.pending_start_level.unwrap_or(level);
        if !line.is_empty() {
            emit_line(lvl, line.trim_end_matches('\r'));
        }
    }
}

fn emit_full_message(level: raw::ggml_log_level, message: &str) {
    let trimmed = message.trim_end_matches(['\n', '\r']);
    if trimmed.is_empty() {
        return;
    }
    for line in trimmed.split('\n') {
        let line = line.trim_end_matches('\r');
        if !line.is_empty() {
            emit_line(level, line);
        }
    }
}

fn emit_line(level: raw::ggml_log_level, message: &str) {
    if message.is_empty() {
        return;
    }
    const TARGET: &str = "xlai::native::llama";
    match level {
        raw::ggml_log_level_GGML_LOG_LEVEL_ERROR => {
            tracing::error!(target: TARGET, "{message}");
        }
        raw::ggml_log_level_GGML_LOG_LEVEL_WARN => {
            tracing::warn!(target: TARGET, "{message}");
        }
        raw::ggml_log_level_GGML_LOG_LEVEL_INFO | raw::ggml_log_level_GGML_LOG_LEVEL_NONE => {
            tracing::info!(target: TARGET, "{message}");
        }
        raw::ggml_log_level_GGML_LOG_LEVEL_DEBUG => {
            tracing::debug!(target: TARGET, "{message}");
        }
        raw::ggml_log_level_GGML_LOG_LEVEL_CONT => {
            tracing::debug!(target: TARGET, "{message}");
        }
        _ => {
            tracing::debug!(target: TARGET, "{message}");
        }
    }
}
