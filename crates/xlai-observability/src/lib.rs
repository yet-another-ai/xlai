//! Process-wide [`tracing`] initialization for xlai binaries and examples.
//!
//! Library crates should only emit spans/events; call [`init_logging`] or
//! [`try_init_logging`] from `main` (or tests) exactly once per process.

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

const DEFAULT_FILTER: &str = "info";

/// Build an [`EnvFilter`]: `XLAI_LOG` if set and valid, else `RUST_LOG`, else `info`.
#[must_use]
pub fn env_filter() -> EnvFilter {
    if let Ok(spec) = std::env::var("XLAI_LOG")
        && let Ok(filter) = EnvFilter::try_new(&spec)
    {
        return filter;
    }
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER))
}

/// Try to install the global tracing subscriber (stderr, filtered).
///
/// # Errors
///
/// Returns [`tracing_subscriber::util::TryInitError`] if a global default was already set.
pub fn try_init_logging() -> Result<(), tracing_subscriber::util::TryInitError> {
    let filter = env_filter();
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(true).with_writer(std::io::stderr))
        .try_init()
}

/// Same as [`try_init_logging`], but ignores duplicate initialization.
pub fn init_logging() {
    let _ = try_init_logging();
}
