//! Process-wide `tracing` initialization (inlined from former `xlai-observability`).

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

const DEFAULT_FILTER: &str = "info";

#[must_use]
pub fn env_filter() -> EnvFilter {
    if let Ok(spec) = std::env::var("XLAI_LOG")
        && let Ok(filter) = EnvFilter::try_new(&spec)
    {
        return filter;
    }
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER))
}

pub fn try_init_logging() -> Result<(), tracing_subscriber::util::TryInitError> {
    let filter = env_filter();
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(true).with_writer(std::io::stderr))
        .try_init()
}

pub fn init_logging() {
    let _ = try_init_logging();
}
