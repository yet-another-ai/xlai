//! Shared tracing init for `xlai-qts-core` examples.

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

const DEFAULT_FILTER: &str = "info";

pub fn init_logging() {
    let filter = if let Ok(spec) = std::env::var("XLAI_LOG")
        && let Ok(f) = EnvFilter::try_new(&spec)
    {
        f
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER))
    };
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(true).with_writer(std::io::stderr))
        .try_init();
}
