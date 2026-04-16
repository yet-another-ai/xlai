use std::env;

#[must_use]
pub fn feature_enabled(name: &str) -> bool {
    env::var(format!(
        "CARGO_FEATURE_{}",
        name.to_ascii_uppercase().replace('-', "_")
    ))
    .is_ok()
}
