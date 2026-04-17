//! `xlai-native` always pulls local chat via `xlai-facade` with `llama`. Optional `qts` adds a second native GGML stack.

fn main() {
    if std::env::var("CARGO_FEATURE_QTS").is_ok() {
        println!(
            "cargo:warning=xlai-native: `qts` links `xlai-sys-ggml` alongside the default `llama` stack (`xlai-sys-llama`). This loads two separate ggml implementations in one binary. See ARCHITECTURE.md."
        );
    }
}
