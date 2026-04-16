//! Warn when both local chat (`llama`) and native QTS (`qts`) are enabled: two separate native GGML stacks.

fn main() {
    let llama = std::env::var("CARGO_FEATURE_LLAMA").is_ok();
    let qts = std::env::var("CARGO_FEATURE_QTS").is_ok();
    if llama && qts {
        println!(
            "cargo:warning=xlai-facade: `llama` + `qts` link two native stacks (`xlai-sys-llama` bundles ggml with llama.cpp; `xlai-sys-ggml` links standalone ggml). If you see duplicate symbols or instability, use separate processes or disable one stack. See ARCHITECTURE.md."
        );
    }
}
