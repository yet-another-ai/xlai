//! Native platform entrypoint: re-exports the shared [`xlai_facade`] integration surface
//! (llama.cpp, OpenAI, Gemini, QTS, transformers.js).
//!
//! Prefer this crate for native applications. Semver-stable domain types and traits live in
//! [`xlai_core`]; `xlai_facade` is an internal workspace crate (not on crates.io).

pub use xlai_facade::*;
