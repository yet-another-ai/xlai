//! Stage modules mirroring `predict-woo/qwen3-tts.cpp` (implemented incrementally).

pub(crate) mod backend;
mod byte_unicode;
pub mod speaker_encoder;
pub mod reference_codec_encoder;
pub mod tokenizer;
pub mod tts_transformer;
pub mod vocoder;
