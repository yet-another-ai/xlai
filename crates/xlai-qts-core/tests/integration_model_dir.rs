//! Layer-B style checks: requires real model artifacts on disk.
//!
//! ```text
//! export XLAI_QTS_MODEL_DIR=/path/to/models   # contains qwen3-tts-0.6b-f16.gguf + qwen3-tts-vocoder.onnx
//! cargo test -p xlai-qts-core integration_ -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use xlai_qts_core::{Qwen3TtsEngine, SynthesizeRequest, VoiceCloneMode, VoiceClonePromptV2};

fn require_model_dir() -> PathBuf {
    std::env::var("XLAI_QTS_MODEL_DIR")
        .map(PathBuf::from)
        .expect("XLAI_QTS_MODEL_DIR must be set when running ignored integration tests")
}

fn load_fixture_prompt(engine: &Qwen3TtsEngine, name: &str) -> VoiceClonePromptV2 {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|err| {
        panic!(
            "failed to read fixture {}: {err}\n\
             regenerate with: XLAI_QTS_MODEL_DIR=/path/to/models cargo run -p xlai-qts-core --example export_voice_clone_prompts",
            path.display()
        );
    });
    engine
        .decode_voice_clone_prompt(&bytes)
        .expect("decode voice clone prompt fixture")
}

fn shared_ref_audio_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("fixtures/audio/transcription-sample.wav")
}

#[test]
#[ignore = "set XLAI_QTS_MODEL_DIR to run"]
fn integration_loads_models() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load models");
    assert!(engine.model_paths().main_exists());
    assert!(engine.model_paths().vocoder_exists());
    assert!(!engine.encode_for_tts("hello").is_empty());
}

#[test]
#[ignore = "set XLAI_QTS_MODEL_DIR to run"]
fn integration_synthesize_direct_path_audio() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load");
    let req = SynthesizeRequest {
        text: "hello".into(),
        max_audio_frames: 4,
        ..Default::default()
    };
    let result = engine.synthesize(&req).expect("synthesize audio");
    assert_eq!(result.sample_rate_hz, 24_000);
    assert!(result.generated_frames > 0);
    assert!(!result.pcm_f32.is_empty());
}

#[test]
#[ignore = "set XLAI_QTS_MODEL_DIR to run"]
fn integration_voice_clone_prompt_xvector_mode() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load");
    let prompt = load_fixture_prompt(&engine, "sample1.xvector.voice-clone-prompt.cbor");
    assert!(prompt.x_vector_only_mode);
    assert!(!prompt.icl_mode);
    assert!(prompt.ref_code.is_none());
    assert_eq!(
        prompt.speaker_embedding_dim(),
        Some(engine.speaker_embedding_size())
    );
    let req = SynthesizeRequest {
        text: "hello".into(),
        max_audio_frames: 4,
        ..Default::default()
    };
    let result = engine
        .synthesize_with_voice_clone_prompt(&req, &prompt)
        .expect("synthesize xvector prompt");
    assert_eq!(result.sample_rate_hz, 24_000);
    assert!(result.generated_frames > 0);
    assert!(!result.pcm_f32.is_empty());
}

#[test]
#[ignore = "set XLAI_QTS_MODEL_DIR to run"]
fn integration_voice_clone_prompt_icl_mode() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load");
    let prompt = load_fixture_prompt(&engine, "sample1.icl.voice-clone-prompt.cbor");
    assert!(!prompt.x_vector_only_mode);
    assert!(prompt.icl_mode);
    assert!(prompt.ref_code.is_some());
    assert!(!prompt.ref_text.is_empty());
    assert_eq!(
        prompt.speaker_embedding_dim(),
        Some(engine.speaker_embedding_size())
    );
    let (frames, codebooks) = prompt.ref_code_shape().expect("ICL ref_code shape");
    assert!(frames > 0, "expected at least one ICL reference frame");
    assert_eq!(
        codebooks,
        engine.transformer().config().n_codebooks as usize
    );
    let req = SynthesizeRequest {
        text: "world".into(),
        max_audio_frames: 4,
        ..Default::default()
    };
    let result = engine
        .synthesize_with_voice_clone_prompt(&req, &prompt)
        .expect("synthesize icl prompt");
    assert_eq!(result.sample_rate_hz, 24_000);
    assert!(result.generated_frames > 0);
    assert!(!result.pcm_f32.is_empty());
}

#[test]
#[ignore = "set XLAI_QTS_MODEL_DIR to run"]
fn integration_native_xvector_prompt_parity_shape() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load");
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("sample1.xvector.voice-clone-prompt.cbor");
    if !fixture_path.is_file() {
        eprintln!("skip: missing fixture {}", fixture_path.display());
        return;
    }
    let golden = load_fixture_prompt(&engine, "sample1.xvector.voice-clone-prompt.cbor");
    let local_ref_wav = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("sample1_ref.wav");
    let ref_wav = if local_ref_wav.is_file() {
        local_ref_wav
    } else {
        shared_ref_audio_path()
    };
    if !ref_wav.is_file() {
        eprintln!(
            "skip: missing reference WAV {}; run the exporter example or place sample1_ref.wav next to fixtures",
            ref_wav.display()
        );
        return;
    }
    let wav = std::fs::read(&ref_wav).expect("read ref wav");
    let built = engine
        .create_voice_clone_prompt(&wav, None, VoiceCloneMode::XVectorOnly)
        .expect("native xvector prompt");
    assert_eq!(built.x_vector_only_mode, golden.x_vector_only_mode);
    assert_eq!(built.icl_mode, golden.icl_mode);
    assert_eq!(
        built.speaker_embedding_dim(),
        golden.speaker_embedding_dim()
    );
}
