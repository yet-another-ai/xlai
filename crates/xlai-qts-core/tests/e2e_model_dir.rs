#![allow(clippy::expect_used)]

//! Ignored smoke tests for the full QTS runtime path.
//!
//! ```sh
//! export XLAI_QTS_MODEL_DIR=/path/to/models
//! export XLAI_QTS_REF_AUDIO_WAV=/path/to/reference.wav
//! export XLAI_QTS_REF_TEXT="optional transcript for ICL tests"
//! cargo test -p xlai-qts-core e2e_model_dir -- --ignored --nocapture --test-threads=1
//! ```

use std::env;
use std::path::PathBuf;

use futures_util::StreamExt;
use tokio::runtime::Runtime;
use xlai_core::{
    MediaSource, Metadata, TtsChunk, TtsDeliveryMode, TtsRequest, VoiceReferenceSample, VoiceSpec,
};
use xlai_qts_core::{QtsTtsConfig, QtsTtsModel};
use xlai_runtime::RuntimeBuilder;

fn require_model_dir() -> PathBuf {
    env::var("XLAI_QTS_MODEL_DIR")
        .expect("XLAI_QTS_MODEL_DIR")
        .into()
}

fn require_ref_audio_wav() -> Vec<u8> {
    let path: PathBuf = env::var("XLAI_QTS_REF_AUDIO_WAV")
        .expect("XLAI_QTS_REF_AUDIO_WAV")
        .into();
    let read = std::fs::read(&path);
    assert!(
        read.is_ok(),
        "failed to read reference WAV {}: {}",
        path.display(),
        read.as_ref()
            .err()
            .map_or_else(String::new, ToString::to_string)
    );
    read.unwrap_or_default()
}

fn ref_text() -> String {
    env::var("XLAI_QTS_REF_TEXT").unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../xlai-qts-core/testdata/sample1.txt")
            .canonicalize()
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .unwrap_or_else(|| "Reference transcript for QTS voice clone smoke tests.".to_owned())
    })
}

fn clone_request(
    input: &str,
    ref_wav: Vec<u8>,
    transcript: Option<String>,
    metadata: Metadata,
    delivery: TtsDeliveryMode,
) -> TtsRequest {
    TtsRequest {
        model: None,
        input: input.to_owned(),
        voice: VoiceSpec::Clone {
            references: vec![VoiceReferenceSample {
                audio: MediaSource::InlineData {
                    mime_type: "audio/wav".to_owned(),
                    data: ref_wav,
                },
                mime_type: Some("audio/wav".to_owned()),
                transcript,
                weight: None,
                metadata: Metadata::default(),
            }],
        },
        response_format: None,
        speed: None,
        instructions: None,
        delivery,
        metadata,
    }
}

fn assert_wav_response(response: xlai_runtime::TtsResponse) {
    assert!(response.mime_type.contains("wav"));
    assert!(
        matches!(response.audio, MediaSource::InlineData { .. }),
        "expected inline wav response"
    );
    let MediaSource::InlineData { data, .. } = response.audio else {
        return;
    };
    assert!(data.len() > 44, "expected non-empty WAV payload");
    let reader = hound::WavReader::new(std::io::Cursor::new(data)).expect("decode wav");
    assert_eq!(reader.spec().sample_rate, 24_000);
    assert!(reader.duration() > 0, "expected at least one audio sample");
}

#[test]
#[ignore = "requires XLAI_QTS_MODEL_DIR with qwen3-tts GGUF and vocoder ONNX"]
fn e2e_synthesize_via_runtime() {
    let dir = require_model_dir();
    let rt = Runtime::new().expect("runtime");
    rt.block_on(async {
        let model = QtsTtsModel::new(QtsTtsConfig::new(dir)).expect("load qts");
        let xlai = RuntimeBuilder::new()
            .with_tts_backend(model)
            .build()
            .expect("xlai runtime");
        let out = xlai
            .synthesize(TtsRequest {
                model: None,
                input: "hello".to_owned(),
                voice: VoiceSpec::Preset {
                    name: "default".to_owned(),
                },
                response_format: None,
                speed: None,
                instructions: None,
                delivery: Default::default(),
                metadata: Default::default(),
            })
            .await
            .expect("synthesize");
        assert_wav_response(out);
    });
}

#[test]
#[ignore = "requires XLAI_QTS_MODEL_DIR and XLAI_QTS_REF_AUDIO_WAV"]
fn e2e_synthesize_voice_clone_xvector_via_runtime() {
    let dir = require_model_dir();
    let ref_wav = require_ref_audio_wav();
    let rt = Runtime::new().expect("runtime");
    rt.block_on(async {
        let model = QtsTtsModel::new(QtsTtsConfig::new(dir)).expect("load qts");
        let xlai = RuntimeBuilder::new()
            .with_tts_backend(model)
            .build()
            .expect("xlai runtime");
        let out = xlai
            .synthesize(clone_request(
                "Say hello in the reference speaker's style.",
                ref_wav,
                None,
                Metadata::default(),
                TtsDeliveryMode::Unary,
            ))
            .await
            .expect("synthesize xvector clone");
        assert_wav_response(out);
    });
}

#[test]
#[ignore = "requires XLAI_QTS_MODEL_DIR, XLAI_QTS_REF_AUDIO_WAV, and reference-codec artifacts in the model dir"]
fn e2e_synthesize_voice_clone_icl_via_runtime() {
    let dir = require_model_dir();
    let ref_wav = require_ref_audio_wav();
    let transcript = ref_text();
    let rt = Runtime::new().expect("runtime");
    rt.block_on(async {
        let model = QtsTtsModel::new(QtsTtsConfig::new(dir)).expect("load qts");
        let xlai = RuntimeBuilder::new()
            .with_tts_backend(model)
            .build()
            .expect("xlai runtime");
        let out = xlai
            .synthesize(clone_request(
                "Repeat the phrase with the reference style.",
                ref_wav,
                Some(transcript),
                Metadata::default(),
                TtsDeliveryMode::Unary,
            ))
            .await
            .expect("synthesize icl clone");
        assert_wav_response(out);
    });
}

#[test]
#[ignore = "requires XLAI_QTS_MODEL_DIR and XLAI_QTS_REF_AUDIO_WAV"]
fn e2e_stream_voice_clone_xvector_via_runtime() {
    let dir = require_model_dir();
    let ref_wav = require_ref_audio_wav();
    let rt = Runtime::new().expect("runtime");
    rt.block_on(async {
        let model = QtsTtsModel::new(QtsTtsConfig::new(dir)).expect("load qts");
        let xlai = RuntimeBuilder::new()
            .with_tts_backend(model)
            .build()
            .expect("xlai runtime");
        let mut stream = xlai
            .stream_synthesize(clone_request(
                "Short streaming clone test.",
                ref_wav,
                None,
                Metadata::default(),
                TtsDeliveryMode::Stream,
            ))
            .expect("start stream");

        let mut saw_delta = false;
        let mut finished = None;
        while let Some(item) = stream.next().await {
            match item.expect("stream item") {
                TtsChunk::Started { mime_type, .. } => {
                    assert!(mime_type.contains("wav"));
                }
                TtsChunk::AudioDelta { data } => {
                    if !data.is_empty() {
                        saw_delta = true;
                    }
                }
                TtsChunk::Finished { response } => finished = Some(response),
            }
        }

        assert!(saw_delta, "expected at least one non-empty audio delta");
        assert_wav_response(finished.expect("finished response"));
    });
}

#[test]
#[ignore = "requires XLAI_QTS_MODEL_DIR and XLAI_QTS_REF_AUDIO_WAV"]
fn e2e_synthesize_voice_clone_xvector_override_via_metadata() {
    let dir = require_model_dir();
    let ref_wav = require_ref_audio_wav();
    let transcript = ref_text();
    let rt = Runtime::new().expect("runtime");
    rt.block_on(async {
        let model = QtsTtsModel::new(QtsTtsConfig::new(dir)).expect("load qts");
        let xlai = RuntimeBuilder::new()
            .with_tts_backend(model)
            .build()
            .expect("xlai runtime");
        let mut metadata = Metadata::default();
        metadata.insert(
            "xlai.qts.voice_clone_mode".to_owned(),
            serde_json::Value::String("xvector".to_owned()),
        );
        let out = xlai
            .synthesize(clone_request(
                "Use the reference speaker without ICL conditioning.",
                ref_wav,
                Some(transcript),
                metadata,
                TtsDeliveryMode::Unary,
            ))
            .await
            .expect("synthesize xvector override clone");
        assert_wav_response(out);
    });
}
