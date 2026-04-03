#![allow(clippy::expect_used)]

//! Ignored smoke test: set `XLAI_QTS_MODEL_DIR` to a directory with GGUF + vocoder ONNX.
//!
//! ```sh
//! export XLAI_QTS_MODEL_DIR=/path/to/models
//! cargo test -p xlai-backend-qts e2e_model_dir -- --ignored --nocapture
//! ```

use std::env;
use std::path::PathBuf;

use tokio::runtime::Runtime;
use xlai_backend_qts::{QtsTtsConfig, QtsTtsModel};
use xlai_core::{TtsRequest, VoiceSpec};
use xlai_runtime::RuntimeBuilder;

#[test]
#[ignore = "requires XLAI_QTS_MODEL_DIR with qwen3-tts GGUF and vocoder ONNX"]
fn e2e_synthesize_via_runtime() {
    let dir: PathBuf = env::var("XLAI_QTS_MODEL_DIR")
        .expect("XLAI_QTS_MODEL_DIR")
        .into();
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
        assert!(out.mime_type.contains("wav"));
    });
}
