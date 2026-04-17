mod tracing_setup;

use std::fs;
use std::path::{Path, PathBuf};

use xlai_qts_core::{Qwen3TtsEngine, VoiceCloneMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_setup::init_logging();
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("failed to locate workspace root")?
        .to_path_buf();

    let model_dir = std::env::var_os("XLAI_QTS_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("fixtures/qts"));
    let ref_audio = workspace_root.join("fixtures/audio/transcription-sample.wav");
    let ref_text_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/sample1.txt");
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata");

    let engine = Qwen3TtsEngine::from_model_dir(&model_dir)?;
    let wav = fs::read(&ref_audio)?;
    let ref_text = fs::read_to_string(&ref_text_path)?;

    fs::create_dir_all(&out_dir)?;

    let xvector = engine.create_voice_clone_prompt(&wav, None, VoiceCloneMode::XVectorOnly)?;
    let xvector_path = out_dir.join("sample1.xvector.voice-clone-prompt.cbor");
    fs::write(&xvector_path, xvector.to_cbor_vec()?)?;

    let icl = engine.create_voice_clone_prompt(&wav, Some(ref_text.trim()), VoiceCloneMode::Icl)?;
    let icl_path = out_dir.join("sample1.icl.voice-clone-prompt.cbor");
    fs::write(&icl_path, icl.to_cbor_vec()?)?;

    println!("wrote {}", xvector_path.display());
    println!("wrote {}", icl_path.display());
    Ok(())
}
