use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use xlai_qts_core::Qwen3TtsEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().ok_or("missing model dir")?);
    let text = args.next().ok_or("missing text")?;
    let output = PathBuf::from(args.next().ok_or("missing output path")?);

    let engine = Qwen3TtsEngine::from_model_dir(model_dir)?;
    let tokens = engine.encode_for_tts(&text);
    let mut writer = BufWriter::new(File::create(&output)?);
    for token in tokens {
        writer.write_all(&(token as i64).to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}
