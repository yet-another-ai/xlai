use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use xlai_qts_core::{Qwen3TtsEngine, SynthesizeRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    xlai_observability::init_logging();
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().ok_or("missing model dir")?);
    let text = args.next().ok_or("missing text")?;
    let output = PathBuf::from(args.next().ok_or("missing output wav path")?);

    let engine = Qwen3TtsEngine::from_model_dir(model_dir)?;
    let result = engine.synthesize(&SynthesizeRequest {
        text,
        max_audio_frames: 16,
        ..Default::default()
    })?;

    write_wav_i16(&output, result.sample_rate_hz, &result.pcm_f32)?;
    println!(
        "wrote {} samples across {} frames to {}",
        result.pcm_f32.len(),
        result.generated_frames,
        output.display()
    );
    Ok(())
}

fn write_wav_i16(
    path: &std::path::Path,
    sample_rate_hz: u32,
    pcm_f32: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    let pcm_i16 = pcm_f32
        .iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect::<Vec<_>>();

    let data_len = (pcm_i16.len() * std::mem::size_of::<i16>()) as u32;
    let riff_len = 36 + data_len;
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_len.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&sample_rate_hz.to_le_bytes())?;
    let byte_rate = sample_rate_hz * 2;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&2u16.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_len.to_le_bytes())?;
    for sample in pcm_i16 {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}
