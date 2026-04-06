mod tracing_setup;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use xlai_qts_core::{PrefillConditioning, Qwen3TtsEngine, SynthesizeRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_setup::init_logging();
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().ok_or("missing model dir")?);
    let text = args.next().ok_or("missing text")?;
    let output = PathBuf::from(args.next().ok_or("missing output path")?);
    let temperature = args
        .next()
        .map(|value| value.parse::<f32>())
        .transpose()?
        .unwrap_or(0.9);
    let top_k = args
        .next()
        .map(|value| value.parse::<i32>())
        .transpose()?
        .unwrap_or(50);

    let engine = Qwen3TtsEngine::from_model_dir(model_dir)?;
    let req = SynthesizeRequest {
        text,
        max_audio_frames: 16,
        temperature,
        top_k,
        ..Default::default()
    };

    let tokens = engine.encode_for_tts(&req.text);
    let zero_speaker = vec![0.0f32; engine.transformer().config().hidden_size as usize];
    let prepared_inputs = engine.transformer().build_prefill_inputs(
        PrefillConditioning {
            text_tokens: &tokens,
            speaker_embd: Some(&zero_speaker),
            ref_codebook_0: &[],
            language_id: req.language_id,
        },
        req.thread_count,
    )?;
    let rollout = engine.transformer().rollout_codec_frames_recompute(
        &prepared_inputs.prefill_embd,
        &prepared_inputs.trailing_text_hidden,
        &prepared_inputs.tts_pad_embed,
        &[],
        req.thread_count,
        req.max_audio_frames,
        req.repetition_penalty,
        req.temperature,
        req.top_k,
        req.top_p,
    )?;

    let mut writer = BufWriter::new(File::create(&output)?);
    for frame in &rollout.frames {
        for &token in &frame.codebook_tokens {
            writer.write_all(&(token as i64).to_le_bytes())?;
        }
    }
    writer.flush()?;
    println!(
        "wrote {} frames / {} codes to {}",
        rollout.frames.len(),
        rollout
            .frames
            .iter()
            .map(|f| f.codebook_tokens.len())
            .sum::<usize>(),
        output.display()
    );
    Ok(())
}
