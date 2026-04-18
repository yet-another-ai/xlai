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

    let engine = Qwen3TtsEngine::from_model_dir(model_dir)?;
    let req = SynthesizeRequest {
        text,
        max_audio_frames: 1,
        temperature: 0.0,
        top_k: 0,
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

    let frame = rollout.frames.first().ok_or("no frames generated")?;
    let mut writer = BufWriter::new(File::create(&output)?);
    for value in &frame.hidden_state {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    println!(
        "cb0={} hidden_len={}",
        frame.codebook_0_token,
        frame.hidden_state.len()
    );
    println!("codes={:?}", frame.codebook_tokens);
    Ok(())
}
