mod tracing_setup;

use std::path::PathBuf;
use std::time::Instant;

use xlai_qts_core::{Qwen3TtsEngine, SynthesizeRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_setup::init_logging();
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().ok_or("missing model dir")?);
    let text = args.next().ok_or("missing text")?;
    let thread_count = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(4);
    let max_audio_frames = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(16);
    let temperature = args
        .next()
        .map(|value| value.parse::<f32>())
        .transpose()?
        .unwrap_or(0.0);
    let top_k = args
        .next()
        .map(|value| value.parse::<i32>())
        .transpose()?
        .unwrap_or(0);
    let top_p = args
        .next()
        .map(|value| value.parse::<f32>())
        .transpose()?
        .unwrap_or(1.0);

    let engine = Qwen3TtsEngine::from_model_dir(model_dir)?;
    let started = Instant::now();
    let result = engine.synthesize(&SynthesizeRequest {
        text,
        thread_count,
        max_audio_frames,
        temperature,
        top_k,
        top_p,
        ..Default::default()
    })?;
    let elapsed = started.elapsed().as_secs_f64();
    let audio_seconds = result.pcm_f32.len() as f64 / result.sample_rate_hz as f64;
    let rtf = if audio_seconds > 0.0 {
        elapsed / audio_seconds
    } else {
        0.0
    };
    let x_realtime = if elapsed > 0.0 {
        audio_seconds / elapsed
    } else {
        0.0
    };

    println!("elapsed_s={elapsed:.6}");
    println!("audio_s={audio_seconds:.6}");
    println!("sample_rate_hz={}", result.sample_rate_hz);
    println!("generated_frames={}", result.generated_frames);
    println!("samples={}", result.pcm_f32.len());
    println!("rtf={rtf:.6}");
    println!("x_realtime={x_realtime:.6}");
    Ok(())
}
