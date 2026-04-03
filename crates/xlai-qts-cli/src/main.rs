#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]

use std::env;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use hound::{SampleFormat, WavSpec, WavWriter};
use serde_json::json;
use tokio::runtime::Runtime;
use xlai_backend_qts::{QtsTtsConfig, QtsTtsModel};
use xlai_core::{MediaSource, Metadata, TtsRequest, TtsResponse, VoiceSpec};
use xlai_qts_core::{Qwen3TtsEngine, SynthesisStageTimings, VoiceCloneMode, VoiceClonePromptV2};
use xlai_runtime::RuntimeBuilder;

mod cli_support;
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tui;

use crate::cli_support::{CommonSynthesisArgs, load_engine, parse_value_arg};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("synthesize") => run_synthesize(args.collect()),
        Some("profile") => run_profile(args.collect()),
        Some("tui") => tui::run(args.collect()),
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn run_synthesize(args: Vec<String>) -> Result<()> {
    let mut common = CommonSynthesisArgs::new()?;

    let mut idx = 0;
    while idx < args.len() {
        if common.parse_flag(&args, &mut idx)? {
            continue;
        }
        bail!("unknown synthesize argument: {}", args[idx]);
    }

    common.validate_conditioning()?;
    let text = common.require_text()?;
    let out_path = common.require_out_path()?;

    if common.voice_clone_prompt.is_some() || common.ref_audio.is_some() {
        let engine = load_engine(&common.model_dir, &common.runtime_backends)?;
        let request = common.build_request(text)?;
        let conditioning = load_synthesis_conditioning(&engine, &common)?;
        let result = if let LoadedConditioning::VoiceClonePrompt(prompt) = &conditioning {
            engine.synthesize_with_voice_clone_prompt(&request, prompt)?
        } else {
            engine.synthesize(&request)?
        };
        write_wav_f32(&out_path, result.sample_rate_hz, &result.pcm_f32)?;
        eprintln!(
            "wrote synthesis: path={} sample_rate={} frames={} samples={}",
            out_path.display(),
            result.sample_rate_hz,
            result.generated_frames,
            result.pcm_f32.len()
        );
        return Ok(());
    }

    common.runtime_backends.apply_env_overrides();
    let tokio_rt = Runtime::new().context("tokio runtime")?;
    let model = QtsTtsModel::new(QtsTtsConfig::new(common.model_dir.clone()))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let xlai_rt = RuntimeBuilder::new()
        .with_tts_backend(model)
        .build()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let tts_req = tts_request_from_common(&common, text);
    let response = tokio_rt
        .block_on(xlai_rt.synthesize(tts_req))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let wav_bytes = inline_wav_bytes(&response).context("decode TTS response")?;
    fs::write(&out_path, &wav_bytes).with_context(|| format!("write {}", out_path.display()))?;
    eprintln!(
        "wrote synthesis (xlai-runtime + xlai-backend-qts): path={} bytes={}",
        out_path.display(),
        wav_bytes.len()
    );
    Ok(())
}

fn run_profile(args: Vec<String>) -> Result<()> {
    if args.iter().any(|a| matches!(a.as_str(), "--help" | "-h")) {
        eprintln!(
            "xlai-qts-cli profile — print per-stage synthesis timings (wall clock)\n\n\
             usage:\n  profile --text TEXT [--model-dir DIR] [--runs N] [--out OUT.wav] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language-id N] [--vocoder-threads N] [--chunk-size N] [--talker-kv-mode f16|turboquant] [--voice-clone-prompt PATH | --ref-audio WAV [--ref-text STR] [--voice-clone-mode xvector|icl]] [--backend auto|cpu|metal|vulkan] [--backend-fallback LIST] [--vocoder-ep auto|cpu|cuda|nvrtx|tensorrt|coreml|directml] [--vocoder-ep-fallback LIST]\n\n\
             CLI flags override environment variables.\n\
             Default transformer auto chain: Apple = metal,vulkan,cpu ; others = vulkan,cpu.\n\
             Default vocoder auto chain: Apple = coreml,cpu ; Windows = cuda,nvrtx,tensorrt,directml,cpu ; Linux/others = cuda,nvrtx,tensorrt,cpu.\n\n\
             If --frames is omitted, the CLI derives a text-length-based max frame budget.\n\
             --runs N averages stage times over N full synthesize passes (default 1).\n\
             --out writes WAV from the first pass only when --runs > 1."
        );
        return Ok(());
    }

    let mut common = CommonSynthesisArgs::new()?;
    let mut runs = 1usize;

    let mut idx = 0;
    while idx < args.len() {
        if common.parse_flag(&args, &mut idx)? {
            continue;
        }
        match args[idx].as_str() {
            "--runs" => {
                runs = parse_value_arg(&args, &mut idx, "--runs")?;
            }
            other => bail!("unknown profile argument: {other}"),
        }
    }

    if runs == 0 {
        bail!("--runs must be >= 1");
    }

    common.validate_conditioning()?;
    let text = common.require_text()?;
    let engine = load_engine(&common.model_dir, &common.runtime_backends)?;
    let request = common.build_request(text)?;
    let conditioning = load_synthesis_conditioning(&engine, &common)?;

    let mut samples = Vec::with_capacity(runs);
    let mut first_result = None;

    for run_idx in 0..runs {
        let (result, timings) = match &conditioning {
            LoadedConditioning::VoiceClonePrompt(prompt) => {
                engine.synthesize_with_voice_clone_prompt_profile(&request, prompt)?
            }
            LoadedConditioning::None => engine.synthesize_with_profile(&request)?,
        };
        if run_idx == 0 {
            first_result = Some(result);
        }
        samples.push(timings);
    }

    if let Some(path) = common.out_path {
        let result = first_result.context("internal: missing first synthesis result")?;
        write_wav_f32(&path, result.sample_rate_hz, &result.pcm_f32)?;
        eprintln!(
            "wrote profile run #1 WAV: path={} sample_rate={} frames={}",
            path.display(),
            result.sample_rate_hz,
            result.generated_frames
        );
    }

    let summary = if runs == 1 {
        samples[0].format_table()
    } else {
        let avg = SynthesisStageTimings::average(&samples).expect("non-empty samples");
        format!("averaged over {runs} runs\n{}", avg.format_table())
    };
    eprint!("{summary}");

    Ok(())
}

enum LoadedConditioning {
    None,
    VoiceClonePrompt(VoiceClonePromptV2),
}

fn load_synthesis_conditioning(
    engine: &Qwen3TtsEngine,
    args: &CommonSynthesisArgs,
) -> Result<LoadedConditioning> {
    if let Some(path) = &args.voice_clone_prompt {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let prompt = engine.decode_voice_clone_prompt(&bytes)?;
        return Ok(LoadedConditioning::VoiceClonePrompt(prompt));
    }
    if let Some(path) = &args.ref_audio {
        let wav = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let mode = args.voice_clone_mode.unwrap_or_else(|| {
            if args
                .ref_text
                .as_ref()
                .map(|t| !t.trim().is_empty())
                .unwrap_or(false)
            {
                VoiceCloneMode::Icl
            } else {
                VoiceCloneMode::XVectorOnly
            }
        });
        let prompt = engine
            .create_voice_clone_prompt(&wav, args.ref_text.as_deref(), mode)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        return Ok(LoadedConditioning::VoiceClonePrompt(prompt));
    }
    Ok(LoadedConditioning::None)
}

fn tts_request_from_common(common: &CommonSynthesisArgs, text: String) -> TtsRequest {
    let mut metadata = Metadata::default();
    metadata.insert(
        "xlai.qts.thread_count".to_owned(),
        json!(common.thread_count),
    );
    if let Some(frames) = common.max_audio_frames {
        metadata.insert("xlai.qts.max_audio_frames".to_owned(), json!(frames));
    }
    metadata.insert("xlai.qts.temperature".to_owned(), json!(common.temperature));
    metadata.insert("xlai.qts.top_p".to_owned(), json!(common.top_p));
    metadata.insert("xlai.qts.top_k".to_owned(), json!(common.top_k));
    metadata.insert(
        "xlai.qts.repetition_penalty".to_owned(),
        json!(common.repetition_penalty),
    );
    metadata.insert("xlai.qts.language_id".to_owned(), json!(common.language_id));
    metadata.insert(
        "xlai.qts.vocoder_thread_count".to_owned(),
        json!(common.vocoder_thread_count),
    );
    metadata.insert(
        "xlai.qts.vocoder_chunk_size".to_owned(),
        json!(common.vocoder_chunk_size),
    );
    metadata.insert(
        "xlai.qts.talker_kv_mode".to_owned(),
        json!(common.talker_kv_mode.as_str()),
    );

    TtsRequest {
        model: None,
        input: text,
        voice: VoiceSpec::Preset {
            name: "default".to_owned(),
        },
        response_format: None,
        speed: None,
        instructions: None,
        delivery: Default::default(),
        metadata,
    }
}

fn inline_wav_bytes(response: &TtsResponse) -> Result<Vec<u8>> {
    match &response.audio {
        MediaSource::InlineData { data, .. } => Ok(data.clone()),
        MediaSource::Url { .. } => bail!("unexpected URL audio in QTS response"),
    }
}

fn write_wav_f32(path: &Path, sample_rate_hz: u32, pcm_f32: &[f32]) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("failed to create {}", path.display()))?;
    for sample in pcm_f32.iter().copied() {
        let clamped = sample.clamp(-1.0, 1.0);
        writer
            .write_sample((clamped * i16::MAX as f32) as i16)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("failed to finalize {}", path.display()))?;
    Ok(())
}

fn print_usage() {
    eprintln!(
        "usage:\n  cargo run -p xlai-qts-cli -- synthesize --text TEXT --out OUT.wav [--model-dir DIR] [--voice-clone-prompt prompt.cbor | --ref-audio ref.wav [--ref-text STR] [--voice-clone-mode xvector|icl]] ...\n  cargo run -p xlai-qts-cli -- profile ...\n  cargo run -p xlai-qts-cli -- tui ...\n\nWithout voice clone flags, synthesize uses xlai-runtime + xlai-backend-qts.\nWith --voice-clone-prompt or --ref-audio, the direct Qwen3TtsEngine path is used.\n\nEnv: QWEN3_TTS_BACKEND / QWEN3_TTS_VOCODER_EP / QWEN3_TTS_TALKER_KV_MODE (see docs)."
    );
}
