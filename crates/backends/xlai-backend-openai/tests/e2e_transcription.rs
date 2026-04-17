use xlai_backend_openai::OpenAiConfig;
use xlai_core::{ErrorKind, MediaSource, RuntimeCapability, TranscriptionRequest, XlaiError};
use xlai_runtime::RuntimeBuilder;

const SAMPLE_WAV_BYTES: &[u8] = include_bytes!("../../../fixtures/audio/transcription-sample.wav");

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and OPENAI_TRANSCRIPTION_MODEL in the protected e2e environment"]
async fn openai_transcription_smoke_test() -> Result<(), XlaiError> {
    let _ = dotenvy::dotenv();

    let runtime = build_transcription_runtime()?;

    let response = runtime
        .transcribe(TranscriptionRequest {
            model: None,
            audio: MediaSource::InlineData {
                mime_type: "audio/wav".to_owned(),
                data: SAMPLE_WAV_BYTES.to_vec(),
            },
            mime_type: Some("audio/wav".to_owned()),
            filename: Some("transcription-sample.wav".to_owned()),
            language: Some("en".to_owned()),
            prompt: Some("Return a transcription for this audio sample.".to_owned()),
            temperature: Some(0.0),
            metadata: Default::default(),
        })
        .await?;

    assert!(
        !response.text.trim().is_empty(),
        "transcription response should not be empty",
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and OPENAI_TRANSCRIPTION_MODEL in the protected e2e environment"]
async fn openai_transcription_runtime_reports_capability() -> Result<(), XlaiError> {
    let _ = dotenvy::dotenv();

    let runtime = build_transcription_runtime()?;

    assert!(
        runtime.has_capability(RuntimeCapability::Transcription),
        "runtime should expose transcription capability when ASR backend is configured",
    );

    let response = runtime
        .transcribe(TranscriptionRequest {
            model: None,
            audio: MediaSource::InlineData {
                mime_type: "audio/wav".to_owned(),
                data: SAMPLE_WAV_BYTES.to_vec(),
            },
            mime_type: Some("audio/wav".to_owned()),
            filename: Some("transcription-sample.wav".to_owned()),
            language: None,
            prompt: None,
            temperature: Some(0.0),
            metadata: Default::default(),
        })
        .await?;

    assert!(
        !response.text.trim().is_empty(),
        "transcription capability check should still return text",
    );

    Ok(())
}

fn build_transcription_runtime() -> Result<xlai_runtime::XlaiRuntime, XlaiError> {
    let api_key = require_env("OPENAI_API_KEY")?;
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());
    let chat_model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_owned());
    let transcription_model = require_env("OPENAI_TRANSCRIPTION_MODEL")?;

    RuntimeBuilder::new()
        .with_transcription_backend(
            OpenAiConfig::new(base_url, api_key, chat_model)
                .with_transcription_model(transcription_model),
        )
        .build()
}

fn require_env(name: &str) -> Result<String, XlaiError> {
    std::env::var(name).map_err(|error| {
        XlaiError::new(
            ErrorKind::Configuration,
            format!("{name} must be set for e2e tests: {error}"),
        )
    })
}
