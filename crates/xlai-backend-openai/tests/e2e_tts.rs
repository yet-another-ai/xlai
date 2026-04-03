use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures_util::StreamExt;
use xlai_backend_openai::OpenAiConfig;
use xlai_core::{
    ErrorKind, RuntimeCapability, TtsChunk, TtsDeliveryMode, TtsRequest, VoiceSpec, XlaiError,
};
use xlai_runtime::RuntimeBuilder;

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and OPENAI_TTS_MODEL in the protected e2e environment"]
async fn openai_tts_unary_smoke_test() -> Result<(), XlaiError> {
    let _ = dotenvy::dotenv();

    let runtime = build_tts_runtime()?;

    let response = runtime
        .synthesize(TtsRequest {
            model: None,
            input: "Say only the word test.".to_owned(),
            voice: VoiceSpec::Preset {
                name: "alloy".to_owned(),
            },
            response_format: Some(xlai_core::TtsAudioFormat::Mp3),
            speed: Some(1.0),
            instructions: None,
            delivery: TtsDeliveryMode::Unary,
            metadata: Default::default(),
        })
        .await?;

    let xlai_core::MediaSource::InlineData { data_base64, .. } = response.audio else {
        return Err(XlaiError::new(
            ErrorKind::Provider,
            "expected inline audio from unary TTS",
        ));
    };

    let decoded = STANDARD.decode(data_base64.trim()).map_err(|error| {
        XlaiError::new(
            ErrorKind::Provider,
            format!("tts response must be valid base64: {error}"),
        )
    })?;

    assert!(
        decoded.len() > 16,
        "unary TTS should return non-trivial audio bytes",
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and OPENAI_TTS_MODEL (use a model that supports speech SSE, e.g. gpt-4o-mini-tts, not tts-1)"]
async fn openai_tts_stream_smoke_test() -> Result<(), XlaiError> {
    let _ = dotenvy::dotenv();

    let runtime = build_tts_runtime()?;

    let mut stream = runtime
        .stream_synthesize(TtsRequest {
            model: None,
            input: "Short stream test.".to_owned(),
            voice: VoiceSpec::Preset {
                name: "alloy".to_owned(),
            },
            response_format: Some(xlai_core::TtsAudioFormat::Mp3),
            speed: Some(1.0),
            instructions: None,
            delivery: TtsDeliveryMode::Stream,
            metadata: Default::default(),
        })?
        .boxed();

    let mut saw_delta = false;
    let mut finished = false;

    while let Some(item) = stream.next().await {
        match item? {
            TtsChunk::Started { .. } => {}
            TtsChunk::AudioDelta { data_base64 } => {
                if !data_base64.is_empty() {
                    saw_delta = true;
                }
            }
            TtsChunk::Finished { response } => {
                finished = true;
                let xlai_core::MediaSource::InlineData { data_base64, .. } = response.audio else {
                    return Err(XlaiError::new(
                        ErrorKind::Provider,
                        "expected inline audio in finished TTS chunk",
                    ));
                };
                let decoded = STANDARD.decode(data_base64.trim()).map_err(|error| {
                    XlaiError::new(
                        ErrorKind::Provider,
                        format!("assembled TTS must be valid base64: {error}"),
                    )
                })?;
                assert!(
                    decoded.len() > 16,
                    "streamed TTS should assemble non-trivial audio",
                );
            }
        }
    }

    assert!(saw_delta, "expected at least one non-empty audio delta");
    assert!(finished, "expected a Finished chunk");

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and OPENAI_TTS_MODEL in the protected e2e environment"]
async fn openai_tts_runtime_reports_capability() -> Result<(), XlaiError> {
    let _ = dotenvy::dotenv();

    let runtime = build_tts_runtime()?;

    assert!(
        runtime.has_capability(RuntimeCapability::Tts),
        "runtime should expose TTS capability when backend is configured",
    );

    Ok(())
}

fn build_tts_runtime() -> Result<xlai_runtime::XlaiRuntime, XlaiError> {
    let api_key = require_env("OPENAI_API_KEY")?;
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());
    let chat_model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_owned());
    let tts_model = require_env("OPENAI_TTS_MODEL")?;

    RuntimeBuilder::new()
        .with_tts_backend(
            OpenAiConfig::new(base_url, api_key, chat_model).with_tts_model(tts_model),
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
