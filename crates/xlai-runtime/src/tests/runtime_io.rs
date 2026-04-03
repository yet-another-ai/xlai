//! Runtime tests: filesystem, transcription, TTS.
use std::sync::{Arc, Mutex};

use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::StreamExt;
use xlai_core::{
    ChatResponse, FinishReason, FsEntryKind, FsPath, MediaSource, RuntimeCapability,
    TranscriptionRequest, TranscriptionResponse, TtsChunk, TtsRequest, TtsResponse, VoiceSpec,
    XlaiError,
};

use super::common::*;
use crate::{MemoryFileSystem, RuntimeBuilder};

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_file_system_helpers_use_configured_backend() -> Result<(), XlaiError> {
    let runtime = RuntimeBuilder::new()
        .with_file_system(Arc::new(MemoryFileSystem::new()))
        .build()?;

    assert!(runtime.has_capability(RuntimeCapability::FileSystem));

    runtime.create_dir_all(&FsPath::from("/docs")).await?;
    runtime
        .write_file(&FsPath::from("/docs/readme.md"), b"runtime".to_vec())
        .await?;

    let bytes = runtime.read_file(&FsPath::from("/docs/readme.md")).await?;
    assert_eq!(bytes, b"runtime".to_vec());

    let exists = runtime
        .path_exists(&FsPath::from("/docs/readme.md"))
        .await?;
    assert!(exists);

    let entries = runtime.list_directory(&FsPath::from("/docs")).await?;
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, FsPath::from("/docs/readme.md"));
    assert_eq!(entries[0].kind, FsEntryKind::File);

    runtime
        .delete_path(&FsPath::from("/docs/readme.md"))
        .await?;
    let exists = runtime
        .path_exists(&FsPath::from("/docs/readme.md"))
        .await?;
    assert!(!exists);

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_file_system_helpers_error_without_backend() -> Result<(), XlaiError> {
    let runtime = RuntimeBuilder::new()
        .with_chat_model(Arc::new(RecordingChatModel::new(
            Arc::new(Mutex::new(Vec::new())),
            vec![ChatResponse {
                message: assistant_message("unused"),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            }],
        )))
        .build()?;

    let error = match runtime.read_file(&FsPath::from("/docs/readme.md")).await {
        Ok(_) => {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Validation,
                "expected missing filesystem dependency error",
            ));
        }
        Err(error) => error,
    };
    assert_eq!(error.kind, xlai_core::ErrorKind::Unsupported);
    assert!(
        error.message.contains("file system"),
        "expected error message to mention file system"
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_transcribe_uses_configured_backend() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let runtime = RuntimeBuilder::new()
        .with_transcription_model(Arc::new(RecordingTranscriptionModel::new(
            requests.clone(),
            vec![TranscriptionResponse {
                text: "hello world".to_owned(),
                metadata: empty_metadata(),
            }],
        )))
        .build()?;

    assert!(runtime.has_capability(RuntimeCapability::Transcription));

    let decoded = STANDARD
        .decode("UklGRg==")
        .map_err(|error| XlaiError::new(xlai_core::ErrorKind::Validation, error.to_string()))?;
    let response = runtime
        .transcribe(TranscriptionRequest {
            model: Some("gpt-4o-mini-transcribe".to_owned()),
            audio: MediaSource::InlineData {
                mime_type: "audio/wav".to_owned(),
                data: decoded,
            },
            mime_type: Some("audio/wav".to_owned()),
            filename: Some("sample.wav".to_owned()),
            language: Some("en".to_owned()),
            prompt: Some("Keep punctuation.".to_owned()),
            temperature: Some(0.0),
            metadata: empty_metadata(),
        })
        .await?;

    assert_eq!(response.text, "hello world");
    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].filename.as_deref(), Some("sample.wav"));

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_transcribe_errors_without_backend() -> Result<(), XlaiError> {
    let runtime = RuntimeBuilder::new()
        .with_chat_model(Arc::new(RecordingChatModel::new(
            Arc::new(Mutex::new(Vec::new())),
            vec![ChatResponse {
                message: assistant_message("unused"),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            }],
        )))
        .build()?;

    let decoded = STANDARD
        .decode("UklGRg==")
        .map_err(|error| XlaiError::new(xlai_core::ErrorKind::Validation, error.to_string()))?;
    let error = match runtime
        .transcribe(TranscriptionRequest {
            model: None,
            audio: MediaSource::InlineData {
                mime_type: "audio/wav".to_owned(),
                data: decoded,
            },
            mime_type: Some("audio/wav".to_owned()),
            filename: None,
            language: None,
            prompt: None,
            temperature: None,
            metadata: empty_metadata(),
        })
        .await
    {
        Ok(_) => {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Validation,
                "expected missing transcription dependency error",
            ));
        }
        Err(error) => error,
    };
    assert_eq!(error.kind, xlai_core::ErrorKind::Unsupported);
    assert!(
        error.message.contains("transcription model"),
        "expected error message to mention transcription model"
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_synthesize_uses_configured_backend() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let decoded = STANDARD
        .decode("AAAA")
        .map_err(|error| XlaiError::new(xlai_core::ErrorKind::Validation, error.to_string()))?;
    let runtime = RuntimeBuilder::new()
        .with_tts_model(Arc::new(RecordingTtsModel::new(
            requests.clone(),
            vec![TtsResponse {
                audio: MediaSource::InlineData {
                    mime_type: "audio/mpeg".to_owned(),
                    data: decoded,
                },
                mime_type: "audio/mpeg".to_owned(),
                metadata: empty_metadata(),
            }],
        )))
        .build()?;

    assert!(runtime.has_capability(RuntimeCapability::Tts));

    let response = runtime
        .synthesize(TtsRequest {
            model: Some("tts-1".to_owned()),
            input: "hello".to_owned(),
            voice: VoiceSpec::Preset {
                name: "alloy".to_owned(),
            },
            response_format: None,
            speed: None,
            instructions: None,
            delivery: xlai_core::TtsDeliveryMode::Unary,
            metadata: empty_metadata(),
        })
        .await?;

    assert_eq!(response.mime_type, "audio/mpeg");
    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].input, "hello");

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_synthesize_errors_without_backend() -> Result<(), XlaiError> {
    let runtime = RuntimeBuilder::new()
        .with_chat_model(Arc::new(RecordingChatModel::new(
            Arc::new(Mutex::new(Vec::new())),
            vec![ChatResponse {
                message: assistant_message("unused"),
                tool_calls: Vec::new(),
                usage: None,
                finish_reason: FinishReason::Completed,
                metadata: empty_metadata(),
            }],
        )))
        .build()?;

    let error = match runtime
        .synthesize(TtsRequest {
            model: None,
            input: "hi".to_owned(),
            voice: VoiceSpec::Preset {
                name: "alloy".to_owned(),
            },
            response_format: None,
            speed: None,
            instructions: None,
            delivery: xlai_core::TtsDeliveryMode::Unary,
            metadata: empty_metadata(),
        })
        .await
    {
        Ok(_) => {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Validation,
                "expected missing tts dependency error",
            ));
        }
        Err(error) => error,
    };
    assert_eq!(error.kind, xlai_core::ErrorKind::Unsupported);
    assert!(
        error.message.contains("tts model"),
        "expected error message to mention tts model"
    );

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_stream_synthesize_uses_trait_default_for_unary_only_model() -> Result<(), XlaiError>
{
    let requests = Arc::new(Mutex::new(Vec::new()));
    let decoded = STANDARD
        .decode("QkVBVg==")
        .map_err(|error| XlaiError::new(xlai_core::ErrorKind::Validation, error.to_string()))?;
    let runtime = RuntimeBuilder::new()
        .with_tts_model(Arc::new(RecordingTtsModel::new(
            requests.clone(),
            vec![TtsResponse {
                audio: MediaSource::InlineData {
                    mime_type: "audio/mpeg".to_owned(),
                    data: decoded,
                },
                mime_type: "audio/mpeg".to_owned(),
                metadata: empty_metadata(),
            }],
        )))
        .build()?;

    let mut stream = runtime
        .stream_synthesize(TtsRequest {
            model: None,
            input: "stream test".to_owned(),
            voice: VoiceSpec::Preset {
                name: "nova".to_owned(),
            },
            response_format: None,
            speed: None,
            instructions: None,
            delivery: xlai_core::TtsDeliveryMode::Stream,
            metadata: empty_metadata(),
        })?
        .boxed();

    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item?);
    }

    assert!(
        matches!(chunks.first(), Some(TtsChunk::Started { .. })),
        "expected Started chunk"
    );
    assert!(
        chunks
            .iter()
            .any(|c| matches!(c, TtsChunk::AudioDelta { .. })),
        "expected at least one AudioDelta"
    );
    assert!(
        matches!(chunks.last(), Some(TtsChunk::Finished { .. })),
        "expected Finished chunk"
    );

    Ok(())
}
