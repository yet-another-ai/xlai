//! OpenAI-compatible `POST /audio/speech` mapping.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};
use xlai_core::{
    ErrorKind, MediaSource, Metadata, TtsAudioFormat, TtsRequest, TtsResponse,
    VoiceSpec, XlaiError,
};

use crate::OpenAiConfig;

pub(crate) fn openai_voice_json(voice: &VoiceSpec) -> Result<Value, XlaiError> {
    match voice {
        VoiceSpec::Preset { name } => Ok(Value::String(name.clone())),
        VoiceSpec::ProviderRef { id, provider } => {
            if let Some(p) = provider
                && !p.is_empty()
                && !p.eq_ignore_ascii_case("openai")
            {
                return Err(XlaiError::new(
                    ErrorKind::Unsupported,
                    format!("openai-compatible TTS ignores non-openai voice provider: {p}"),
                ));
            }
            Ok(json!({ "id": id }))
        }
        VoiceSpec::Clone { .. } => Err(XlaiError::new(
            ErrorKind::Unsupported,
            "openai-compatible TTS does not support VoiceSpec::Clone",
        )),
    }
}

pub(crate) fn response_format_str(format: TtsAudioFormat) -> &'static str {
    match format {
        TtsAudioFormat::Mp3 => "mp3",
        TtsAudioFormat::Opus => "opus",
        TtsAudioFormat::Aac => "aac",
        TtsAudioFormat::Flac => "flac",
        TtsAudioFormat::Wav => "wav",
        TtsAudioFormat::Pcm => "pcm",
    }
}

pub(crate) fn mime_for_tts_format(format: TtsAudioFormat) -> &'static str {
    match format {
        TtsAudioFormat::Mp3 => "audio/mpeg",
        TtsAudioFormat::Opus => "audio/opus",
        TtsAudioFormat::Aac => "audio/aac",
        TtsAudioFormat::Flac => "audio/flac",
        TtsAudioFormat::Wav => "audio/wav",
        TtsAudioFormat::Pcm => "audio/pcm",
    }
}

pub(crate) fn resolved_tts_model(config: &OpenAiConfig, request: &TtsRequest) -> String {
    request
        .model
        .clone()
        .or_else(|| config.tts_model.clone())
        .unwrap_or_else(|| config.model.clone())
}

pub(crate) fn openai_model_supports_speech_sse(model: &str) -> bool {
    !matches!(model, "tts-1" | "tts-1-hd")
}

pub(crate) fn build_speech_json_body(
    config: &OpenAiConfig,
    request: &TtsRequest,
    stream_format: Option<&'static str>,
) -> Result<Value, XlaiError> {
    let model = resolved_tts_model(config, request);
    let voice = openai_voice_json(&request.voice)?;
    let response_format = request.response_format.unwrap_or(TtsAudioFormat::Mp3);

    let mut map = Map::new();
    map.insert("model".to_owned(), Value::String(model));
    map.insert("input".to_owned(), Value::String(request.input.clone()));
    map.insert("voice".to_owned(), voice);
    map.insert(
        "response_format".to_owned(),
        Value::String(response_format_str(response_format).to_owned()),
    );

    if let Some(speed) = request.speed {
        map.insert("speed".to_owned(), json!(speed));
    }

    if let Some(ref instructions) = request.instructions
        && !instructions.is_empty()
    {
        map.insert(
            "instructions".to_owned(),
            Value::String(instructions.clone()),
        );
    }

    if let Some(sf) = stream_format {
        map.insert("stream_format".to_owned(), Value::String(sf.to_owned()));
    }

    Ok(Value::Object(map))
}

pub(crate) fn tts_response_from_unary_bytes(
    bytes: Vec<u8>,
    response_format: TtsAudioFormat,
    extra_metadata: Metadata,
) -> TtsResponse {
    let mime = mime_for_tts_format(response_format);
    TtsResponse {
        audio: MediaSource::InlineData {
            mime_type: mime.to_owned(),
            data_base64: STANDARD.encode(bytes),
        },
        mime_type: mime.to_owned(),
        metadata: extra_metadata,
    }
}

/// Parse one SSE `data:` payload from `/audio/speech` when `stream_format` is `sse`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParsedSpeechSse {
    DeltaBase64(String),
    Done,
    Ignored,
}

pub(crate) fn parse_speech_sse_data(data: &str) -> Result<ParsedSpeechSse, XlaiError> {
    let v: Value = serde_json::from_str(data).map_err(|error| {
        XlaiError::new(
            ErrorKind::Provider,
            format!("failed to parse speech SSE JSON: {error}"),
        )
    })?;

    let Some(t) = v.get("type").and_then(|x| x.as_str()) else {
        return Ok(ParsedSpeechSse::Ignored);
    };

    match t {
        "speech.audio.delta" => {
            let b64 = v
                .get("audio")
                .and_then(|x| x.as_str())
                .or_else(|| v.get("delta").and_then(|x| x.as_str()))
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            Ok(match b64 {
                Some(s) => ParsedSpeechSse::DeltaBase64(s),
                None => ParsedSpeechSse::Ignored,
            })
        }
        "speech.audio.done" => Ok(ParsedSpeechSse::Done),
        _ => Ok(ParsedSpeechSse::Ignored),
    }
}

pub(crate) fn merge_header_metadata(headers: &reqwest::header::HeaderMap) -> Metadata {
    let mut meta = Metadata::new();
    if let Some(id) = headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
    {
        meta.insert(
            "x_request_id".to_owned(),
            Value::String(id.to_owned()),
        );
    }
    meta
}
