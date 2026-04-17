use serde_json::json;
use xlai_core::{TtsAudioFormat, TtsRequest, VoiceSpec};

use crate::synthesize_request_from_tts;

#[test]
fn maps_metadata_into_synthesize_request() {
    let req = TtsRequest {
        model: None,
        input: "hello".to_owned(),
        voice: VoiceSpec::Preset {
            name: "default".to_owned(),
        },
        response_format: Some(TtsAudioFormat::Wav),
        speed: None,
        instructions: None,
        delivery: Default::default(),
        metadata: [
            ("xlai.qts.temperature".to_owned(), json!(0.5)),
            ("xlai.qts.thread_count".to_owned(), json!(8)),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    let sr = synthesize_request_from_tts(&req).expect("map");
    assert_eq!(sr.text, "hello");
    assert!((sr.temperature - 0.5).abs() < f32::EPSILON);
    assert_eq!(sr.thread_count, 8);
}
