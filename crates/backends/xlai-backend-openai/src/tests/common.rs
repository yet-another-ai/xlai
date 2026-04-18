use crate::OpenAiConfig;

pub(super) fn test_config() -> OpenAiConfig {
    OpenAiConfig {
        base_url: "https://api.openai.com/v1".to_owned(),
        api_key: "k".to_owned(),
        model: "gpt-test".to_owned(),
        image_model: None,
        transcription_model: None,
        tts_model: None,
    }
}
